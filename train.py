import time
import torch
import torch.backends.cudnn as cudnn
import torch.optim
from torch import nn
import option
from torch.nn import functional as F
from os.path import *
import torch.utils.data as data
from torch.utils.data import DataLoader
import random
import os
from torch.autograd import Variable
from utils import *
import torch.utils.data as tud
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
import time
from torch.autograd import Variable
import importlib
import logging
import scipy.io
from einops import rearrange
from thop import profile
from thop import clever_format

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def train():
    start_time = time.time()

    # Random seed
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    ## New model
    print(f"===> New Model {opt.model}")
    module_path = f'model.{opt.model}'
    model_module = importlib.import_module(module_path)
    Net = getattr(model_module, f'{opt.model}')
    model = Net(opt=opt)

    ## set the number of parallel GPUs
    print("===> Setting GPU")
    model = dataparallel(model, 1)
    logger.info(os.environ["CUDA_VISIBLE_DEVICES"])

    ## Initialize weight
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            nn.init.xavier_uniform_(layer.weight)
        if isinstance(layer, nn.ConvTranspose2d):
            nn.init.xavier_uniform_(layer.weight)

    ## Flops and Params
    inputx = torch.randn(1, opt.hschannel, opt.trainset_sizeI//opt.sf, opt.trainset_sizeI//opt.sf).cuda()
    inputy = torch.randn(1, opt.mschannel, opt.trainset_sizeI, opt.trainset_sizeI).cuda()
    flops, params = profile(model, inputs=(inputx, inputy))
    macs, params = clever_format([flops, params], "%.2f")
    logger.info(f"Model Flops: {macs}")
    logger.info(f"Model Params: {params}")

    ## Load training data
    print('===> Loading Datasets')
    file_path = join(opt.data_path, opt.dataset, 'Train')
    file_list = loadpath(join(file_path, 'Train.txt'))
    hrhsi_train, hrmsi_train = prepare_data(opt, file_path, file_list, opt.trainset_file_num, 'Train')
    file_path = join(opt.data_path, opt.dataset, 'Test')
    file_list = loadpath(join(file_path, 'Test.txt'))
    hrhsi_val, hrmsi_val = prepare_data(opt, file_path, file_list, opt.testset_file_num, 'Test')

    ## Load trained model
    save_path = f'{opt.save_path}/{opt.dataset}/f{opt.sf}/{opt.model}/model'
    os.makedirs(save_path, exist_ok=True)
    initial_epoch = findLastCheckpoint(save_path)
    if initial_epoch > 0:
        logger.info(f'resuming by loading epoch {initial_epoch:04d}')
        model = torch.load(os.path.join(save_path, f'model_{initial_epoch:04d}.pth'))

    ## Loss function
    criterion = nn.L1Loss()

    ## Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.epsilon)
    scheduler = MultiStepLR(optimizer, milestones=list(range(1,150,5)), gamma=0.95)

    ## pipline of training
    best_psnr = 0
    best_epoch = 0
    for epoch in range(initial_epoch, opt.epochs):
        epoch_start_time = time.time()
        ## train
        model.train()

        data = data_preparation_tensor(opt, hrhsi_train, hrmsi_train, 'train')
        loader_train = tud.DataLoader(data, num_workers=opt.num_workers, batch_size=opt.batch_size, shuffle=True)

        scheduler.step(epoch)
        epoch_loss = 0

        train_start_time = time.time()
        for i, (HRHS, HRMS, LRHS) in enumerate(loader_train):

            out = model(LRHS.cuda(), HRMS.cuda())

            loss = criterion(out, HRHS.cuda())
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % 100 == 0:
                logger.info(f'[Epcoh:{epoch+1} {(i+1)}/{len(data)//opt.batch_size}]    Loss:{(epoch_loss/((i+1)*opt.batch_size)):.6e}')

        train_elapsed_time = time.time() - train_start_time
        logger.info(f'Train_Time:{train_elapsed_time:.1f}s')

        ## val
        val_start_time = time.time()
        data = data_preparation_tensor(opt, hrhsi_val, hrmsi_val, 'val')
        loader_val = tud.DataLoader(data, batch_size=1)
        model = model.eval()

        psnr_total = 0
        sam_total = 0
        ergas_total = 0
        ssim_total = 0
        for j, (HRHS, HRMS, LRHS) in enumerate(loader_val):
            with torch.no_grad():
                out = reconstruction(opt, model, LRHS.cuda(), HRMS.cuda(), opt.trainset_sizeI, 32)

            out = out.cpu().data.squeeze().clamp(0, 1).numpy().transpose(1,2,0)
            HR = HRHS.cpu().data.squeeze().clamp(0, 1).numpy().transpose(1,2,0)

            psnr, ssim, sam, ergas = qualityEvaluation(opt, HR, out)
            psnr_total = psnr_total + psnr
            ssim_total = ssim_total + ssim
            sam_total = sam_total + sam
            ergas_total = ergas_total + ergas

        PSNR, SSIM, SAM, ERGAS = qualityEvaluationTotal(opt, psnr_total, ssim_total, sam_total, ergas_total, type='val')
        if best_psnr < PSNR:
            best_epoch = epoch+1
            best_psnr = PSNR
        val_elapsed_time = time.time() - val_start_time
        torch.save(model, f'{save_path}/model_{epoch+1:04d}.pth')

        epoch_elapsed_time = time.time() - epoch_start_time
        elapsed_time = time.time() - start_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)

        logger.info(f'{opt.model}_{opt.dataset}_×{opt.sf}_val')
        logger.info(f'Epcoh:{epoch+1}    Loss:{((epoch_loss)/len(data)):.6e}    PSNR:{PSNR:.2f}    SSIM:{SSIM:.4f}    SAM:{SAM:.2f}    ERGAS:{ERGAS:.2f}    (Best:{best_psnr:.2f}  Epcoh:{best_epoch})')
        logger.info(f'Train_Time:{train_elapsed_time:.1f}s    Val_Time:{val_elapsed_time:.1f}s    Epoch_Time:{epoch_elapsed_time:.1f}s    Total_Time:{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}')


def test():
    test_start_time = time.time()

    ## Load testing data
    print('===> Loading Test Datasets')
    file_path = join(opt.data_path, opt.dataset, 'Test')
    file_list = loadpath(join(file_path, 'Test.txt'))
    hrhsi_test, hrmsi_test = prepare_data(opt, file_path, file_list, opt.testset_file_num, 'Test')
    
    data = data_preparation_tensor(opt, hrhsi_test, hrmsi_test, 'test')
    loader_test = tud.DataLoader(data, batch_size=1)

    save_path = f'{opt.save_path}/{opt.dataset}/f{opt.sf}/{opt.model}/model'
    model = torch.load(os.path.join(save_path, f'model_{opt.test_model_epoch:04d}.pth'))
    model = model.cuda().eval()

    # test
    psnr_total = 0
    sam_total = 0
    ergas_total = 0
    ssim_total = 0
    for j, (HRHS, HRMS, LRHS) in enumerate(loader_test):
        with torch.no_grad():
            out = reconstruction(opt, model, LRHS.cuda(), HRMS.cuda(), opt.trainset_sizeI, 32)


        out = out.cpu().data.squeeze().clamp(0, 1).numpy().transpose(1,2,0)
        HR = HRHS.cpu().data.squeeze().clamp(0, 1).numpy().transpose(1,2,0)


        save_path = f'{opt.save_path}/{opt.dataset}/f{opt.sf}/{opt.model}/result/{opt.model}_{j}.mat'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        scipy.io.savemat(save_path, {'X': HR})


        psnr, ssim, sam, ergas = qualityEvaluation(opt, HR, out)
        psnr_total = psnr_total + psnr
        ssim_total = ssim_total + ssim
        sam_total = sam_total + sam
        ergas_total = ergas_total + ergas

    PSNR, SSIM, SAM, ERGAS = qualityEvaluationTotal(opt, psnr_total, ssim_total, sam_total, ergas_total, type='test')
    test_elapsed_time = time.time() - test_start_time
    logger.info(f'{opt.model}_{opt.dataset}_×{opt.sf}_test    PSNR:{PSNR:.2f}    SSIM:{SSIM:.4f}    SAM:{SAM:.2f}    ERGAS:{ERGAS:.2f}    Test_Time:{test_elapsed_time:.1f}s\n')


if __name__ == '__main__':
    # Load opt
    print("Load Opt:")
    opt = option.args_parser()
    opt = optset(opt)

    # Set logging
    log_dir = f'{opt.save_path}/{opt.dataset}/f{opt.sf}/{opt.model}'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'{opt.model}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()
    logger.info(opt)

    # PrepareData(opt)
    train()
    test()
