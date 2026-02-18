import os
import re
import cv2
import math
import time
import h5py
import glob
import torch
import Pypher
import random
import numpy as np
from scipy.sparse import linalg
from math import exp
from os.path import *
import torch.nn as nn
import scipy.io as sio
import torch.utils.data as tud
import torch.nn.functional as F
import torch.utils.data as data
import skimage.measure as measure
from torch.autograd import Variable
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as compare_ssim

def optset(opt):
    if opt.dataset == 'Cave':
        opt.hschannel = 31
        opt.mschannel = 3
        opt.trainset_imageX = 512
        opt.trainset_imageY = 512
        opt.trainset_file_num = 20
        opt.valset_sizeI = 512
        opt.valset_file_num = 12
        opt.testset_sizeI = 512
        opt.testset_file_num = 12
    elif opt.dataset == 'Harvard':
        opt.hschannel = 31
        opt.mschannel = 3
        opt.trainset_imageX = 1024
        opt.trainset_imageY = 1024
        opt.trainset_file_num = 30
        opt.valset_sizeI = 1024
        opt.valset_file_num = 20
        opt.testset_sizeI = 1024
        opt.testset_file_num = 20
    elif opt.dataset == 'Pavia':
        opt.hschannel = 102
        opt.mschannel = 4
        opt.trainset_imageX = 1024
        opt.trainset_imageY = 459
        opt.trainset_file_num = 1
        opt.valset_sizeI = 256
        opt.valset_file_num = 4
        opt.testset_sizeI = 256
        opt.testset_file_num = 4
    elif opt.dataset == 'Chikusei':
        opt.hschannel = 128
        opt.mschannel = 3
        opt.trainset_imageX = 512
        opt.trainset_imageY = 512
        opt.trainset_file_num = 10
        opt.valset_sizeI = 512
        opt.valset_file_num = 6
        opt.testset_sizeI = 512
        opt.testset_file_num = 6
    else:
        raise ValueError(f"Invalid dataset type: {opt.dataset}.")
    return opt


def dataparallel(model, ngpus, gpu0=0):
    if ngpus==0:
        assert False, "only support gpu mode"
    gpu_list = list(range(gpu0, gpu0+ngpus))
    assert torch.cuda.device_count() >= gpu0 + ngpus
    if ngpus > 1:
        if not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model, gpu_list).cuda()
        else:
            model = model.cuda()
    elif ngpus == 1:
        model = model.cuda()
    return model


def findLastCheckpoint(save_dir):
    file_list = glob.glob(join(save_dir, 'model_*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch


def prepare_data(opt, path, file_list, file_num, type):
    if type == "Train":
        imageX = opt.trainset_imageX
        imageY = opt.trainset_imageY
    elif type == "Val":
        imageX = opt.valset_sizeI
        imageY = opt.valset_sizeI
    elif type == "Test":
        imageX = opt.testset_sizeI
        imageY = opt.testset_sizeI
    else:
        raise ValueError(f"Invalid dataset type: {type}. Expected 'train', 'val' or 'test'.")

    HR_HSI = np.zeros((((imageX,imageY,opt.hschannel,file_num))))
    HR_MSI = np.zeros((((imageX,imageY,opt.mschannel,file_num))))
    
    for idx in range(file_num):
        ####  read HR-HSI
        HR_code = file_list[idx]
        path1 = os.path.join(path, 'HSI/') + HR_code + '.mat'
        data = sio.loadmat(path1)
        HR_HSI[:,:,:,idx] = data['X']

        ####  get HR-HRMS
        path2 = os.path.join(path, 'MSI/') + HR_code + '.mat'
        data = sio.loadmat(path2)
        HR_MSI[:,:,:,idx] = data['Y']
    return HR_HSI, HR_MSI



def loadpath(pathlistfile,shuffle=True):
    fp = open(pathlistfile)
    pathlist = fp.read().splitlines()
    fp.close()
    if shuffle==True:
        random.shuffle(pathlist)
    return pathlist


def para_setting(kernel_type,sf,sz,sigma):
    if kernel_type ==  'uniform_blur':
        psf = np.ones([sf,sf]) / (sf *sf)
    elif kernel_type == 'gaussian_blur':
        psf = np.multiply(cv2.getGaussianKernel(sf, sigma), (cv2.getGaussianKernel(sf, sigma)).T)

    fft_B = Pypher.psf2otf(psf, sz)
    fft_BT = np.conj(fft_B)
    return fft_B,fft_BT


class data_preparation_tensor(tud.Dataset):
    def __init__(self, opt, hrhsi, hrmsi, type):
        super(data_preparation_tensor, self).__init__()
        
        self.sigma = 2.0
        self.type = type
        self.sf = opt.sf
        self.hrhsi = hrhsi
        self.hrmsi = hrmsi
        self.sizeX, self.sizeY, self.channel, _ = hrhsi.shape

        if type == 'train':
            self.num = opt.trainset_num
            self.file_num = opt.trainset_file_num
            self.sizeI = opt.trainset_sizeI
        elif type == 'val':
            self.num = opt.valset_file_num
            self.sizeI = opt.valset_sizeI
        elif type == 'test':
            self.num = opt.testset_file_num
            self.sizeI = opt.testset_sizeI
        else:
            raise ValueError(f"Invalid dataset type: {type}. Expected 'train', 'val' or 'test'.")

    def H_z(self, z, factor, fft_B):
        f = torch.fft.fft2(z, dim=(-2, -1))
        f = torch.stack((f.real,f.imag),-1)
        # -------------------complex myltiply-----------------#
        if len(z.shape) == 3:
            ch, h, w = z.shape
            fft_B = fft_B.unsqueeze(0).repeat(ch, 1, 1, 1)
            M = torch.cat(((f[:, :, :, 0] * fft_B[:, :, :, 0] - f[:, :, :, 1] * fft_B[:, :, :, 1]).unsqueeze(3),
                           (f[:, :, :, 0] * fft_B[:, :, :, 1] + f[:, :, :, 1] * fft_B[:, :, :, 0]).unsqueeze(3)), 3)
            Hz = torch.irfft(M, 2, onesided=False)
            x = Hz[:, int(factor // 2)-1::factor, int(factor // 2)-1::factor]
        elif len(z.shape) == 4:
            bs, ch, h, w = z.shape
            fft_B = fft_B.unsqueeze(0).unsqueeze(0).repeat(bs, ch, 1, 1, 1)
            M = torch.cat(
                ((f[:, :, :, :, 0] * fft_B[:, :, :, :, 0] - f[:, :, :, :, 1] * fft_B[:, :, :, :, 1]).unsqueeze(4),
                 (f[:, :, :, :, 0] * fft_B[:, :, :, :, 1] + f[:, :, :, :, 1] * fft_B[:, :, :, :, 0]).unsqueeze(4)), 4)

            Hz = torch.fft.ifft2(torch.complex(M[..., 0],M[..., 1]), dim=(-2, -1))    
            x = Hz[:, :, int(factor // 2)-1::factor, int(factor // 2)-1::factor]
        return x.real

    def __getitem__(self, index):
        if self.type == 'train':
            index1   = random.randint(0, self.file_num-1)
        else:
            index1 = index
        
        hrhsi = self.hrhsi[:,:,:,index1]
        hrmsi = self.hrmsi[:,:,:,index1]

        sz = [self.sizeI, self.sizeI]
        fft_B, fft_BT = para_setting('gaussian_blur', self.sf, sz, self.sigma)
        fft_B = torch.cat((torch.Tensor(np.real(fft_B)).unsqueeze(2), torch.Tensor(np.imag(fft_B)).unsqueeze(2)),2)
        fft_BT = torch.cat((torch.Tensor(np.real(fft_BT)).unsqueeze(2), torch.Tensor(np.imag(fft_BT)).unsqueeze(2)), 2)

        px = random.randint(0, self.sizeX - self.sizeI)
        py = random.randint(0, self.sizeY - self.sizeI)
        hrhsi = hrhsi[px:px + self.sizeI:1, py:py + self.sizeI:1, :]
        hrmsi = hrmsi[px:px + self.sizeI:1, py:py + self.sizeI:1, :]

        if self.type == 'train':
            rotTimes = random.randint(0, 3)
            vFlip    = random.randint(0, 1)
            hFlip    = random.randint(0, 1)

            # Random rotation
            for j in range(rotTimes):
                hrhsi = np.rot90(hrhsi)
                hrmsi = np.rot90(hrmsi)

            # Random vertical Flip
            for j in range(vFlip):
                hrhsi = hrhsi[:, ::-1, :].copy()
                hrmsi = hrmsi[:, ::-1, :].copy()

            # Random Horizontal Flip
            for j in range(hFlip):
                hrhsi = hrhsi[::-1, :, :].copy()
                hrmsi = hrmsi[::-1, :, :].copy()

        hrhsi = torch.FloatTensor(hrhsi.copy()).permute(2,0,1).unsqueeze(0)
        hrmsi = torch.FloatTensor(hrmsi.copy()).permute(2,0,1).unsqueeze(0)
        lrhsi = self.H_z(hrhsi, self.sf, fft_B)
        lrhsi = torch.FloatTensor(lrhsi)

        hrhsi = hrhsi.squeeze(0)
        hrmsi = hrmsi.squeeze(0)
        lrhsi = lrhsi.squeeze(0)

        return hrhsi, hrmsi, lrhsi
    
    def __len__(self):
        return self.num


def qualityEvaluation1(opt, HRHS, result):
    psnr = PSNR(result.cpu().detach().numpy(), HRHS.numpy(), data_range=1.0)
    ssim = SSIM(result, HRHS.cuda())
    sam = SAM(result, HRHS)
    ergas = ERGAS(result, HRHS, opt.sf)
    return psnr, ssim, sam, ergas


def qualityEvaluation(opt, HRHS, result):
    psnr = compute_psnr(result, HRHS)
    ssim = compute_ssim(result, HRHS)
    sam = compute_sam(result, HRHS)
    ergas = compute_ergas(result, HRHS, opt.sf)
    return psnr, ssim, sam, ergas


def qualityEvaluationTotal(opt, psnr_total, ssim_total, sam_total, ergas_total, type='test'):
    if type == 'test':
        num = opt.testset_file_num
    elif type == 'val':
        num = opt.valset_file_num
    PSNR = psnr_total/num
    SSIM = ssim_total/num
    SAM = sam_total/num
    ERGAS = ergas_total/num 
    return PSNR, SSIM, SAM, ERGAS



def PSNR(im_true, im_test, data_range=None):
    im_true, im_test = _as_floats(im_true, im_test)

    err = compare_mse(im_true, im_test)
    return 10 * np.log10((data_range ** 2) / err)

def _as_floats(im1, im2):
    float_type = np.result_type(im1.dtype, im2.dtype, np.float32)
    im1 = np.asarray(im1, dtype=float_type)
    im2 = np.asarray(im2, dtype=float_type)
    return im1, im2

def compare_mse(im1, im2):
    im1, im2 = _as_floats(im1, im2)
    return np.mean(np.square(im1 - im2), dtype=np.float64)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def SSIM(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)

def SAM(im_true, im_fake):
    I_true = im_true.data.cpu().numpy()
    I_fake = im_fake.data.cpu().numpy()
    N = I_true.shape[0]
    C = I_true.shape[1]
    H = I_true.shape[2]
    W = I_true.shape[3]
    batch_sam = 0
    for i in range(N):
        true = I_true[i,:,:,:].reshape(C, H*W)
        fake = I_fake[i,:,:,:].reshape(C, H*W)
        nom = np.sum(np.multiply(true, fake), 0).reshape(H*W, 1)
        denom1 = np.sqrt(np.sum(np.square(true), 0)).reshape(H*W, 1)
        denom2 = np.sqrt(np.sum(np.square(fake), 0)).reshape(H*W, 1)
        # print(np.min(np.multiply(denom1,denom2)))
        sam = np.arccos(np.divide(nom,np.multiply(denom1,denom2))).reshape(H*W, 1)
        sam = sam/np.pi*180
        # ignore pixels that have zero norm
        idx = (np.isfinite(sam))
        batch_sam += np.sum(sam[idx])/np.sum(idx)
        if np.sum(~idx) != 0:
            print("waring: some values were ignored when computing SAM")
    return batch_sam/N


def ERGAS(img_fus, img_tgt, scale):
    img_tgt = img_tgt.squeeze(0).data.cpu().numpy()
    img_fus = img_fus.squeeze(0).data.cpu().numpy()
    img_tgt = np.squeeze(img_tgt)
    img_fus = np.squeeze(img_fus)
    img_tgt = img_tgt.reshape(img_tgt.shape[0], -1)
    img_fus = img_fus.reshape(img_fus.shape[0], -1)

    rmse = np.mean((img_tgt-img_fus)**2, axis=1)
    rmse = rmse**0.5
    mean = np.mean(img_tgt, axis=1)

    ergas = np.mean((rmse/mean)**2)
    ergas = 100/ scale *ergas**0.5

    return ergas


def compute_ssim(im1, im2):
    n = im1.shape[2]
    ms_ssim = 0.0
    for i in range(n):
        single_ssim = compare_ssim(im1[:, :, i], im2[:, :, i], data_range=1.0)
        ms_ssim += single_ssim
    return ms_ssim / n


def compute_psnr(im1, im2):
    num_spectral = im1.shape[-1]
    im1 = np.reshape(im1, (-1, num_spectral))
    im2 = np.reshape(im2, (-1, num_spectral))
    diff = im1 - im2

    mse = np.mean(np.square(diff), axis=0)

    return np.mean(10 * np.log10(1 / mse))


def compute_ergas(out, gt, scale):
    num_spectral = out.shape[-1]
    out = np.reshape(out, (-1, num_spectral))
    gt = np.reshape(gt, (-1, num_spectral))
    diff = gt - out
    mse = np.mean(np.square(diff), axis=0)
    gt_mean = np.mean(gt, axis=0)
    mse = np.reshape(mse, (num_spectral, 1))
    gt_mean = np.reshape(gt_mean, (num_spectral, 1))
    ergas = 100 / scale * np.sqrt(np.mean(mse / (gt_mean ** 2 + 1e-6)))
    return ergas


def compute_sam(im1, im2):
    num_spectral = im1.shape[-1]
    im1 = np.reshape(im1, (-1, num_spectral))
    im2 = np.reshape(im2, (-1, num_spectral))
    mole = np.sum(np.multiply(im1, im2), axis=1)
    im1_norm = np.sqrt(np.sum(np.square(im1), axis=1))
    im2_norm = np.sqrt(np.sum(np.square(im2), axis=1))
    deno = np.multiply(im1_norm, im2_norm)

    sam = np.rad2deg(np.arccos((mole) / (deno + 1e-7)))
    sam = np.mean(sam)
    return sam