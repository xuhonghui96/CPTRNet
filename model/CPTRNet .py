import torch
from torch import nn
import torch.nn.functional as F
from functools import lru_cache
from torch.nn.functional import conv2d


class KAGNConv(nn.Module):
    def __init__(self, conv_class, norm_class, conv_w_fun, input_dim, output_dim, degree, kernel_size,
                 groups=1, padding=0, stride=1, dilation=1, dropout: float = 0.0, ndim: int = 2.,
                 **norm_kwargs):
        super(KAGNConv, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.base_activation = nn.SiLU()
        self.conv_w_fun = conv_w_fun
        self.ndim = ndim
        self.dropout = None
        self.norm_kwargs = norm_kwargs
        self.p_dropout = dropout
        if dropout > 0:
            if ndim == 1:
                self.dropout = nn.Dropout1d(p=dropout)
            if ndim == 2:
                self.dropout = nn.Dropout2d(p=dropout)
            if ndim == 3:
                self.dropout = nn.Dropout3d(p=dropout)

        if groups <= 0:
            raise ValueError('groups must be a positive integer')
        if input_dim % groups != 0:
            raise ValueError('input_dim must be divisible by groups')
        if output_dim % groups != 0:
            raise ValueError('output_dim must be divisible by groups')

        self.base_conv = conv_class(input_dim,
                                    output_dim,
                                    kernel_size,
                                    stride,
                                    padding,
                                    dilation,
                                    groups=groups,
                                    bias=False)

        self.layer_norm = norm_class(output_dim, **norm_kwargs)

        self.poly_conv = conv_class(input_dim * (degree + 1),
                                    output_dim,
                                    kernel_size,
                                    stride,
                                    padding,
                                    dilation,
                                    groups=groups,
                                    bias=False)

        self.beta_weights = nn.Parameter(torch.zeros(degree + 1, dtype=torch.float32))
        
        nn.init.kaiming_uniform_(self.base_conv.weight, nonlinearity='linear')
        nn.init.kaiming_uniform_(self.poly_conv.weight, nonlinearity='linear')

        nn.init.normal_(
            self.beta_weights,
            mean=0.0,
            std=1.0 / ((kernel_size ** ndim) * self.inputdim * (self.degree + 1.0)),
        )

    def beta(self, n, m):
        return (
                       ((m + n) * (m - n) * n ** 2) / (m ** 2 / (4.0 * n ** 2 - 1.0))
               ) * self.beta_weights[n]

    @lru_cache(maxsize=128)
    def gram_poly(self, x, degree):
        p0 = x.new_ones(x.size())

        if degree == 0:
            return p0.unsqueeze(-1)

        p1 = x
        grams_basis = [p0, p1]

        for i in range(2, degree + 1):
            p2 = x * p1 - self.beta(i - 1, i) * p0
            grams_basis.append(p2)
            p0, p1 = p1, p2

        indexes = [i*(degree + 1) + j for i in range(x.shape[1]) for j in range(degree + 1)]

        grams_basis = torch.concatenate(grams_basis, dim=1)
        grams_basis = grams_basis[:, indexes]
        return grams_basis

    def forward_kag(self, x):
        basis = self.base_conv(self.base_activation(x))
        x = torch.tanh(x).contiguous()

        if self.dropout is not None:
            x = self.dropout(x)

        grams_basis = self.base_activation(self.gram_poly(x, self.degree))
        y = self.poly_conv(grams_basis)
        y = self.base_activation(self.layer_norm(y + basis))
        return y

    def forward(self, x):
        return self.forward_kag(x)

class KAGNConv2D(KAGNConv):
    def __init__(self, input_dim, output_dim, kernel_size, degree=3, groups=1, padding=0, stride=1, dilation=1,
                 dropout: float = 0.0, norm_layer=nn.InstanceNorm2d, **norm_kwargs):
        super(KAGNConv2D, self).__init__(nn.Conv2d, norm_layer, conv2d,
                                              input_dim, output_dim,
                                              degree, kernel_size,
                                              groups=groups, padding=padding, stride=stride, dilation=dilation,
                                              ndim=2, dropout=dropout, **norm_kwargs)


class EnhanceBlock(nn.Module):
    def __init__(self, c_in):
        super(EnhanceBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(c_in, c_in, kernel_size=(1, 1), bias=False),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(c_in, c_in // 2, kernel_size=(3, 3),
                      stride=(1, 1), padding=1, bias=False),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(c_in // 2, c_in // 2, kernel_size=(1, 1), bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
            nn.Conv2d(c_in // 2, c_in, kernel_size=(1, 1), bias=False),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out + x
        out = F.leaky_relu(out, negative_slope=0.2, inplace=False)
        return out


class KVG(nn.Module):
    def __init__(self, num_direction, reduction):
        super(KVG, self).__init__()
        self.conv = nn.Conv2d(num_direction, 1, (1, 1), bias=False)
        self.softmax = nn.Softmax(dim=2)
        self.fc = nn.Sequential(
            nn.Linear(num_direction, num_direction // reduction, bias=False),
            nn.PReLU(),
            nn.Linear(num_direction // reduction, num_direction, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        input_x = x
        input_x = input_x.view(b, c, h * w).unsqueeze(1)
        mask = self.conv(x).view(b, 1, h * w)
        mask = self.softmax(mask).unsqueeze(-1)
        y = torch.matmul(input_x, mask).view(b, c)
        return self.fc(y)


class GKAN(nn.Module):
    def __init__(self, c_in, w_in, h_in):
        super(GKAN, self).__init__()
        self.msa_c = nn.Sequential(
            KAGNConv2D(input_dim=c_in, output_dim=c_in, kernel_size=1)
        )
        self.msa_h = nn.Sequential(
            KAGNConv2D(input_dim=c_in, output_dim=c_in, kernel_size=1)
        )
        self.msa_w = nn.Sequential(
            KAGNConv2D(input_dim=c_in, output_dim=c_in, kernel_size=1)
        )
        self.conv_c = nn.Sequential(
            nn.Conv2d(c_in, c_in, (1, 1), bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
        )
        self.conv_w = nn.Sequential(
            nn.Conv2d(w_in, w_in, (1, 1), bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
        )
        self.conv_h = nn.Sequential(
            nn.Conv2d(h_in, h_in, (1, 1), bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
        )

        self.layernorm_c0 = nn.LayerNorm(c_in)
        self.layernorm_w0 = nn.LayerNorm(w_in)
        self.layernorm_h0 = nn.LayerNorm(h_in)

        self.layernorm_c = nn.LayerNorm(c_in)
        self.layernorm_w = nn.LayerNorm(w_in)
        self.layernorm_h = nn.LayerNorm(h_in)

        self.rtgb_c1 = KVG(c_in, 2)
        self.rtgb_h1 = KVG(h_in, 2)
        self.rtgb_w1 = KVG(w_in, 2)

        self.conv = nn.Sequential(
            nn.Conv2d(c_in * 3, c_in, (3, 3), padding=1),
        )

    def forward(self, x_in):
        x_c = x_in
        x_w = x_in.permute(0, 2, 1, 3)
        x_h = x_in.permute(0, 3, 2, 1)
        out_c = self.msa_c(x_c)
        out_h = self.msa_h(x_h)
        out_w = self.msa_w(x_w)
        out_c = self.conv_c(out_c + x_c)
        out_h = self.conv_h(out_h + x_h)
        out_w = self.conv_w(out_w + x_w)
        vector_c = out_c.permute(0, 1, 3, 2)
        vector_h = out_h.permute(0, 3, 2, 1)
        vector_w = out_w.permute(0, 2, 1, 3)
        o_all = torch.cat((vector_c, vector_h,vector_w), dim=1)
        out = self.conv(o_all)
        return out


class RTKAN(nn.Module):
    def __init__(self, c_in, w_in, h_in, heads_num, rtgb_num):
        super(RTKAN, self).__init__()
        self.rtgb1 = GKAN(c_in, w_in, h_in, heads_num)
        self.rtgb2 = GKAN(c_in, w_in, h_in, heads_num)
        self.rtgb3 = GKAN(c_in, w_in, h_in, heads_num)
        self.conv = nn.Sequential(
            nn.Conv2d(c_in * rtgb_num, c_in, (3, 3), padding=1),
        )

    def forward(self, x):
        o1 = self.rtgb1(x)
        o2 = self.rtgb2(x - o1)
        o3 = self.rtgb3(x - o1 - o2)
        o_all = torch.cat((o1, o2,o3), dim=1)
        output = self.conv(o_all)
        return output


class CP_Model(nn.Module):
    def __init__(self, w_in, h_in, cp_in_c, heads_num, rtgb_num):
        super(CP_Model, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cp_in_c, cp_in_c, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
            nn.Conv2d(cp_in_c, cp_in_c, kernel_size=(3, 3), stride=(1, 1), padding=1),
        )
        self.transformer_DRTLM = RTKAN(cp_in_c, w_in, h_in, heads_num, rtgb_num)
        self.resnet = EnhanceBlock(cp_in_c)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.transformer_DRTLM(x1)
        out = x2 * x1
        out = self.resnet(out)
        return out


class CPTRNet(nn.Module):
    def __init__(self, opt):
        super(CPTRNet, self).__init__()
        self.sf = opt.sf
        c_in=opt.hschannel+opt.mschannel
        w_in=64 
        h_in=64
        cp_in_c=64
        c_out=opt.hschannel
        heads_num=4
        rtgb_num=3
        self.stage = 1
        self.conv0 = nn.Conv2d(c_in, cp_in_c, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv00 = nn.Conv2d(c_in, c_out, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.cp1 = CP_Model(w_in , h_in , cp_in_c, heads_num, rtgb_num)
        self.bottleneck1 = nn.Sequential(
            nn.Conv2d(cp_in_c, cp_in_c, (4, 4), (2, 2), 1, bias=False),
            nn.Conv2d(cp_in_c, cp_in_c, (1, 1), (1, 1), bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
            nn.Conv2d(cp_in_c, cp_in_c, (3, 3), (1, 1), padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
            nn.ConvTranspose2d(cp_in_c, cp_in_c, stride=2, kernel_size=2, padding=0, output_padding=0)
        )

        self.fusion1 = nn.Conv2d(cp_in_c*2, cp_in_c, (1, 1), (1, 1), bias=False)
        self.cp2 = CP_Model(w_in , h_in , cp_in_c, heads_num, rtgb_num)

        self.conv_up1 = nn.ConvTranspose2d(cp_in_c, cp_in_c, stride=2, kernel_size=2, padding=0, output_padding=0)
        self.conv_up2 = nn.ConvTranspose2d(cp_in_c, cp_in_c, stride=2, kernel_size=2, padding=0, output_padding=0)

        self.conv1 = nn.Sequential(
            nn.Conv2d(cp_in_c, cp_in_c, (3, 3), (1, 1), 1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
            nn.Conv2d(cp_in_c, cp_in_c, (3, 3), (1, 1), 1, bias=False),
            nn.Conv2d(cp_in_c, cp_in_c, (3, 3), (1, 1), 1, bias=False),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(cp_in_c, cp_in_c, (3, 3), (1, 1), 1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
            nn.Conv2d(cp_in_c, cp_in_c, (3, 3), (1, 1), 1, bias=False),
            nn.Conv2d(cp_in_c, c_out, (3, 3), (1, 1), 1, bias=False),
        )
        self.conv1_fu = nn.Conv2d(cp_in_c, cp_in_c, (1, 1), (1, 1), bias=False)
        self.conv1_fu2 = nn.Conv2d(cp_in_c, cp_in_c, (1, 1), (1, 1), bias=False)
        self.conv1_fu3 = nn.Conv2d(cp_in_c, cp_in_c, (1, 1), (1, 1), bias=False)

    def forward(self, x, y):
        x = F.interpolate(x, scale_factor=self.sf, mode='bicubic', align_corners=False)
        x = torch.cat((x, y),dim=1)
        x0 = self.conv0(x)
        fea1 = self.cp1(x0)
        fea = self.bottleneck1(fea1)
        fea = self.fusion1(torch.cat([fea, fea1], dim=1))
        fea2 = self.cp2(fea)
        fea = self.conv1_fu3(fea2)
        temp_x1 = self.conv1_fu2(fea1)
        out = self.conv1(fea + temp_x1)
        temp_x0 = self.conv1_fu(x0)
        out = self.conv2(temp_x0 + out)
        out1 = self.conv00(x)
        out = out1 + out
        return out