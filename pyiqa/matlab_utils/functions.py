import numpy as np
import torch
import torch.nn.functional as F
from pyiqa.archs.arch_util import ExactPadding2d


def fspecial_gauss(size, sigma, channels=1):
    r""" Function same as 'fspecial gaussian' in MATLAB 
    """
    if type(size) is int:
        shape = (size, size)
    else:
        shape = size
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma) )
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    h = torch.from_numpy(h).float().repeat(channels, 1, 1, 1)
    return h 


def conv2d(input, weight, bias=None, stride=1, padding='same', dilation=1, groups=1):
    """matlab like conv2, weights needs to be reversed 
    """
    kernel_size = weight.shape[2:]
    pad_func = ExactPadding2d(kernel_size, stride, dilation, mode=padding)
    weight = torch.flip(weight, dims=(-1, -2)) 
    return F.conv2d(
        pad_func(input), weight, bias, stride, dilation=dilation, groups=groups)


def imfilter(input, weight, bias=None, stride=1, padding='same', dilation=1, groups=1):
    """imfilter same as matlab.
    """
    kernel_size = weight.shape[2:]
    pad_func = ExactPadding2d(kernel_size, stride, dilation, mode=padding)
        
    return F.conv2d(
        pad_func(input), weight, bias, stride, dilation=dilation, groups=groups)


def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=-1)

    Vc = torch.view_as_real(torch.fft.fft(v, dim=-1))

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def dct2d(x, norm='ortho'):
    """
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)