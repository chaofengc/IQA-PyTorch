import numpy as np
import torch
import torch.nn.functional as F
from pyiqa.archs.arch_util import ExactPadding2d


def fspecial_gauss(size, sigma, channels=1):
    r""" Function same as 'fspecial gaussian' in MATLAB
    Args:
        size (int or tuple): size of window
        sigma (float): sigma of gaussian
        channels (int): channels of output
    """
    if type(size) is int:
        shape = (size, size)
    else:
        shape = size
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    h = torch.from_numpy(h).float().repeat(channels, 1, 1, 1)
    return h


def conv2d(input, weight, bias=None, stride=1, padding='same', dilation=1, groups=1):
    """Matlab like conv2, weights needs to be flipped.
    Args:
        input (tensor): (b, c, h, w)
        weight (tensor): (out_ch, in_ch, kh, kw), conv weight
        bias (bool or None): bias
        stride (int or tuple): conv stride
        padding (str): padding mode
        dilation (int): conv dilation
    """
    kernel_size = weight.shape[2:]
    pad_func = ExactPadding2d(kernel_size, stride, dilation, mode=padding)
    weight = torch.flip(weight, dims=(-1, -2))
    return F.conv2d(pad_func(input), weight, bias, stride, dilation=dilation, groups=groups)


def imfilter(input, weight, bias=None, stride=1, padding='same', dilation=1, groups=1):
    """imfilter same as matlab.
    Args:
        input (tensor): (b, c, h, w) tensor to be filtered
        weight (tensor): (out_ch, in_ch, kh, kw) filter kernel
        padding (str): padding mode
        dilation (int): dilation of conv
        groups (int): groups of conv
    """
    kernel_size = weight.shape[2:]
    pad_func = ExactPadding2d(kernel_size, stride, dilation, mode=padding)

    return F.conv2d(pad_func(input), weight, bias, stride, dilation=dilation, groups=groups)


def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    Args:
        x: the input signal
        norm: the normalization, None or 'ortho'
    Return:
        the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=-1)

    Vc = torch.view_as_real(torch.fft.fft(v, dim=-1))

    k = -torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
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


def fitweibull(x, iters=50, eps=1e-2):
    """Simulate wblfit function in matlab.

    ref: https://github.com/mlosch/python-weibullfit/blob/master/weibull/backend_pytorch.py

    Fits a 2-parameter Weibull distribution to the given data using maximum-likelihood estimation.
    :param x (tensor): (B, N), batch of samples from an (unknown) distribution. Each value must satisfy x > 0.
    :param iters: Maximum number of iterations
    :param eps: Stopping criterion. Fit is stopped ff the change within two iterations is smaller than eps.
    :param use_cuda: Use gpu
    :return: Tuple (Shape, Scale) which can be (NaN, NaN) if a fit is impossible.
        Impossible fits may be due to 0-values in x.
    """
    ln_x = torch.log(x)
    k = 1.2 / torch.std(ln_x, dim=1, keepdim=True)
    k_t_1 = k

    for t in range(iters):
        # Partial derivative df/dk
        x_k = x**k.repeat(1, x.shape[1])
        x_k_ln_x = x_k * ln_x
        ff = torch.sum(x_k_ln_x, dim=-1, keepdim=True)
        fg = torch.sum(x_k, dim=-1, keepdim=True)
        f1 = torch.mean(ln_x, dim=-1, keepdim=True)
        f = ff / fg - f1 - (1.0 / k)

        ff_prime = torch.sum(x_k_ln_x * ln_x, dim=-1, keepdim=True)
        fg_prime = ff
        f_prime = (ff_prime / fg - (ff / fg * fg_prime / fg)) + (1. / (k * k))

        # Newton-Raphson method k = k - f(k;x)/f'(k;x)
        k = k - f / f_prime
        error = torch.abs(k - k_t_1).max().item()
        if error < eps:
            break
        k_t_1 = k

    # Lambda (scale) can be calculated directly
    lam = torch.mean(x**k.repeat(1, x.shape[1]), dim=-1, keepdim=True)**(1.0 / k)

    return torch.cat((k, lam), dim=1)  # Shape (SC), Scale (FE)
