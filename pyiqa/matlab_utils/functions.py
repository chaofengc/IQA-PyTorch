import math
import numpy as np
import torch
import torch.nn.functional as F
from pyiqa.archs.arch_util import ExactPadding2d, to_2tuple, symm_pad


def fspecial(size=None, sigma=None, channels=1, filter_type='gaussian'):
    r""" Function same as 'fspecial' in MATLAB, only support gaussian now.
    Args:
        size (int or tuple): size of window
        sigma (float): sigma of gaussian
        channels (int): channels of output
    """
    if filter_type == 'gaussian':
        shape = to_2tuple(size)
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        h = torch.from_numpy(h).float().repeat(channels, 1, 1, 1)
        return h
    else:
        raise NotImplementedError(f'Only support gaussian filter now, got {filter_type}')


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


def filter2(input, weight, shape='same'):
    if shape == 'same':
        return imfilter(input, weight, groups=input.shape[1])
    elif shape == 'valid':
        return F.conv2d(input, weight, stride=1, padding=0, groups=input.shape[1])
    else:
        raise NotImplementedError(f'Shape type {shape} is not implemented.')


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


def cov(tensor, rowvar=True, bias=False):
    r"""Estimate a covariance matrix (np.cov)
    Ref: https://gist.github.com/ModarTensai/5ab449acba9df1a26c12060240773110
    """
    tensor = tensor if rowvar else tensor.transpose(-1, -2)
    tensor = tensor - tensor.mean(dim=-1, keepdim=True)
    factor = 1 / (tensor.shape[-1] - int(not bool(bias)))
    return factor * tensor @ tensor.transpose(-1, -2)


def nancov(x):
    r"""Calculate nancov for batched tensor, rows that contains nan value 
    will be removed.

    Args:
        x (tensor): (B, row_num, feat_dim)  

    Return:
        cov (tensor): (B, feat_dim, feat_dim)
    """
    assert len(x.shape) == 3, f'Shape of input should be (batch_size, row_num, feat_dim), but got {x.shape}'
    b, rownum, feat_dim = x.shape
    nan_mask = torch.isnan(x).any(dim=2, keepdim=True)
    cov_x = []
    for i in range(b):
        x_no_nan = x[i].masked_select(~nan_mask[i]).reshape(-1, feat_dim)
        cov_x.append(cov(x_no_nan, rowvar=False))
    return torch.stack(cov_x)


def nanmean(v, *args, inplace=False, **kwargs):
    r"""nanmean same as matlab function: calculate mean values by removing all nan.
    """
    if not inplace:
        v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0
    return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)


def im2col(x, kernel, mode='sliding'):
    r"""simple im2col as matlab

    Args:
        x (Tensor): shape (b, c, h, w)
        kernel (int): kernel size
        mode (string): 
            - sliding (default): rearranges sliding image neighborhoods of kernel size into columns with no zero-padding
            - distinct: rearranges discrete image blocks of kernel size into columns, zero pad right and bottom if necessary
    Return:
        flatten patch (Tensor): (b, h * w / kernel **2, kernel * kernel)
    """
    b, c, h, w = x.shape
    kernel = to_2tuple(kernel)

    if mode == 'sliding':
        stride = 1
    elif mode == 'distinct':
        stride = kernel
        h2 = math.ceil(h / stride[0])
        w2 = math.ceil(w / stride[1])
        pad_row = (h2 - 1) * stride[0] + kernel[0] - h
        pad_col = (w2 - 1) * stride[1] + kernel[1] - w
        x = F.pad(x, (0, pad_col, 0, pad_row))
    else:
        raise NotImplementedError(f'Type {mode} is not implemented yet.')

    patches = F.unfold(x, kernel, dilation=1, stride=stride)
    b, _, pnum = patches.shape
    patches = patches.transpose(1, 2).reshape(b, pnum, -1)
    return patches


def blockproc(x, kernel, fun, border_size=None, pad_partial=False, pad_method='zero', **func_args):
    r"""blockproc function like matlab

    Difference:
        - Partial blocks is discarded (if exist) for fast GPU process.

    Args:
        x (tensor): shape (b, c, h, w)
        kernel (int or tuple): block size
        func (function): function to process each block
        border_size (int or tuple): border pixels to each block
        pad_partial: pad partial blocks to make them full-sized, default False
        pad_method: [zero, replicate, symmetric] how to pad partial block when pad_partial is set True

    Return:
        results (tensor): concatenated results of each block
    """
    assert len(x.shape) == 4, f'Shape of input has to be (b, c, h, w) but got {x.shape}'
    kernel = to_2tuple(kernel)
    if pad_partial:
        b, c, h, w = x.shape
        stride = kernel
        h2 = math.ceil(h / stride[0])
        w2 = math.ceil(w / stride[1])
        pad_row = (h2 - 1) * stride[0] + kernel[0] - h
        pad_col = (w2 - 1) * stride[1] + kernel[1] - w
        padding = (0, pad_col, 0, pad_row)
        if pad_method == 'zero':
            x = F.pad(x, padding, mode='constant')
        elif pad_method == 'symmetric':
            x = symm_pad(x, padding)
        else:
            x = F.pad(x, padding, mode=pad_method)

    if border_size is not None:
        raise NotImplementedError('Blockproc with border is not implemented yet')
    else:
        b, c, h, w = x.shape
        block_size_h, block_size_w = kernel
        num_block_h = math.floor(h / block_size_h)
        num_block_w = math.floor(w / block_size_w)

        # extract blocks in (row, column) manner, i.e., stored with column first
        blocks = F.unfold(x, kernel, stride=kernel)
        blocks = blocks.reshape(b, c, *kernel, num_block_h, num_block_w)
        blocks = blocks.permute(5, 4, 0, 1, 2, 3).reshape(num_block_h * num_block_w * b, c, *kernel)

        results = fun(blocks, func_args)
        results = results.reshape(num_block_h * num_block_w, b, *results.shape[1:]).transpose(0, 1)
        return results
