r"""Mathematical utilities

Created by: https://github.com/tomrunia/PyTorchSteerablePyramid/blob/master/steerable/math_utils.py

Modified by: Jiadi Mo (https://github.com/JiadiMo)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch


def abs(x):
    return torch.sqrt(x[..., 0]**2 + x[..., 1]**2 + 1e-12)


def roll_n(X, axis, n):
    f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None) for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None) for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)


def batch_fftshift2d(x):
    '''Args:
        x: An complex tensor. Shape :math:`(N, C, H, W)`.
        Pytroch version >= 1.8.0
    '''
    real, imag = x.real, x.imag
    for dim in range(1, len(real.size())):
        n_shift = real.size(dim) // 2
        if real.size(dim) % 2 != 0:
            n_shift += 1  # for odd-sized images
        real = roll_n(real, axis=dim, n=n_shift)
        imag = roll_n(imag, axis=dim, n=n_shift)
    return torch.stack((real, imag), -1)  # last dim=2 (real&imag)


def batch_ifftshift2d(x):
    '''Args:
        x: An input tensor. Shape :math:`(N, C, H, W, 2)`.
    Return:
        An complex tensor. Shape :math:`(N, C, H, W)`.
    '''
    real, imag = torch.unbind(x, -1)
    for dim in range(len(real.size()) - 1, 0, -1):
        real = roll_n(real, axis=dim, n=real.size(dim) // 2)
        imag = roll_n(imag, axis=dim, n=imag.size(dim) // 2)
    return torch.complex(real, imag)  # convert to complex (real&imag)


def prepare_grid(m, n):
    x = np.linspace(-(m // 2) / (m / 2), (m // 2) / (m / 2) - (1 - m % 2) * 2 / m, num=m)
    y = np.linspace(-(n // 2) / (n / 2), (n // 2) / (n / 2) - (1 - n % 2) * 2 / n, num=n)
    xv, yv = np.meshgrid(y, x)
    angle = np.arctan2(yv, xv)
    rad = np.sqrt(xv**2 + yv**2)
    rad[m // 2][n // 2] = rad[m // 2][n // 2 - 1]
    log_rad = np.log2(rad)
    return log_rad, angle


def rcosFn(width, position):
    N = 256  # abritrary
    X = np.pi * np.array(range(-N - 1, 2)) / 2 / N
    Y = np.cos(X)**2
    Y[0] = Y[1]
    Y[N + 2] = Y[N + 1]
    X = position + 2 * width / np.pi * (X + np.pi / 4)
    return X, Y


def pointOp(im, Y, X):
    out = np.interp(im.flatten(), X, Y)
    return np.reshape(out, im.shape)


def getlist(coeff):
    straight = [bands for scale in coeff[1:-1] for bands in scale]
    straight = [coeff[0]] + straight + [coeff[-1]]
    return straight
