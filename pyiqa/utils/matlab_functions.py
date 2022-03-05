import math
import numpy as np
import torch
from .resize import imresize

__all__ = ['imresize', 'fspecial_gauss']

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

