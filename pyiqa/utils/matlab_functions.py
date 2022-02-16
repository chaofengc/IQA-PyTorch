import math
import numpy as np
import torch
from .resize import imresize

__all__ = ['imresize', 'fspecial_gauss']

def fspecial_gauss(size, sigma, channels):
    r""" Function same as 'fspecial gaussian' in MATLAB 
    """
    if type(size) is int:
        x, y = np.mgrid[-size // 2 + 1:size // 2 + 1,
                        -size // 2 + 1:size // 2 + 1]
    else:
        x, y = np.mgrid[-size[0] // 2 + 1:size[0] // 2 + 1,
                        -size[1] // 2 + 1:size[1] // 2 + 1]
    g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
    g = torch.from_numpy(g / g.sum()).float().unsqueeze(0).unsqueeze(0)
    return g.repeat(channels, 1, 1, 1)

