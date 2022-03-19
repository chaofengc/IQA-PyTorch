"""This folder contains pytorch implementations of matlab functions.
And should produce the same results as matlab.

Note: to enable GPU acceleration, all functions take batched tensors as inputs,
and return batched results.

"""
from .resize import imresize
from .functions import fspecial_gauss, imfilter, dct2d, conv2d, fitweibull
from .scfpyr_util import SCFpyr_PyTorch

__all__ = [
    'imresize',
    'fspecial_gauss',
    'SCFpyr_PyTorch',
    'imfilter',
    'dct2d',
    'conv2d',
    'fitweibull',
]
