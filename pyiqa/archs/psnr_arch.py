r"""Peak signal-to-noise ratio (PSNR) Metric

Created by: https://github.com/photosynthesis-team/piq

Modified by: Jiadi Mo (https://github.com/JiadiMo)

Refer to:
    Wikipedia from https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    QIQA from https://github.com/francois-rozet/piqa/blob/master/piqa/psnr.py

"""

import torch
import torch.nn as nn

from pyiqa.utils.registry import ARCH_REGISTRY
from pyiqa.utils.color_util import to_y_channel


def psnr(x, y, test_y_channel=False, data_range=1.0, eps=1e-8, color_space='yiq'):
    r"""Compute Peak Signal-to-Noise Ratio for a batch of images.
    Supports both greyscale and color images with RGB channel order.
    Args:
        x: An input tensor. Shape :math:`(N, C, H, W)`.
        y: A target tensor. Shape :math:`(N, C, H, W)`.
        test_y_channel (Boolean): Convert RGB image to YCbCr format and computes PSNR
            only on luminance channel if `True`. Compute on all 3 channels otherwise.
        data_range: Maximum value range of images (default 1.0).

    Returns:
        PSNR Index of similarity betwen two images.
    """

    if (x.shape[1] == 3) and test_y_channel:
        # Convert RGB image to YCbCr and use Y-channel
        x = to_y_channel(x, data_range, color_space)
        y = to_y_channel(y, data_range, color_space)

    mse = torch.mean((x - y)**2, dim=[1, 2, 3])
    score = 10 * torch.log10(data_range**2 / (mse + eps))

    return score


@ARCH_REGISTRY.register()
class PSNR(nn.Module):
    r"""
    Args:
        X, Y (torch.Tensor): distorted image and reference image tensor with shape (B, 3, H, W)
        test_y_channel (Boolean): Convert RGB image to YCbCr format and computes PSNR
            only on luminance channel if `True`. Compute on all 3 channels otherwise.
        kwargs: other parameters, including
            - data_range: maximun numeric value
            - eps: small constant for numeric stability
    Return:
        score (torch.Tensor): (B, 1)
    """

    def __init__(self, test_y_channel=False, crop_border=0, **kwargs):
        super().__init__()
        self.test_y_channel = test_y_channel
        self.kwargs = kwargs
        self.crop_border = crop_border

    def forward(self, X, Y):
        assert X.shape == Y.shape, f'Input and reference images should have the same shape, but got {X.shape} and {Y.shape}'

        if self.crop_border != 0:
            crop_border = self.crop_border
            X = X[..., crop_border:-crop_border, crop_border:-crop_border]
            Y = Y[..., crop_border:-crop_border, crop_border:-crop_border]

        score = psnr(X, Y, self.test_y_channel, **self.kwargs)
        return score
