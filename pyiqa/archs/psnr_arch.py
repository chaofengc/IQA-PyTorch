r"""SSIM Metric

Created by: https://github.com/photosynthesis-team/piq

Modified by: Jiadi Mo (https://github.com/JiadiMo)

Refer to:
    Wikipedia from https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    QIQA from https://github.com/francois-rozet/piqa/blob/master/piqa/psnr.py

"""

import torch
import torch.nn as nn

from pyiqa.utils.registry import ARCH_REGISTRY
from pyiqa.utils.color_util import rgb2ycbcr


def psnr(x, y, test_y_channel=False, data_range=1.0):
    r"""Compute Peak Signal-to-Noise Ratio for a batch of images.
    Supports both greyscale and color images with RGB channel order.
    Args:
        x: An input tensor. Shape :math:`(N, C, H, W)`.
        y: A target tensor. Shape :math:`(N, C, H, W)`.
        test_y_channel: Convert RGB image to YCbCr format and computes PSNR
            only on luminance channel if `True`. Compute on all 3 channels otherwise.
        data_range: Maximum value range of images (default 1.0).
    Returns:
        PSNR Index of similarity betwen two images.
    """
    # Constant for numerical stability
    EPS = 1e-8

    if (x.size(1) == 3) and test_y_channel:
        # Convert RGB image to YCbCr and use Y-channel
        x = rgb2ycbcr(x)[:, 0, :, :].unsqueeze(1)
        y = rgb2ycbcr(y)[:, 0, :, :].unsqueeze(1)

        data_range = 255.

    mse = torch.mean((x - y)**2, dim=[1, 2, 3])
    score = 10 * torch.log10(data_range**2 / (mse + EPS))

    return score


@ARCH_REGISTRY.register()
class PSNR(nn.Module):
    r"""Args:
        test_y_channel: Convert RGB image to YCbCr format and computes PSNR
            only on luminance channel if `True`. Compute on all 3 channels otherwise.
        `**kwargs` are transmitted to `psnr`.
    """

    def __init__(self, test_y_channel=False, **kwargs):

        super().__init__()
        self.test_y_channel = test_y_channel
        self.kwargs = kwargs

    def forward(self, X, Y):
        assert X.shape == Y.shape, f"Input and reference images should have the same shape, but got {X.shape} and {Y.shape}"
        score = psnr(X, Y, self.test_y_channel, **self.kwargs)
        return score
