r"""MS-SSIM Metric

Created by: https://github.com/dingkeyan93/IQA-optimization/blob/master/IQA_pytorch/MS_SSIM.py

Modified by: Jiadi Mo (https://github.com/JiadiMo)

Refer to:
    Matlab code from https://ece.uwaterloo.ca/~z70wang/research/iwssim/msssim.zip; 

"""

import torch
from torch.nn import functional as F

from pyiqa.archs.ssim_arch import ssim, to_y_channel, fspecial_gauss
from pyiqa.utils.registry import ARCH_REGISTRY


def ms_ssim(X, Y, win, data_range=1., downsample=False, test_y_channel=True):
    r"""Compute Multiscale structural similarity for a batch of images.
    Args:
        x: An input tensor. Shape :math:`(N, C, H, W)`.
        y: A target tensor. Shape :math:`(N, C, H, W)`.
        win: Window setting.
        downsample: Boolean, whether to downsample which mimics official SSIM matlab code.
        test_y_channel: Boolean, whether to use y channel on ycbcr.
    Returns:
        Index of similarity betwen two images. Usually in [0, 1] interval.
    """
    if not X.shape == Y.shape:
        raise ValueError('Input images must have the same dimensions.')

    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363,
                                 0.1333]).to(X.device, dtype=X.dtype)

    levels = weights.shape[0]
    mcs = []

    if test_y_channel and X.shape[1] == 3:
        X = to_y_channel(X)
        Y = to_y_channel(Y)
        data_range = 255

    for _ in range(levels):
        ssim_val, cs = ssim(X,
                            Y,
                            win=win,
                            get_cs=True,
                            downsample=downsample,
                            data_range=data_range,
                            test_y_channel=test_y_channel)
        mcs.append(cs)
        padding = (X.shape[2] % 2, X.shape[3] % 2)
        X = F.avg_pool2d(X, kernel_size=2, padding=padding)
        Y = F.avg_pool2d(Y, kernel_size=2, padding=padding)

    mcs = torch.stack(mcs, dim=0)
    msssim_val = torch.prod(
        (mcs[:-1]**weights[:-1].unsqueeze(1)), dim=0) * (ssim_val**weights[-1])
    return msssim_val


@ARCH_REGISTRY.register()
class MS_SSIM(torch.nn.Module):
    r"""Args:
        channel: Number of channel.
        downsample: Boolean, whether to downsample which mimics official SSIM matlab code.
        test_y_channel: Boolean, whether to use y channel on ycbcr which mimics official matlab code.
    References:
        Wang, Zhou, Eero P. Simoncelli, and Alan C. Bovik. "Multiscale structural similarity for image 
        quality assessment." In The Thrity-Seventh Asilomar Conference on Signals, Systems & Computers, 
        2003, vol. 2, pp. 1398-1402. Ieee, 2003.
    """

    def __init__(self, channels=3, downsample=False, test_y_channel=True):
        super(MS_SSIM, self).__init__()
        self.win = fspecial_gauss(11, 1.5, channels)
        self.downsample = downsample
        self.test_y_channel = test_y_channel

    def forward(self, X, Y):
        """Computation of MS-SSIM metric.
        Args:
            x: An input tensor. Shape :math:`(N, C, H, W)`.
            y: A target tensor. Shape :math:`(N, C, H, W)`.
        Returns:
            Value of MS-SSIM metric in [0, 1] range.
        """
        assert X.shape == Y.shape, f"Input {X.shape} and reference images should have the same shape"
        score = ms_ssim(X,
                        Y,
                        win=self.win,
                        downsample=self.downsample,
                        test_y_channel=self.test_y_channel)
        return score