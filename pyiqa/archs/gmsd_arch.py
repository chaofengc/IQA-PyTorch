r"""GMSD Metric

Created by: https://github.com/dingkeyan93/IQA-optimization/blob/master/IQA_pytorch/GMSD.py

Modified by: Jiadi Mo (https://github.com/JiadiMo)

Refer to:
    Matlab code from https://www4.comp.polyu.edu.hk/~cslzhang/IQA/GMSD/GMSD.m;

"""

import torch
from torch import nn
from torch.nn import functional as F

from pyiqa.utils.color_util import to_y_channel
from pyiqa.utils.registry import ARCH_REGISTRY


def gmsd(
    x: torch.Tensor,
    y: torch.Tensor,
    T: int = 170,
    channels: int = 3,
    test_y_channel: bool = True,
) -> torch.Tensor:
    r"""GMSD metric.
    Args:
        x: A distortion tensor. Shape :math:`(N, C, H, W)`.
        y: A reference tensor. Shape :math:`(N, C, H, W)`.
        T: A positive constant that supplies numerical stability.
        channels: Number of channels.
        test_y_channel: bool, whether to use y channel on ycbcr.
    """
    if test_y_channel:
        x = to_y_channel(x, 255)
        y = to_y_channel(y, 255)
        channels = 1
    else:
        x = x * 255.
        y = y * 255.

    dx = (torch.Tensor([[1, 0, -1], [1, 0, -1], [1, 0, -1]]) / 3.).unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1,
                                                                                                    1).to(x)
    dy = (torch.Tensor([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]) / 3.).unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1,
                                                                                                    1).to(x)
    aveKernel = torch.ones(channels, 1, 2, 2).to(x) / 4.

    Y1 = F.conv2d(x, aveKernel, stride=2, padding=0, groups=channels)
    Y2 = F.conv2d(y, aveKernel, stride=2, padding=0, groups=channels)

    IxY1 = F.conv2d(Y1, dx, stride=1, padding=1, groups=channels)
    IyY1 = F.conv2d(Y1, dy, stride=1, padding=1, groups=channels)
    gradientMap1 = torch.sqrt(IxY1**2 + IyY1**2 + 1e-12)

    IxY2 = F.conv2d(Y2, dx, stride=1, padding=1, groups=channels)
    IyY2 = F.conv2d(Y2, dy, stride=1, padding=1, groups=channels)
    gradientMap2 = torch.sqrt(IxY2**2 + IyY2**2 + 1e-12)

    quality_map = (2 * gradientMap1 * gradientMap2 + T) / (gradientMap1**2 + gradientMap2**2 + T)
    score = torch.std(quality_map.view(quality_map.shape[0], -1), dim=1)

    return score


@ARCH_REGISTRY.register()
class GMSD(nn.Module):
    r'''Gradient Magnitude Similarity Deviation Metric.
    Args:
        channels: Number of channels.
        test_y_channel: bool, whether to use y channel on ycbcr.
    Reference:
        Xue, Wufeng, Lei Zhang, Xuanqin Mou, and Alan C. Bovik.
        "Gradient magnitude similarity deviation: A highly efficient
        perceptual image quality index." IEEE Transactions on Image
        Processing 23, no. 2 (2013): 684-695.
    '''

    def __init__(self, channels: int = 3, test_y_channel: bool = True) -> None:
        super(GMSD, self).__init__()
        self.channels = channels
        self.test_y_channel = test_y_channel

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r"""Args:
            x: A distortion tensor. Shape :math:`(N, C, H, W)`.
            y: A reference tensor. Shape :math:`(N, C, H, W)`.
            Order of input is important.
        """
        assert x.shape == y.shape, f'Input and reference images should have the same shape, but got {x.shape} and {y.shape}'
        score = gmsd(x, y, channels=self.channels, test_y_channel=self.test_y_channel)

        return score
