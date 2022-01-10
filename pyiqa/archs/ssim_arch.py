r"""SSIM Metric

Created by: https://github.com/dingkeyan93/IQA-optimization/blob/master/IQA_pytorch/SSIM.py

Modified by: Jiadi Mo (https://github.com/JiadiMo)

Refer to:
    Offical matlab code from https://www.cns.nyu.edu/~lcv/ssim/; 
    PIQ from https://github.com/photosynthesis-team/piq;
    BasicSR from https://github.com/xinntao/BasicSR/blob/master/basicsr/metrics/psnr_ssim.py

"""

import numpy as np
import torch
from torchvision import models, transforms
import torch.nn as nn
import torch.nn.functional as F

from pyiqa.utils.registry import ARCH_REGISTRY
from pyiqa.utils.color_util import rgb2ycbcr


def to_y_channel(img):
    """Change to Y channel of YCbCr.
    Args:
        image tensor: tensor with shape (N, 3, H, W) in range [0, 1].
    Returns:
        image tensor: Y channel of the input tensor 
    """
    assert img.ndim == 4 and img.shape[
        1] == 3, 'input image tensor should be RGB image batches with shape (N, 3, H, W)'
    img = rgb2ycbcr(img)
    return img[:, [0], :, :]


def fspecial_gauss(size, sigma, channels):
    # Function to mimic the 'fspecial' gaussian MATLAB function
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
    g = torch.from_numpy(g / g.sum()).float().unsqueeze(0).unsqueeze(0)
    return g.repeat(channels, 1, 1, 1)


def gaussian_filter(input, win):
    out = F.conv2d(input, win, stride=1, padding=0, groups=input.shape[1])
    return out


def ssim(X,
         Y,
         win,
         get_ssim_map=False,
         get_cs=False,
         get_weight=False,
         downsample=False,
         test_y_channel=True):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    # Whether calculate on y channel of ycbcr
    if test_y_channel:
        X = to_y_channel(X)
        Y = to_y_channel(Y)

    # Averagepool image if the size is large enough
    f = max(1, round(min(X.size()[-2:]) / 256))
    # Downsample operation is used in official matlab code
    if (f > 1) and downsample:
        X = F.avg_pool2d(X, kernel_size=f)
        Y = F.avg_pool2d(Y, kernel_size=f)

    win = win.to(X.device)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = gaussian_filter(X * X, win) - mu1_sq
    sigma2_sq = gaussian_filter(Y * Y, win) - mu2_sq
    sigma12 = gaussian_filter(X * Y, win) - mu1_mu2

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    cs_map = F.relu(
        cs_map
    )  #force the ssim response to be nonnegative to avoid negative results.
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map
    ssim_val = ssim_map.mean([1, 2, 3])

    if get_weight:
        weights = torch.log((1 + sigma1_sq / C2) * (1 + sigma2_sq / C2))
        return ssim_map, weights

    if get_ssim_map:
        return ssim_map

    if get_cs:
        return ssim_val, cs_map.mean([1, 2, 3])

    return ssim_val


@ARCH_REGISTRY.register()
class SSIM(torch.nn.Module):
    r"""Args:
        channel: number of channel.
        downsample: boolean, whether to downsample which mimics official matlab code.
        test_y_channel: boolean, whether to use y channel on ycbcr which mimics official matlab code.
    """

    def __init__(self, channels=3, downsample=False, test_y_channel=True):

        super(SSIM, self).__init__()
        self.win = fspecial_gauss(11, 1.5, channels)
        self.downsample = downsample
        self.test_y_channel = test_y_channel

    def forward(self, X, Y, as_loss=False):
        assert X.shape == Y.shape
        if as_loss:
            score = ssim(X,
                         Y,
                         win=self.win,
                         downsample=self.downsample,
                         test_y_channel=self.test_y_channel)
            return 1 - score.mean()
        else:
            with torch.no_grad():
                score = ssim(X,
                             Y,
                             win=self.win,
                             downsample=self.downsample,
                             test_y_channel=self.test_y_channel)
            return score
