r"""SSIM, MS-SSIM, CW-SSIM Metric

Created by:
- https://github.com/dingkeyan93/IQA-optimization/blob/master/IQA_pytorch/SSIM.py
- https://github.com/dingkeyan93/IQA-optimization/blob/master/IQA_pytorch/MS_SSIM.py
- https://github.com/dingkeyan93/IQA-optimization/blob/master/IQA_pytorch/CW_SSIM.py

Modified by: Jiadi Mo (https://github.com/JiadiMo)

Refer to:
    - Offical SSIM matlab code from https://www.cns.nyu.edu/~lcv/ssim/;
    - PIQ from https://github.com/photosynthesis-team/piq;
    - BasicSR from https://github.com/xinntao/BasicSR/blob/master/basicsr/metrics/psnr_ssim.py;
    - Offical MS-SSIM matlab code from https://ece.uwaterloo.ca/~z70wang/research/iwssim/msssim.zip;
    - Offical CW-SSIM matlab code from
    https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/43017/versions/1/download/zip;

"""

import numpy as np
import torch
import torch.nn.functional as F

from pyiqa.utils.color_util import to_y_channel
from pyiqa.matlab_utils import fspecial, SCFpyr_PyTorch, math_util, filter2
from pyiqa.utils.registry import ARCH_REGISTRY


def ssim(X,
         Y,
         win,
         get_ssim_map=False,
         get_cs=False,
         get_weight=False,
         downsample=False,
         data_range=1.,
         test_y_channel=True,
         color_space='yiq'):

    data_range = 255
    # Whether calculate on y channel of ycbcr
    if test_y_channel and X.shape[1] == 3:
        X = to_y_channel(X, data_range, color_space)
        Y = to_y_channel(Y, data_range, color_space)
    else:
        X = X * data_range
        X = X - X.detach() + X.round()
        Y = Y * data_range
        Y = Y - Y.detach() + Y.round()

    C1 = (0.01 * data_range)**2
    C2 = (0.03 * data_range)**2

    # Averagepool image if the size is large enough
    f = max(1, round(min(X.size()[-2:]) / 256))
    # Downsample operation is used in official matlab code
    if (f > 1) and downsample:
        X = F.avg_pool2d(X, kernel_size=f)
        Y = F.avg_pool2d(Y, kernel_size=f)

    win = win.to(X.device)

    mu1 = filter2(X, win, 'valid')
    mu2 = filter2(Y, win, 'valid')
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(X * X, win, 'valid') - mu1_sq
    sigma2_sq = filter2(Y * Y, win, 'valid') - mu2_sq
    sigma12 = filter2(X * Y, win, 'valid') - mu1_mu2

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    cs_map = F.relu(cs_map)  # force the ssim response to be nonnegative to avoid negative results.
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
        downsample: boolean, whether to downsample same as official matlab code.
        test_y_channel: boolean, whether to use y channel on ycbcr same as official matlab code.
    """

    def __init__(self, channels=3, downsample=False, test_y_channel=True, color_space='yiq', crop_border=0.):

        super(SSIM, self).__init__()
        self.win = fspecial(11, 1.5, channels)
        self.downsample = downsample
        self.test_y_channel = test_y_channel
        self.color_space = color_space
        self.crop_border = crop_border

    def forward(self, X, Y):
        assert X.shape == Y.shape, f'Input {X.shape} and reference images should have the same shape'

        if self.crop_border != 0:
            crop_border = self.crop_border
            X = X[..., crop_border:-crop_border, crop_border:-crop_border]
            Y = Y[..., crop_border:-crop_border, crop_border:-crop_border]

        score = ssim(
            X,
            Y,
            win=self.win,
            downsample=self.downsample,
            test_y_channel=self.test_y_channel,
            color_space=self.color_space)
        return score


def ms_ssim(X, Y, win, data_range=1., downsample=False, test_y_channel=True, is_prod=True, color_space='yiq'):
    r"""Compute Multiscale structural similarity for a batch of images.
    Args:
        x: An input tensor. Shape :math:`(N, C, H, W)`.
        y: A target tensor. Shape :math:`(N, C, H, W)`.
        win: Window setting.
        downsample: Boolean, whether to downsample which mimics official SSIM matlab code.
        test_y_channel: Boolean, whether to use y channel on ycbcr.
        is_prod: Boolean, calculate product or sum between mcs and weight.
    Returns:
        Index of similarity betwen two images. Usually in [0, 1] interval.
    """
    if not X.shape == Y.shape:
        raise ValueError('Input images must have the same dimensions.')

    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(X)

    levels = weights.shape[0]
    mcs = []

    for _ in range(levels):
        ssim_val, cs = ssim(
            X,
            Y,
            win=win,
            get_cs=True,
            downsample=downsample,
            data_range=data_range,
            test_y_channel=test_y_channel,
            color_space=color_space)
        mcs.append(cs)
        padding = (X.shape[2] % 2, X.shape[3] % 2)
        X = F.avg_pool2d(X, kernel_size=2, padding=padding)
        Y = F.avg_pool2d(Y, kernel_size=2, padding=padding)

    mcs = torch.stack(mcs, dim=0)

    if is_prod:
        msssim_val = torch.prod((mcs[:-1]**weights[:-1].unsqueeze(1)), dim=0) * (ssim_val**weights[-1])
    else:
        weights = weights / torch.sum(weights)
        msssim_val = torch.sum((mcs[:-1] * weights[:-1].unsqueeze(1)), dim=0) + (ssim_val * weights[-1])

    return msssim_val


@ARCH_REGISTRY.register()
class MS_SSIM(torch.nn.Module):
    r"""Multiscale structure similarity

    References:
        Wang, Zhou, Eero P. Simoncelli, and Alan C. Bovik. "Multiscale structural similarity for image
        quality assessment." In The Thrity-Seventh Asilomar Conference on Signals, Systems & Computers,
        2003, vol. 2, pp. 1398-1402. Ieee, 2003.

    Args:
        channel: Number of channel.
        downsample: Boolean, whether to downsample which mimics official SSIM matlab code.
        test_y_channel: Boolean, whether to use y channel on ycbcr which mimics official matlab code.
    """

    def __init__(self, channels=3, downsample=False, test_y_channel=True, is_prod=True, color_space='yiq'):
        super(MS_SSIM, self).__init__()
        self.win = fspecial(11, 1.5, channels)
        self.downsample = downsample
        self.test_y_channel = test_y_channel
        self.color_space = color_space
        self.is_prod = is_prod

    def forward(self, X, Y):
        """Computation of MS-SSIM metric.
        Args:
            x: An input tensor. Shape :math:`(N, C, H, W)`.
            y: A target tensor. Shape :math:`(N, C, H, W)`.
        Returns:
            Value of MS-SSIM metric in [0, 1] range.
        """
        assert X.shape == Y.shape, 'Input and reference images should have the same shape, but got'
        f'{X.shape} and {Y.shape}'
        score = ms_ssim(
            X,
            Y,
            win=self.win,
            downsample=self.downsample,
            test_y_channel=self.test_y_channel,
            is_prod=self.is_prod,
            color_space=self.color_space)
        return score


@ARCH_REGISTRY.register()
class CW_SSIM(torch.nn.Module):
    r'''Complex-Wavelet Structural SIMilarity (CW-SSIM) index.

    References:
        M. P. Sampat, Z. Wang, S. Gupta, A. C. Bovik, M. K. Markey.
        "Complex Wavelet Structural Similarity: A New Image Similarity Index",
        IEEE Transactions on Image Processing, 18(11), 2385-401, 2009.

    Args:
        channel: Number of channel.
        test_y_channel: Boolean, whether to use y channel on ycbcr.
        level: The number of levels to used in the complex steerable pyramid decomposition
        ori: The number of orientations to be used in the complex steerable pyramid decomposition
        guardb: How much is discarded from the four image boundaries.
        K: the constant in the CWSSIM index formula (see the above reference) default value: K=0
    '''

    def __init__(self, channels=1, level=4, ori=8, guardb=0, K=0, test_y_channel=True, color_space='yiq'):

        super(CW_SSIM, self).__init__()
        self.channels = channels
        self.level = level
        self.ori = ori
        self.guardb = guardb
        self.K = K
        self.test_y_channel = test_y_channel
        self.color_space = color_space
        self.register_buffer('win7', torch.ones(channels, 1, 7, 7) / (7 * 7))

    def conj(self, x, y):
        a = x[..., 0]
        b = x[..., 1]
        c = y[..., 0]
        d = -y[..., 1]
        return torch.stack((a * c - b * d, b * c + a * d), dim=1)

    def conv2d_complex(self, x, win, groups=1):
        real = F.conv2d(x[:, 0, ...].unsqueeze(1), win, groups=groups)
        imaginary = F.conv2d(x[:, 1, ...].unsqueeze(1), win, groups=groups)
        return torch.stack((real, imaginary), dim=-1)

    def cw_ssim(self, x, y, test_y_channel):
        r"""Compute CW-SSIM for a batch of images.
        Args:
            x: An input tensor. Shape :math:`(N, C, H, W)`.
            y: A target tensor. Shape :math:`(N, C, H, W)`.
            test_y_channel: Boolean, whether to use y channel on ycbcr.
        Returns:
            Index of similarity betwen two images. Usually in [0, 1] interval.
        """
        # Whether calculate on y channel of ycbcr
        if test_y_channel and x.shape[1] == 3:
            x = to_y_channel(x, 255, self.color_space)
            y = to_y_channel(y, 255, self.color_space)

        pyr = SCFpyr_PyTorch(height=self.level, nbands=self.ori, scale_factor=2, device=x.device)
        cw_x = pyr.build(x)
        cw_y = pyr.build(y)

        bandind = self.level
        band_cssim = []

        s = np.array(cw_x[bandind][0].size()[1:3])
        w = fspecial(s - 7 + 1, s[0] / 4, 1).to(x.device)
        gb = int(self.guardb / (2**(self.level - 1)))

        for i in range(self.ori):

            band1 = cw_x[bandind][i]
            band2 = cw_y[bandind][i]

            band1 = band1[:, gb:s[0] - gb, gb:s[1] - gb, :]
            band2 = band2[:, gb:s[0] - gb, gb:s[1] - gb, :]

            corr = self.conj(band1, band2)
            corr_band = self.conv2d_complex(corr, self.win7, groups=self.channels)
            varr = ((math_util.abs(band1))**2 + (math_util.abs(band2))**2).unsqueeze(1)
            varr_band = F.conv2d(varr, self.win7, stride=1, padding=0, groups=self.channels)
            cssim_map = (2 * math_util.abs(corr_band) + self.K) / (varr_band + self.K)
            band_cssim.append((cssim_map * w.repeat(cssim_map.shape[0], 1, 1, 1)).sum([2, 3]).mean(1))

        return torch.stack(band_cssim, dim=1).mean(1)

    def forward(self, X, Y):
        r"""Computation of CW-SSIM metric.
        Args:
            X: An input tensor. Shape :math:`(N, C, H, W)`.
            Y: A target tensor. Shape :math:`(N, C, H, W)`.
        Returns:
            Value of CW-SSIM metric in [0, 1] range.
        """
        assert X.shape == Y.shape, f'Input {X.shape} and reference images should have the same shape'
        score = self.cw_ssim(X, Y, self.test_y_channel)
        return score
