r"""MAD Metric

Created by: https://github.com/dingkeyan93/IQA-optimization/blob/master/IQA_pytorch/MAD.py

Modified by: Jiadi Mo (https://github.com/JiadiMo)

Note:
    Offical matlab code is not available;
    Pytorch version >= 1.8.0;
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.fft import fftshift
import math

from pyiqa.matlab_utils import math_util
from pyiqa.utils.color_util import to_y_channel
from pyiqa.utils.registry import ARCH_REGISTRY

MAX = nn.MaxPool2d((2, 2), stride=1, padding=1)


def extract_patches_2d(img: torch.Tensor,
                       patch_shape: list = [64, 64],
                       step: list = [27, 27],
                       batch_first: bool = True,
                       keep_last_patch: bool = False) -> torch.Tensor:
    patch_H, patch_W = patch_shape[0], patch_shape[1]

    if (img.size(2) < patch_H):
        num_padded_H_Top = (patch_H - img.size(2)) // 2
        num_padded_H_Bottom = patch_H - img.size(2) - num_padded_H_Top
        padding_H = nn.ConstantPad2d((0, 0, num_padded_H_Top, num_padded_H_Bottom), 0)
        img = padding_H(img)

    if (img.size(3) < patch_W):
        num_padded_W_Left = (patch_W - img.size(3)) // 2
        num_padded_W_Right = patch_W - img.size(3) - num_padded_W_Left
        padding_W = nn.ConstantPad2d((num_padded_W_Left, num_padded_W_Right, 0, 0), 0)
        img = padding_W(img)

    step_int = [0, 0]
    step_int[0] = int(patch_H * step[0]) if (isinstance(step[0], float)) else step[0]
    step_int[1] = int(patch_W * step[1]) if (isinstance(step[1], float)) else step[1]
    patches_fold_H = img.unfold(2, patch_H, step_int[0])

    if ((img.size(2) - patch_H) % step_int[0] != 0) and keep_last_patch:
        patches_fold_H = torch.cat((patches_fold_H, img[:, :, -patch_H:, ].permute(0, 1, 3, 2).unsqueeze(2)), dim=2)

    patches_fold_HW = patches_fold_H.unfold(3, patch_W, step_int[1])
    if ((img.size(3) - patch_W) % step_int[1] != 0) and keep_last_patch:
        patches_fold_HW = torch.cat(
            (patches_fold_HW, patches_fold_H[:, :, :, -patch_W:, :].permute(0, 1, 2, 4, 3).unsqueeze(3)), dim=3)

    patches = patches_fold_HW.permute(2, 3, 0, 1, 4, 5)
    patches = patches.reshape(-1, img.size(0), img.size(1), patch_H, patch_W)

    if (batch_first):
        patches = patches.permute(1, 0, 2, 3, 4)

    return patches


def make_csf(rows, cols, nfreq):
    xvals = np.arange(-(cols - 1) / 2., (cols + 1) / 2.)
    yvals = np.arange(-(rows - 1) / 2., (rows + 1) / 2.)

    xplane, yplane = np.meshgrid(xvals, yvals)  # generate mesh
    plane = ((xplane + 1j * yplane) / cols) * 2 * nfreq
    radfreq = np.abs(plane)  # radial frequency

    w = 0.7
    s = (1 - w) / 2 * np.cos(4 * np.angle(plane)) + (1 + w) / 2
    radfreq = radfreq / s

    # Now generate the CSF
    csf = 2.6 * (0.0192 + 0.114 * radfreq) * np.exp(-(0.114 * radfreq)**1.1)
    csf[radfreq < 7.8909] = 0.9809

    return np.transpose(csf)


def get_moments(d, sk=False):
    # Return the first 4 moments of the data provided
    mean = torch.mean(d, dim=[3, 4], keepdim=True)
    diffs = d - mean
    var = torch.mean(torch.pow(diffs, 2.0), dim=[3, 4], keepdim=True)
    std = torch.pow(var + 1e-12, 0.5)

    if sk:
        zscores = diffs / std
        skews = torch.mean(torch.pow(zscores, 3.0), dim=[3, 4], keepdim=True)
        kurtoses = torch.mean(
            torch.pow(zscores, 4.0), dim=[3, 4], keepdim=True) - 3.0  # excess kurtosis, should be 0 for Gaussian
        return mean, std, skews, kurtoses
    else:
        return mean, std


def ical_stat(x, p=16, s=4):
    B, C, H, W = x.shape
    x1 = extract_patches_2d(x, patch_shape=[p, p], step=[s, s])
    _, std, skews, kurt = get_moments(x1, sk=True)

    STD = std.reshape(B, C, (H - (p - s)) // s, (W - (p - s)) // s)
    SKEWS = skews.reshape(B, C, (H - (p - s)) // s, (W - (p - s)) // s)
    KURT = kurt.reshape(B, C, (H - (p - s)) // s, (W - (p - s)) // s)

    return STD, SKEWS, KURT  # different with original version


def ical_std(x, p=16, s=4):
    B, C, H, W = x.shape
    x1 = extract_patches_2d(x, patch_shape=[p, p], step=[s, s])
    mean, std = get_moments(x1)
    mean = mean.reshape(B, C, (H - (p - s)) // s, (W - (p - s)) // s)
    std = std.reshape(B, C, (H - (p - s)) // s, (W - (p - s)) // s)

    return mean, std


def hi_index(ref_img, dst_img):
    k = 0.02874
    G = 0.5
    C_slope = 1
    Ci_thrsh = -5
    Cd_thrsh = -5

    ref = k * (ref_img + 1e-12)**(2.2 / 3)
    dst = k * (torch.abs(dst_img) + 1e-12)**(2.2 / 3)

    B, C, H, W = ref.shape

    csf = make_csf(H, W, 32)
    csf = torch.from_numpy(csf.reshape(1, 1, H, W, 1)).float().repeat(1, C, 1, 1, 2).to(ref.device)

    x = torch.fft.fft2(ref)
    x1 = math_util.batch_fftshift2d(x)
    x2 = math_util.batch_ifftshift2d(x1 * csf)
    ref = torch.fft.ifft2(x2).real

    x = torch.fft.fft2(dst)
    x1 = math_util.batch_fftshift2d(x)
    x2 = math_util.batch_ifftshift2d(x1 * csf)
    dst = torch.fft.ifft2(x2).real

    m1_1, std_1 = ical_std(ref)
    B, C, H1, W1 = m1_1.shape

    std_1 = (-MAX(-std_1) / 2)[:, :, :H1, :W1]
    _, std_2 = ical_std(dst - ref)

    BSIZE = 16
    eps = 1e-12

    Ci_ref = torch.log(torch.abs((std_1 + eps) / (m1_1 + eps)))
    Ci_dst = torch.log(torch.abs((std_2 + eps) / (m1_1 + eps)))

    Ci_dst = Ci_dst.masked_fill(m1_1 < G, -1000)

    idx1 = (Ci_ref > Ci_thrsh) & (Ci_dst > (C_slope * (Ci_ref - Ci_thrsh) + Cd_thrsh))
    idx2 = (Ci_ref <= Ci_thrsh) & (Ci_dst > Cd_thrsh)

    msk = Ci_ref.clone()
    msk = msk.masked_fill(~idx1, 0)
    msk = msk.masked_fill(~idx2, 0)
    msk[idx1] = Ci_dst[idx1] - (C_slope * (Ci_ref[idx1] - Ci_thrsh) + Cd_thrsh)
    msk[idx2] = Ci_dst[idx2] - Cd_thrsh

    win = torch.ones((1, 1, BSIZE, BSIZE)).repeat(C, 1, 1, 1).to(ref.device) / BSIZE**2
    xx = (ref_img - dst_img)**2

    lmse = F.conv2d(xx, win, stride=4, padding=0, groups=C)

    mp = msk * lmse
    B, C, H, W = mp.shape
    return torch.norm(mp.reshape(B, C, -1), dim=2) / math.sqrt(H * W) * 200


def gaborconvolve(im):

    nscale = 5  # Number of wavelet scales.
    norient = 4  # Number of filter orientations.
    minWaveLength = 3  # Wavelength of smallest scale filter.
    mult = 3  # Scaling factor between successive filters.
    sigmaOnf = 0.55  # Ratio of the standard deviation of the
    wavelength = [
        minWaveLength, minWaveLength * mult, minWaveLength * mult**2, minWaveLength * mult**3, minWaveLength * mult**4
    ]
    # Ratio of angular interval between filter orientations
    dThetaOnSigma = 1.5

    # Fourier transform of image
    B, C, rows, cols = im.shape
    # imagefft    = torch.rfft(im,2, onesided=False)
    imagefft = torch.fft.fft2(im)
    # Pre-compute to speed up filter construction
    x = np.ones((rows, 1)) * np.arange(-cols / 2., (cols / 2.)) / (cols / 2.)
    y = np.dot(np.expand_dims(np.arange(-rows / 2., (rows / 2.)), 1), np.ones((1, cols)) / (rows / 2.))
    # Matrix values contain *normalised* radius from centre.
    radius = np.sqrt(x**2 + y**2)
    # Get rid of the 0 radius value in the middle
    radius[int(np.round(rows / 2 + 1)), int(np.round(cols / 2 + 1))] = 1
    radius = np.log(radius + 1e-12)

    # Matrix values contain polar angle.
    theta = np.arctan2(-y, x)
    sintheta = np.sin(theta)
    costheta = np.cos(theta)

    # Calculate the standard deviation
    thetaSigma = math.pi / norient / dThetaOnSigma
    logGabors = []
    for s in range(nscale):
        # Construct the filter - first calculate the radial filter component.
        fo = 1.0 / wavelength[s]  # Centre frequency of filter.
        rfo = fo / 0.5  # Normalised radius from centre of frequency plane
        # corresponding to fo.
        tmp = -(2 * np.log(sigmaOnf)**2)
        tmp2 = np.log(rfo)
        logGabors.append(np.exp((radius - tmp2)**2 / tmp))
        logGabors[s][int(np.round(rows / 2)), int(np.round(cols / 2))] = 0

    E0 = [[], [], [], []]
    for o in range(norient):
        # Calculate filter angle.
        angl = o * math.pi / norient

        ds = sintheta * np.cos(angl) - costheta * np.sin(angl)  # Difference in sine.
        dc = costheta * np.cos(angl) + sintheta * np.sin(angl)  # Difference in cosine.
        dtheta = np.abs(np.arctan2(ds, dc))  # Absolute angular distance.
        spread = np.exp((-dtheta**2) / (2 * thetaSigma**2))  # Calculate the angular filter component.

        for s in range(nscale):

            filter = fftshift(logGabors[s] * spread)
            filter = torch.from_numpy(filter).reshape(1, 1, rows, cols).to(im.device)
            e0 = torch.fft.ifft2(imagefft * filter)
            E0[o].append(torch.stack((e0.real, e0.imag), -1))

    return E0


def lo_index(ref, dst):
    gabRef = gaborconvolve(ref)
    gabDst = gaborconvolve(dst)
    s = [0.5 / 13.25, 0.75 / 13.25, 1 / 13.25, 5 / 13.25, 6 / 13.25]

    mp = 0
    for gb_i in range(4):
        for gb_j in range(5):
            stdref, skwref, krtref = ical_stat(math_util.abs(gabRef[gb_i][gb_j]))
            stddst, skwdst, krtdst = ical_stat(math_util.abs(gabDst[gb_i][gb_j]))
            mp = mp + s[gb_i] * (
                torch.abs(stdref - stddst) + 2 * torch.abs(skwref - skwdst) + torch.abs(krtref - krtdst))

    B, C, rows, cols = mp.shape
    return torch.norm(mp.reshape(B, C, -1), dim=2) / np.sqrt(rows * cols)


@ARCH_REGISTRY.register()
class MAD(torch.nn.Module):
    r"""Args:
        channel: Number of input channel.
        test_y_channel: bool, whether to use y channel on ycbcr which mimics official matlab code.
    References:
        Larson, Eric Cooper, and Damon Michael Chandler. "Most apparent distortion: full-reference
        image quality assessment and the role of strategy." Journal of electronic imaging 19, no. 1
        (2010): 011006.
    """

    def __init__(self, channels=3, test_y_channel=True):
        super(MAD, self).__init__()
        self.channels = channels
        self.test_y_channel = test_y_channel

    def mad(self, ref, dst):
        r"""Compute MAD for a batch of images.
        Args:
            ref: An reference tensor. Shape :math:`(N, C, H, W)`.
            dst: A distortion tensor. Shape :math:`(N, C, H, W)`.
        """
        if self.test_y_channel and ref.shape[1] == 3:
            ref = to_y_channel(ref, 255.)
            dst = to_y_channel(dst, 255.)
            self.channels = 1

        HI = hi_index(ref, dst)
        LO = lo_index(ref, dst)
        thresh1 = 2.55
        thresh2 = 3.35
        b1 = math.exp(-thresh1 / thresh2)
        b2 = 1 / (math.log(10) * thresh2)
        sig = 1 / (1 + b1 * HI**b2)
        MAD = LO**(1 - sig) * HI**(sig)
        return MAD.mean(1)

    def forward(self, X, Y):
        r"""Computation of CW-SSIM metric.
        Args:
            X: An input tensor. Shape :math:`(N, C, H, W)`.
            Y: A target tensor. Shape :math:`(N, C, H, W)`.
        Returns:
            Value of MAD metric in [0, 1] range.
        """
        assert X.shape == Y.shape, f'Input and reference images should have the same shape, but got {X.shape} and {Y.shape}'
        score = self.mad(Y, X)
        return score
