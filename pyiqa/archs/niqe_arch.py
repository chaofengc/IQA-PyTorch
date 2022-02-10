r"""NIQE Metric

Created by: https://github.com/xinntao/BasicSR/blob/5668ba75eb8a77e8d2dd46746a36fee0fbb0fdcd/basicsr/metrics/niqe.py

Modified by: Jiadi Mo (https://github.com/JiadiMo)

Reference:
    MATLAB codes: http://live.ece.utexas.edu/research/quality/niqe_release.zip
    
"""

import math
import numpy as np
import scipy
import scipy.io
import scipy.misc
import scipy.ndimage
import scipy.special
import torch
import torch.nn.functional as F
import torch.nn as nn
from scipy.special import gamma

from pyiqa.archs.ssim_arch import to_y_channel, fspecial_gauss
from pyiqa.utils.matlab_functions import imresize
from pyiqa.utils.registry import ARCH_REGISTRY


def estimate_aggd_param(block):
    """Estimate AGGD (Asymmetric Generalized Gaussian Distribution) parameters.
    Args:
        block (ndarray): 2D Image block.
    Returns:
        tuple: alpha (float), beta_l (float) and beta_r (float) for the AGGD
            distribution (Estimating the parames in Equation 7 in the paper).
    """
    block = block.flatten(1)
    gam = torch.arange(0.2, 10.001, 0.001, requires_grad=True)
    gam_reciprocal = torch.reciprocal(gam).detach()
    r_gam = torch.square(gamma(gam_reciprocal * 2)) / (gamma(gam_reciprocal) * gamma(gam_reciprocal * 3))

    batches = []
    for b in range(block.size(0)):
        sample_block = block[b,...]
        left_std = torch.sqrt(torch.mean(sample_block[sample_block < 0]**2))
        right_std = torch.sqrt(torch.mean(sample_block[sample_block > 0]**2))

        gammahat = left_std / right_std
        rhat = (torch.mean(torch.abs(sample_block)))**2 / torch.mean(sample_block**2)
        rhatnorm = (rhat * (gammahat**3 + 1) * (gammahat + 1)) / ((gammahat**2 + 1)**2)
        (_, array_position) = torch.min((r_gam - rhatnorm)**2, 0)

        alpha = gam[array_position]
        beta_l = left_std * torch.sqrt(gamma((1 / alpha).detach()) / gamma((3 / alpha).detach()))
        beta_r = right_std * torch.sqrt(gamma((1 / alpha).detach()) / gamma((3 / alpha).detach()))
        batches.append(torch.Tensor([alpha, beta_l, beta_r]))
    return torch.stack(batches)


def compute_feature(block):
    """Compute features.
    Args:
        block (ndarray): 2D Image block.
    Returns:
        list: Features with length of 18.
    """
    res = estimate_aggd_param(block)
    alpha, beta_l, beta_r = res[:,0].unsqueeze(-1), res[:,1].unsqueeze(-1), res[:,2].unsqueeze(-1)
    feat = torch.cat((alpha, (beta_l + beta_r) / 2), 1)


    # distortions disturb the fairly regular structure of natural images.
    # This deviation can be captured by analyzing the sample distribution of
    # the products of pairs of adjacent coefficients computed along
    # horizontal, vertical and diagonal orientations.
    shifts = [[0, 1], [1, 0], [1, 1], [1, -1]]
    for i in range(len(shifts)):
        shifted_block = torch.roll(block, shifts[i], dims=(2, 3))
        res = estimate_aggd_param(block * shifted_block)
        alpha, beta_l, beta_r = res[:,0].unsqueeze(-1), res[:,1].unsqueeze(-1), res[:,2].unsqueeze(-1)
        # Eq. 8
        mean = (beta_r - beta_l) * (gamma(2 / alpha) / gamma(1 / alpha))
        feat = torch.cat((feat, alpha, mean, beta_l, beta_r), 1)

    return torch.Tensor(feat)


def niqe(img, mu_pris_param, cov_pris_param, gaussian_window, block_size_h=96, block_size_w=96):
    """Calculate NIQE (Natural Image Quality Evaluator) metric.
    Args:
        img (ndarray): Input image whose quality needs to be computed. The
            image must be a gray or Y (of YCbCr) image with shape (h, w).
            Range [0, 255] with float type.
        mu_pris_param (ndarray): Mean of a pre-defined multivariate Gaussian
            model calculated on the pristine dataset.
        cov_pris_param (ndarray): Covariance of a pre-defined multivariate
            Gaussian model calculated on the pristine dataset.
        gaussian_window (ndarray): A 7x7 Gaussian window used for smoothing the
            image.
        block_size_h (int): Height of the blocks in to which image is divided.
            Default: 96 (the official recommended value).
        block_size_w (int): Width of the blocks in to which image is divided.
            Default: 96 (the official recommended value).
    """
    assert img.ndim == 4, ('Input image must be a gray or Y (of YCbCr) image with shape (b, c, h, w).')
    # crop image
    b, c, h, w = img.shape
    num_block_h = math.floor(h / block_size_h)
    num_block_w = math.floor(w / block_size_w)
    img = img[..., 0:num_block_h * block_size_h, 0:num_block_w * block_size_w]

    distparam = []  # dist param is actually the multiscale features
    for scale in (1, 2):  # perform on two scales (1, 2)
        rep_one = nn.ReplicationPad2d(3)
        mu = F.conv2d(rep_one(img),gaussian_window,groups=1)
        sigma = torch.sqrt(torch.abs(F.conv2d(rep_one(img**2),gaussian_window,groups=1) - mu**2))
        # normalize, as in Eq. 1 in the paper
        img_nomalized = (img - mu) / (sigma + 1)

        feat = []
        for idx_w in range(num_block_w):
            for idx_h in range(num_block_h):
                # process ecah block
                block = img_nomalized[..., idx_h * block_size_h // scale:(idx_h + 1) * block_size_h // scale,
                                      idx_w * block_size_w // scale:(idx_w + 1) * block_size_w // scale]
                feat.append(compute_feature(block))

        distparam.append(torch.stack(feat).transpose(0,1))

        if scale == 1:
            img = imresize(img / 255., scale=0.5, antialiasing=True)
            img = img * 255.

    distparam = torch.cat(distparam, -1)

    # fit a MVG (multivariate Gaussian) model to distorted patch features
    mu_distparam = torch.nanmean(distparam, axis=1)

    cov_distparam = []
    for in_b in range(b):
        sample_distparam = distparam[in_b,...]
        distparam_no_nan = sample_distparam[~torch.isnan(sample_distparam).any(axis=1)]
        cov_distparam.append(torch.cov(distparam_no_nan.T))
    
    # compute niqe quality, Eq. 10 in the paper
    invcov_param = torch.linalg.pinv((cov_pris_param + torch.stack(cov_distparam)) / 2)
    diff = (mu_pris_param - mu_distparam).unsqueeze(1)
    quality = torch.bmm(torch.bmm(diff, invcov_param), diff.transpose(1,2)).squeeze()

    quality = torch.sqrt(quality)
    return quality


def calculate_niqe(img, crop_border=0, test_y_channel=True, pretrained_model_path=None, **kwargs):
    """Calculate NIQE (Natural Image Quality Evaluator) metric.
    Args:
        img (ndarray): Input image whose quality needs to be computed.
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the metric calculation.
        test_y_channel (Bool): Whether converted to 'y' (of MATLAB YCbCr) or 'gray'.
        pretrained_model_path (String): The pretrained model path.
    Returns:
        float: NIQE result.
    """

    params = scipy.io.loadmat(pretrained_model_path)
    mu_pris_param = np.ravel(params['mu_prisparam'])
    cov_pris_param = params['cov_prisparam']
    mu_pris_param = torch.from_numpy(mu_pris_param)
    cov_pris_param = torch.from_numpy(cov_pris_param)

    mu_pris_param = mu_pris_param.repeat(img.size(0),1).requires_grad_()
    cov_pris_param = cov_pris_param.repeat(img.size(0),1,1).requires_grad_()

    gaussian_window = fspecial_gauss(7, 7.0/6.0, 1)
    gaussian_window = gaussian_window/torch.sum(gaussian_window)

    if test_y_channel and img.shape[1] == 3:
        img = to_y_channel(img)

    if crop_border != 0:
        img = img[..., crop_border:-crop_border, crop_border:-crop_border]

    niqe_result = niqe(img, mu_pris_param, cov_pris_param, gaussian_window)

    return niqe_result


@ARCH_REGISTRY.register()
class NIQE(torch.nn.Module):
    r"""Args:
        channels (int): Number of processed channel.
        test_y_channel (Boolean): whether to use y channel on ycbcr.
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the metric calculation.
        pretrained_model_path (String): The pretrained model path.
    References:
        Mittal, Anish, Rajiv Soundararajan, and Alan C. Bovik. 
        "Making a “completely blind” image quality analyzer." 
        IEEE Signal Processing Letters (SPL) 20.3 (2012): 209-212.
    """

    def __init__(self, channels=1, test_y_channel=True, crop_border=0, pretrained_model_path=None):

        super(NIQE, self).__init__()
        self.channels = channels
        self.test_y_channel = test_y_channel
        self.crop_border = crop_border
        self.pretrained_model_path = pretrained_model_path


    def forward(self, X):
        r"""Computation of NIQE metric.
        Args:
            X: An input tensor. Shape :math:`(N, C, H, W)`.
        Returns:
            Value of niqe metric in [0, 1] range.
        """
        score = calculate_niqe(X, self.crop_border, self.test_y_channel, self.pretrained_model_path)
        return score
