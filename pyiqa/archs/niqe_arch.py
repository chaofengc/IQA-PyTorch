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
from tokenize import String
from typing import Tuple
from xmlrpc.client import Boolean

from pyiqa.utils.color_util import to_y_channel
from pyiqa.utils.download_util import load_file_from_url
from pyiqa.utils.matlab_functions import imresize, fspecial_gauss
from .func_util import estimate_aggd_param, torch_cov, normalize_img_with_guass
from pyiqa.utils.registry import ARCH_REGISTRY


default_model_urls = {
    'url': 'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/niqe_modelparameters.mat'
}


def compute_feature(block: torch.Tensor) -> torch.Tensor:
    """Compute features.
    Args:
        block (Tensor): Image block in shape (b, c, h, w).
    Returns:
        list: Features with length of 18.
    """
    alpha, beta_l, beta_r = estimate_aggd_param(block)
    feat = [alpha, (beta_l + beta_r) / 2]

    # distortions disturb the fairly regular structure of natural images.
    # This deviation can be captured by analyzing the sample distribution of
    # the products of pairs of adjacent coefficients computed along
    # horizontal, vertical and diagonal orientations.
    shifts = [[0, 1], [1, 0], [1, 1], [1, -1]]
    for i in range(len(shifts)):
        shifted_block = torch.roll(block, shifts[i], dims=(2, 3))
        alpha, beta_l, beta_r = estimate_aggd_param(block * shifted_block)
        # Eq. 8
        mean = (beta_r - beta_l) * (torch.lgamma(2 / alpha) -
                                    torch.lgamma(1 / alpha)).exp()
        feat.extend((alpha, mean, beta_l, beta_r))

    return torch.stack(feat, dim=-1)


def niqe(img: torch.Tensor,
         mu_pris_param: torch.Tensor,
         cov_pris_param: torch.Tensor,
         block_size_h: int = 96,
         block_size_w: int = 96) -> torch.Tensor:
    """Calculate NIQE (Natural Image Quality Evaluator) metric.
    Args:
        img (Tensor): Input image.
        mu_pris_param (Tensor): Mean of a pre-defined multivariate Gaussian
            model calculated on the pristine dataset.
        cov_pris_param (Tensor): Covariance of a pre-defined multivariate
            Gaussian model calculated on the pristine dataset.
        gaussian_window (Tensor): A 7x7 Gaussian window used for smoothing the image.
        block_size_h (int): Height of the blocks in to which image is divided.
            Default: 96 (the official recommended value).
        block_size_w (int): Width of the blocks in to which image is divided.
            Default: 96 (the official recommended value).
    """
    assert img.ndim == 4, (
        'Input image must be a gray or Y (of YCbCr) image with shape (b, c, h, w).'
    )
    # crop image
    b, c, h, w = img.shape
    num_block_h = math.floor(h / block_size_h)
    num_block_w = math.floor(w / block_size_w)
    img = img[..., 0:num_block_h * block_size_h, 0:num_block_w * block_size_w]

    distparam = []  # dist param is actually the multiscale features
    for scale in (1, 2):  # perform on two scales (1, 2)
        img_normalized = normalize_img_with_guass(img, padding='replicate')

        feat = []
        for idx_w in range(num_block_w):
            for idx_h in range(num_block_h):
                # process ecah block
                block = img_normalized[..., idx_h * block_size_h //
                                      scale:(idx_h + 1) * block_size_h //
                                      scale, idx_w * block_size_w //
                                      scale:(idx_w + 1) * block_size_w //
                                      scale]
                feat.append(compute_feature(block))

        distparam.append(torch.stack(feat).transpose(0, 1))

        if scale == 1:
            img = imresize(img / 255., scale=0.5, antialiasing=True)
            img = img * 255.

    distparam = torch.cat(distparam, -1)

    # fit a MVG (multivariate Gaussian) model to distorted patch features
    mu_distparam = torch.mean(distparam.masked_select(~torch.isnan(distparam)).reshape_as(distparam), axis=1)

    distparam_no_nan = distparam * (~torch.isnan(distparam))

    cov_distparam = []
    for in_b in range(b):
        sample_distparam = distparam_no_nan[in_b, ...]
        cov_distparam.append(torch_cov(sample_distparam.T))

    # compute niqe quality, Eq. 10 in the paper
    invcov_param = torch.linalg.pinv(
        (cov_pris_param + torch.stack(cov_distparam)) / 2)
    diff = (mu_pris_param - mu_distparam).unsqueeze(1)
    quality = torch.bmm(torch.bmm(diff, invcov_param),
                        diff.transpose(1, 2)).squeeze()

    quality = torch.sqrt(quality)
    return quality


def calculate_niqe(img: torch.Tensor,
                   crop_border: int = 0,
                   test_y_channel: Boolean = True,
                   pretrained_model_path: String = None,
                   **kwargs) -> torch.Tensor:
    """Calculate NIQE (Natural Image Quality Evaluator) metric.
    Args:
        img (Tensor): Input image whose quality needs to be computed.
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the metric calculation.
        test_y_channel (Bool): Whether converted to 'y' (of MATLAB YCbCr) or 'gray'.
        pretrained_model_path (String): The pretrained model path.
    Returns:
        Tensor: NIQE result.
    """

    params = scipy.io.loadmat(pretrained_model_path)
    mu_pris_param = np.ravel(params['mu_prisparam'])
    cov_pris_param = params['cov_prisparam']
    mu_pris_param = torch.from_numpy(mu_pris_param).to(img)
    cov_pris_param = torch.from_numpy(cov_pris_param).to(img)

    mu_pris_param = mu_pris_param.repeat(img.size(0), 1)
    cov_pris_param = cov_pris_param.repeat(img.size(0), 1, 1)

    if test_y_channel and img.shape[1] == 3:
        img = to_y_channel(img, 255)

    if crop_border != 0:
        img = img[..., crop_border:-crop_border, crop_border:-crop_border]

    niqe_result = niqe(img, mu_pris_param, cov_pris_param)

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

    def __init__(self,
                 channels: int = 1,
                 test_y_channel: Boolean = True,
                 crop_border: int = 0,
                 pretrained_model_path: String = None) -> None:

        super(NIQE, self).__init__()
        self.channels = channels
        self.test_y_channel = test_y_channel
        self.crop_border = crop_border
        if pretrained_model_path is not None:
            self.pretrained_model_path = pretrained_model_path
        else:
            self.pretrained_model_path = load_file_from_url(default_model_urls['url'])

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        r"""Computation of NIQE metric.
        Args:
            X: An input tensor. Shape :math:`(N, C, H, W)`.
        Returns:
            Value of niqe metric in [0, 1] range.
        """
        score = calculate_niqe(X, self.crop_border, self.test_y_channel,
                               self.pretrained_model_path)
        return score
