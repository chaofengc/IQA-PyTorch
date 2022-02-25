r"""NIQE Metric

Created by: https://github.com/xinntao/BasicSR/blob/5668ba75eb8a77e8d2dd46746a36fee0fbb0fdcd/basicsr/metrics/niqe.py

Modified by: Jiadi Mo (https://github.com/JiadiMo)

Reference:
    MATLAB codes: http://live.ece.utexas.edu/research/quality/niqe_release.zip

"""

import math
from traceback import print_tb
import numpy as np
import scipy
import scipy.io
import scipy.misc
import scipy.ndimage
import scipy.special
import torch
from scipy.signal import convolve2d
import torch.nn.functional as F
import torch.nn as nn
from tokenize import String
from typing import Tuple
from xmlrpc.client import Boolean
from scipy.stats import exponweib
from scipy.optimize import fmin

from pyiqa.utils.color_util import to_y_channel
from pyiqa.utils.download_util import load_file_from_url
from pyiqa.utils.matlab_functions import imresize, fspecial_gauss
from pyiqa.utils.resize import padding
from .func_util import estimate_aggd_param, torch_cov, normalize_img_with_guass
from pyiqa.archs.fsim_arch import _construct_filters
from pyiqa.utils.registry import ARCH_REGISTRY

default_model_urls = {
    'url': 'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/niqe_modelparameters.mat'
}


def matlab_fspecial(shape=(3, 3), sigma=0.5, channels=1):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    h = torch.from_numpy(h).float().unsqueeze(0).unsqueeze(0)
    return h.repeat(channels, 1, 1, 1)
    # return h

def fitweibull(x):
   def optfun(theta):
      return -np.sum(np.log(exponweib.pdf(x, 1, theta[0], scale = theta[1], loc = 0)))
   logx = np.log(x)
   shape = 1.2 / np.std(logx)
   scale = np.exp(np.mean(logx) + (0.572 / shape))
   return fmin(optfun, [shape, scale], xtol = 0.01, ftol = 0.01, disp = 0)

def gauDerivative(sigma):
    halfLength = math.ceil(3 * sigma)

    x, y = np.meshgrid(
        np.linspace(-halfLength, halfLength, 2 * halfLength + 1),
        np.linspace(-halfLength, halfLength, 2 * halfLength + 1))

    gauDerX = x * np.exp(-(x**2 + y**2) / 2 / sigma / sigma)
    gauDerY = y * np.exp(-(x**2 + y**2) / 2 / sigma / sigma)

    return gauDerX, gauDerY


def conv2d_same_padding(input, weight, bias=None, stride=1, dilation=1, groups=1):
    # padding=same
    input_rows = input.size(2)
    filter_rows = weight.size(2)
    effective_filter_size_rows = (filter_rows - 1) * dilation + 1
    out_rows = (input_rows + stride - 1) // stride
    padding_rows = max(0, (out_rows - 1) * stride + (filter_rows - 1) * dilation + 1 - input_rows)
    rows_odd = (padding_rows % 2 != 0)
    padding_cols = max(0, (out_rows - 1) * stride + (filter_rows - 1) * dilation + 1 - input_rows)
    cols_odd = (padding_rows % 2 != 0)

    if rows_odd or cols_odd:
        input = F.pad(input, [0, int(cols_odd), 0, int(rows_odd)])

    return F.conv2d(
        input, weight, bias, stride, padding=(padding_rows // 2, padding_cols // 2), dilation=dilation, groups=groups)


def conv2(x, y, mode='same'):
    return np.rot90(convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2)


def compute_feature(feature_list: torch.Tensor, block_pos: Tuple = None, il_niqe: Boolean = False) -> torch.Tensor:
    """Compute features.
    Args:
        feature_list (Tensor): Image in shape (b, c, h, w).
    Returns:
        list: Features with length of 18.
    """
    if il_niqe:
        block = feature_list[0, ..., block_pos[0]:block_pos[1], block_pos[2]:block_pos[3]]
    else:
        block = feature_list[..., block_pos[0]:block_pos[1], block_pos[2]:block_pos[3]]

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
        mean = (beta_r - beta_l) * (torch.lgamma(2 / alpha) - torch.lgamma(1 / alpha)).exp()
        feat.extend((alpha, mean, beta_l, beta_r))

    if il_niqe:
        for i in range(1, 4):
            block = feature_list[i, ..., block_pos[0]:block_pos[1], block_pos[2]:block_pos[3]]
            shape_batch, scale_batch = [], []
            for b in range(block.size(0)):
                shape, scale = fitweibull(block[b,...].numpy().flatten('F'))
                shape_batch.append(torch.Tensor([shape]))
                scale_batch.append(torch.Tensor([scale]))
            
            feat.extend([torch.cat(scale_batch), torch.cat(shape_batch)])

        for i in range(4, 7):
            block = feature_list[i, ..., block_pos[0]:block_pos[1], block_pos[2]:block_pos[3]]
            mu = torch.mean(block, dim=(1,2,3))
            sigmaSquare = torch.var(block,dim=(1,2,3))
            feat.extend([mu, sigmaSquare])

        for i in range(7, 85):
            block = feature_list[i, ..., block_pos[0]:block_pos[1], block_pos[2]:block_pos[3]]
            alpha_data, beta_l_data, beta_r_data = estimate_aggd_param(block)
            feat.extend([alpha_data, (beta_l_data + beta_r_data) / 2])

        for i in range(85, 109):
            block = feature_list[i, ..., block_pos[0]:block_pos[1], block_pos[2]:block_pos[3]]
            # shape, scale = fitweibull(block.flatten('F'))
            # feat.extend([scale, shape])
            shape_batch, scale_batch = [], []
            for b in range(block.size(0)):
                shape, scale = fitweibull(block[b,...].numpy().flatten('F'))
                shape_batch.append(torch.Tensor([shape]))
                scale_batch.append(torch.Tensor([scale]))
            
            feat.extend([torch.cat(scale_batch), torch.cat(shape_batch)])
            

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
        img_normalized = normalize_img_with_guass(img, padding='replicate')

        feat = []
        for idx_w in range(num_block_w):
            for idx_h in range(num_block_h):
                # process ecah block
                # block = img_normalized[..., idx_h * block_size_h // scale:(idx_h + 1) * block_size_h // scale,
                #                        idx_w * block_size_w // scale:(idx_w + 1) * block_size_w // scale]
                block_pos = [
                    idx_h * block_size_h // scale, (idx_h + 1) * block_size_h // scale, idx_w * block_size_w // scale,
                    (idx_w + 1) * block_size_w // scale
                ]
                feat.append(compute_feature(img_normalized, block_pos))

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
    invcov_param = torch.linalg.pinv((cov_pris_param + torch.stack(cov_distparam)) / 2)
    diff = (mu_pris_param - mu_distparam).unsqueeze(1)
    quality = torch.bmm(torch.bmm(diff, invcov_param), diff.transpose(1, 2)).squeeze()

    quality = torch.sqrt(quality)
    return quality


def ilniqe(img: torch.Tensor,
           mu_pris_param: torch.Tensor,
           cov_pris_param: torch.Tensor,
           principleVectors: torch.Tensor,
           meanOfSampleData: torch.Tensor,
           resize: Boolean = True,
           block_size_h: int = 84,
           block_size_w: int = 84) -> torch.Tensor:
    """Calculate IL-NIQE (Integrated Local Natural Image Quality Evaluator) metric.
    Args:
        img (Tensor): Input image.
        mu_pris_param (Tensor): Mean of a pre-defined multivariate Gaussian
            model calculated on the pristine dataset.
        cov_pris_param (Tensor): Covariance of a pre-defined multivariate
            Gaussian model calculated on the pristine dataset.
        principleVectors (Tensor): Features from official .mat file.
        meanOfSampleData (Tensor): Features from official .mat file.
        resize (Bloolean): resize image. Default: True.
        block_size_h (int): Height of the blocks in to which image is divided.
            Default: 84 (the official recommended value).
        block_size_w (int): Width of the blocks in to which image is divided.
            Default: 84 (the official recommended value).
    """
    assert img.ndim == 4, ('Input image must be a gray or Y (of YCbCr) image with shape (b, c, h, w).')

    blockrowoverlap = 0
    blockcoloverlap = 0
    sigmaForGauDerivative = 1.66
    KforLog = 0.00001
    normalizedWidth = 524
    minWaveLength = 2.4
    sigmaOnf = 0.55
    mult = 1.31
    dThetaOnSigma = 1.10
    scaleFactorForLoG = 0.87
    scaleFactorForGaussianDer = 0.28
    sigmaForDownsample = 0.9

    infConst = 10000
    nanConst = 2000

    scales = 3
    orientations = 4

    if resize:
        img = img * 255
        img = img.type(torch.uint8)
        img = imresize(img, sizes=(normalizedWidth, normalizedWidth))
        img = img.type(torch.float32)
    else:
        img = img * 255
        img = img.round()

    # crop image
    b, c, h, w = img.shape
    num_block_h = math.floor(h / block_size_h)
    num_block_w = math.floor(w / block_size_w)
    img = img[..., 0:num_block_h * block_size_h, 0:num_block_w * block_size_w]

    O1 = 0.3 * img[:, [0], ...] + 0.04 * img[:, [1], ...] - 0.35 * img[:, [2], ...]
    O2 = 0.34 * img[:, [0], ...] - 0.6 * img[:, [1], ...] + 0.17 * img[:, [2], ...]
    O3 = 0.06 * img[:, [0], ...] + 0.63 * img[:, [1], ...] + 0.27 * img[:, [2], ...]

    RChannel = img[:, [0], ...]
    GChannel = img[:, [1], ...]
    BChannel = img[:, [2], ...]

    distparam = []  # dist param is actually the multiscale features
    for scale in (1, 2):  # perform on two scales (1, 2)
        print(scale)
        O3_normalized = normalize_img_with_guass(O3, kernel_size=5, sigma=5 / 6, padding='replicate')

        dx, dy = gauDerivative(sigmaForGauDerivative / (scale**scaleFactorForGaussianDer))
        dx = torch.from_numpy(dx).to(img)
        dy = torch.from_numpy(dy).to(img)
        dx = dx.repeat(1, 1, 1, 1).to(img)
        dy = dy.repeat(1, 1, 1, 1).to(img)

        IxO1 = -conv2d_same_padding(O1, dx)
        IyO1 = -conv2d_same_padding(O1, dy)
        GMO1 = torch.sqrt(IxO1**2 + IyO1**2) + torch.finfo(O1.dtype).eps
        print(GMO1.shape)

        IxO2 = -conv2d_same_padding(O2, dx)
        IyO2 = -conv2d_same_padding(O2, dy)
        GMO2 = torch.sqrt(IxO2**2 + IyO2**2) + torch.finfo(O2.dtype).eps
        print(IxO2.shape)
        IxO3 = -conv2d_same_padding(O3, dx)
        IyO3 = -conv2d_same_padding(O3, dy)
        GMO3 = torch.sqrt(IxO3**2 + IyO3**2) + torch.finfo(O3.dtype).eps

        logR = torch.log(RChannel + KforLog)
        logG = torch.log(GChannel + KforLog)
        logB = torch.log(BChannel + KforLog)
        logRMS = logR - torch.mean(logR, dim=[2, 3]).view(b,1,1,1)
        logGMS = logG - torch.mean(logG, dim=[2, 3]).view(b,1,1,1)
        logBMS = logB - torch.mean(logB, dim=[2, 3]).view(b,1,1,1)
        print(logRMS)
        Intensity = (logRMS + logGMS + logBMS) / torch.sqrt(torch.Tensor([3]).to(img))
        BY = (logRMS + logGMS - 2 * logBMS) / torch.sqrt(torch.Tensor([6]).to(img))
        RG = (logRMS - logGMS) / torch.sqrt(torch.Tensor([2]).to(img))

        compositeMat = [O3_normalized, GMO1, GMO2, GMO3, Intensity, BY, RG, IxO1, IyO1, IxO2, IyO2, IxO3, IyO3]

        # LGFilters = logGabors(h, w, minWaveLength / (scale**scaleFactorForLoG), sigmaOnf, mult, dThetaOnSigma)
        LGFilters = _construct_filters(
            O3,
            scales=scales,
            orientations=orientations,
            min_length=minWaveLength / (scale**scaleFactorForLoG),
            sigma_f=sigmaOnf,
            mult=mult,
            delta_theta=dThetaOnSigma,
            lowpass_filter=False,
            combine_result=False)
        # fftIm = np.fft.fft2(O3)
        fftIm = torch.fft.fft2(O3)
        print(fftIm.shape)

        logResponse = []
        partialDer = []
        GM = []
        for index in range(scales):
            for oriIndex in range(orientations):
                # response = np.fft.ifft2(LGFilters[scaleIndex][oriIndex]*fftIm)
                response = torch.fft.ifft2(LGFilters[:, [index], [oriIndex], ...] * fftIm)
                # realRes = np.real(response)
                # imagRes = np.imag(response)
                realRes = torch.real(response)
                imagRes = torch.imag(response)

                # compRes = conv2(realRes, dx + 1j*dy, 'same')
                # partialXReal = np.real(compRes)
                # partialYReal = np.imag(compRes)
                # realGM = np.sqrt(partialXReal**2 + partialYReal**2) + np.finfo(partialXReal.dtype).eps

                partialXReal = conv2d_same_padding(realRes, dx)
                partialYReal = conv2d_same_padding(realRes, dy)
                realGM = torch.sqrt(partialXReal**2 + partialYReal**2) + torch.finfo(partialXReal.dtype).eps

                # compRes = conv2(imagRes, dx + 1j*dy, 'same')
                # partialXImag = np.real(compRes)
                # partialYImag = np.imag(compRes)
                # imagGM = np.sqrt(partialXImag**2 + partialYImag**2) + np.finfo(partialXImag.dtype).eps

                partialXImag = conv2d_same_padding(imagRes, dx)
                partialYImag = conv2d_same_padding(imagRes, dy)
                imagGM = torch.sqrt(partialXImag**2 + partialYImag**2) + torch.finfo(partialXImag.dtype).eps

                logResponse.append(realRes)
                logResponse.append(imagRes)
                partialDer.append(partialXReal)
                partialDer.append(partialYReal)
                partialDer.append(partialXImag)
                partialDer.append(partialYImag)
                GM.append(realGM)
                GM.append(imagGM)

        compositeMat.extend(logResponse)
        compositeMat.extend(partialDer)
        compositeMat.extend(GM)
        compositeMat = torch.stack(compositeMat)

        feat = []
        for idx_w in range(num_block_w):
            for idx_h in range(num_block_h):
                # process ecah block
                block_pos = [
                    idx_h * block_size_h // scale, (idx_h + 1) * block_size_h // scale, idx_w * block_size_w // scale,
                    (idx_w + 1) * block_size_w // scale
                ]
                # block = compositeMat[0, ..., block_pos[0]:block_pos[1], block_pos[2]:block_pos[3]]
                feat.append(compute_feature(compositeMat, block_pos, il_niqe=True))

        distparam.append(torch.stack(feat).transpose(0, 1))
        print("=====", torch.stack(feat).transpose(0, 1).shape)
        gauForDS = matlab_fspecial((math.ceil(6 * sigmaForDownsample), math.ceil(6 * sigmaForDownsample)),
                                   sigmaForDownsample, 1).to(img)

        # filterResult = convolve(O1, gauForDS, mode='nearest')
        pad_func = nn.ReplicationPad2d(math.ceil(6 * sigmaForDownsample) // 2 - 1)
        filterResult = F.conv2d(pad_func(O1), gauForDS, groups=1)
        print(filterResult.shape)
        O1 = filterResult[..., ::2, ::2]
        # filterResult = convolve(O2, gauForDS, mode='nearest')
        filterResult = F.conv2d(pad_func(O2), gauForDS, groups=1)
        O2 = filterResult[..., 0::2, 0::2]
        print(O2.shape)
        print(O2)
        # filterResult = convolve(O3, gauForDS, mode='nearest')
        filterResult = F.conv2d(pad_func(O3), gauForDS, groups=1)
        O3 = filterResult[..., 0::2, 0::2]

        # filterResult = convolve(RChannel, gauForDS, mode='nearest')
        filterResult = F.conv2d(pad_func(RChannel), gauForDS, groups=1)
        RChannel = filterResult[..., 0::2, 0::2]
        print(RChannel.shape)
        print(RChannel)
        # filterResult = convolve(GChannel, gauForDS, mode='nearest')
        filterResult = F.conv2d(pad_func(GChannel), gauForDS, groups=1)
        GChannel = filterResult[..., 0::2, 0::2]
        # filterResult = convolve(BChannel, gauForDS, mode='nearest')
        filterResult = F.conv2d(pad_func(BChannel), gauForDS, groups=1)
        BChannel = filterResult[..., 0::2, 0::2]

        # if scale == 1:
        #     img = imresize(img / 255., scale=0.5, antialiasing=True)
        #     img = img * 255.

    distparam = torch.cat(distparam, -1)
    print(distparam.shape)

    # fit a MVG (multivariate Gaussian) model to distorted patch features
    distparam[distparam > infConst] = infConst
    meanMatrix = torch.tile(meanOfSampleData.unsqueeze(-1), (1, 1,distparam.shape[1])) # b, 468, 36
    coefficientsViaPCA = torch.bmm(principleVectors.transpose(1,2), (distparam.transpose(1,2) - meanMatrix)) #b, 430,36
    final_features = coefficientsViaPCA.transpose(1,2)

    mu_distparam = torch.mean(
        final_features.masked_select(~torch.isnan(final_features)).reshape_as(final_features), axis=1)
    mu_distparam[torch.isnan(mu_distparam)] = nanConst

    distparam_no_nan = final_features * (~torch.isnan(final_features))

    cov_distparam = []
    for in_b in range(b):
        sample_distparam = distparam_no_nan[in_b, ...]
        cov_distparam.append(torch_cov(sample_distparam.T))

    # compute niqe quality, Eq. 10 in the paper
    invcov_param = torch.linalg.pinv((cov_pris_param + torch.stack(cov_distparam)) / 2)

    dist = []
    for index_feature in range(final_features.size(1)):
        currentFea = final_features[:,index_feature,:]
        currentFea = torch.where(torch.isnan(currentFea), mu_distparam, currentFea)
        diff = (currentFea - mu_pris_param).unsqueeze(1)
        thisDist = torch.bmm(
            torch.bmm(diff, invcov_param), diff.transpose(1, 2)).squeeze()
        dist.append(torch.sqrt(thisDist))
    
    # quality = torch.bmm(torch.bmm(diff, invcov_param), diff.transpose(1, 2)).squeeze()
    quality = torch.mean(torch.stack(dist), dim=(0))
    
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


def calculate_ilniqe(img: torch.Tensor,
                     crop_border: int = 0,
                     pretrained_model_path: String = None,
                     **kwargs) -> torch.Tensor:
    """Calculate IL-NIQE metric.
    Args:
        img (Tensor): Input image whose quality needs to be computed.
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the metric calculation.
        pretrained_model_path (String): The pretrained model path.
    Returns:
        Tensor: IL-NIQE result.
    """

    params = scipy.io.loadmat(pretrained_model_path)

    mu_pris_param = np.ravel(params['templateModel'][0][0])
    cov_pris_param = params['templateModel'][0][1]
    meanOfSampleData = np.ravel(params['templateModel'][0][2])
    principleVectors = params['templateModel'][0][3]

    mu_pris_param = torch.from_numpy(mu_pris_param).to(img)
    cov_pris_param = torch.from_numpy(cov_pris_param).to(img)
    meanOfSampleData = torch.from_numpy(meanOfSampleData).to(img)
    principleVectors = torch.from_numpy(principleVectors).to(img)

    mu_pris_param = mu_pris_param.repeat(img.size(0), 1)
    cov_pris_param = cov_pris_param.repeat(img.size(0), 1, 1)
    meanOfSampleData = meanOfSampleData.repeat(img.size(0), 1)
    principleVectors = principleVectors.repeat(img.size(0), 1, 1)

    if crop_border != 0:
        img = img[..., crop_border:-crop_border, crop_border:-crop_border]

    ilniqe_result = ilniqe(img, mu_pris_param, cov_pris_param, principleVectors, meanOfSampleData)

    return ilniqe_result


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
        score = calculate_niqe(X, self.crop_border, self.test_y_channel, self.pretrained_model_path)
        return score


@ARCH_REGISTRY.register()
class IL_NIQE(torch.nn.Module):
    r"""Args:
        channels (int): Number of processed channel.
        test_y_channel (Boolean): whether to use y channel on ycbcr.
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the metric calculation.
        pretrained_model_path (String): The pretrained model path.
    References:
        Zhang, Lin, Lei Zhang, and Alan C. Bovik. "A feature-enriched 
        completely blind image quality evaluator." IEEE Transactions 
        on Image Processing 24.8 (2015): 2579-2591.
    """

    def __init__(self, channels: int = 3, crop_border: int = 0, pretrained_model_path: String = None) -> None:

        super(IL_NIQE, self).__init__()
        self.channels = channels
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
        score = calculate_ilniqe(X, self.crop_border, self.pretrained_model_path)
        return score
