r"""NIQE and ILNIQE Metrics
NIQE Metric
    Created by: https://github.com/xinntao/BasicSR/blob/5668ba75eb8a77e8d2dd46746a36fee0fbb0fdcd/basicsr/metrics/niqe.py
    Modified by: Jiadi Mo (https://github.com/JiadiMo)
    Reference:
        MATLAB codes: http://live.ece.utexas.edu/research/quality/niqe_release.zip

ILNIQE Metric
    Created by: Chaofeng Chen (https://github.com/chaofengc)
    Reference:
        - Python codes: https://github.com/IceClear/IL-NIQE/blob/master/IL-NIQE.py
        - Matlab codes: https://www4.comp.polyu.edu.hk/~cslzhang/IQA/ILNIQE/Files/ILNIQE.zip
"""

import math
import numpy as np
import scipy
import scipy.io
import torch

from pyiqa.utils.color_util import to_y_channel
from pyiqa.utils.download_util import load_file_from_url
from pyiqa.matlab_utils import imresize, fspecial, conv2d, imfilter, fitweibull, nancov, nanmean, blockproc
from .func_util import estimate_aggd_param, normalize_img_with_guass, diff_round
from pyiqa.archs.fsim_arch import _construct_filters
from pyiqa.utils.registry import ARCH_REGISTRY

default_model_urls = {
    'url': 'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/niqe_modelparameters.mat',
    'niqe': 'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/niqe_modelparameters.mat',
    'ilniqe': 'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/ILNIQE_templateModel.mat',
}


def compute_feature(
    block: torch.Tensor,
    ilniqe: bool = False,
) -> torch.Tensor:
    """Compute features.
    Args:
        block (Tensor): Image block in shape (b, c, h, w).
    Returns:
        list: Features with length of 18.
    """
    bsz = block.shape[0]
    aggd_block = block[:, [0]]
    alpha, beta_l, beta_r = estimate_aggd_param(aggd_block)
    feat = [alpha, (beta_l + beta_r) / 2]

    # distortions disturb the fairly regular structure of natural images.
    # This deviation can be captured by analyzing the sample distribution of
    # the products of pairs of adjacent coefficients computed along
    # horizontal, vertical and diagonal orientations.
    shifts = [[0, 1], [1, 0], [1, 1], [1, -1]]
    for i in range(len(shifts)):
        shifted_block = torch.roll(aggd_block, shifts[i], dims=(2, 3))
        alpha, beta_l, beta_r = estimate_aggd_param(aggd_block * shifted_block)
        # Eq. 8
        mean = (beta_r - beta_l) * (torch.lgamma(2 / alpha) - torch.lgamma(1 / alpha)).exp()
        feat.extend((alpha, mean, beta_l, beta_r))
    feat = [x.reshape(bsz, 1) for x in feat]

    if ilniqe:
        tmp_block = block[:, 1:4]
        channels = 4 - 1
        shape_scale = fitweibull(tmp_block.reshape(bsz * channels, -1))
        scale_shape = shape_scale[:, [1, 0]].reshape(bsz, -1)
        feat.append(scale_shape)

        mu = torch.mean(block[:, 4:7], dim=(2, 3))
        sigmaSquare = torch.var(block[:, 4:7], dim=(2, 3))
        mu_sigma = torch.stack((mu, sigmaSquare), dim=-1).reshape(bsz, -1)
        feat.append(mu_sigma)

        channels = 85 - 7
        tmp_block = block[:, 7:85].reshape(bsz * channels, 1, *block.shape[2:])
        alpha_data, beta_l_data, beta_r_data = estimate_aggd_param(tmp_block)
        alpha_data = alpha_data.reshape(bsz, channels)
        beta_l_data = beta_l_data.reshape(bsz, channels)
        beta_r_data = beta_r_data.reshape(bsz, channels)
        alpha_beta = torch.stack([alpha_data, (beta_l_data + beta_r_data) / 2], dim=-1).reshape(bsz, -1)
        feat.append(alpha_beta)

        tmp_block = block[:, 85:109]
        channels = 109 - 85
        shape_scale = fitweibull(tmp_block.reshape(bsz * channels, -1))
        scale_shape = shape_scale[:, [1, 0]].reshape(bsz, -1)
        feat.append(scale_shape)

    feat = torch.cat(feat, dim=-1)
    return feat


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
    assert img.ndim == 4, ('Input image must be a gray or Y (of YCbCr) image with shape (b, c, h, w).')
    # crop image
    b, c, h, w = img.shape
    num_block_h = math.floor(h / block_size_h)
    num_block_w = math.floor(w / block_size_w)
    img = img[..., 0:num_block_h * block_size_h, 0:num_block_w * block_size_w]

    distparam = []  # dist param is actually the multiscale features
    for scale in (1, 2):  # perform on two scales (1, 2)
        img_normalized = normalize_img_with_guass(img, padding='replicate')

        distparam.append(blockproc(img_normalized, [block_size_h // scale, block_size_w // scale], fun=compute_feature))

        if scale == 1:
            img = imresize(img / 255., scale=0.5, antialiasing=True)
            img = img * 255.

    distparam = torch.cat(distparam, -1)

    # fit a MVG (multivariate Gaussian) model to distorted patch features
    mu_distparam = nanmean(distparam, dim=1)
    cov_distparam = nancov(distparam)

    # compute niqe quality, Eq. 10 in the paper
    invcov_param = torch.linalg.pinv((cov_pris_param + cov_distparam) / 2)
    diff = (mu_pris_param - mu_distparam).unsqueeze(1)
    quality = torch.bmm(torch.bmm(diff, invcov_param), diff.transpose(1, 2)).squeeze()

    quality = torch.sqrt(quality)
    return quality


def calculate_niqe(img: torch.Tensor,
                   crop_border: int = 0,
                   test_y_channel: bool = True,
                   pretrained_model_path: str = None,
                   color_space: str = 'yiq',
                   **kwargs) -> torch.Tensor:
    """Calculate NIQE (Natural Image Quality Evaluator) metric.
    Args:
        img (Tensor): Input image whose quality needs to be computed.
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the metric calculation.
        test_y_channel (Bool): Whether converted to 'y' (of MATLAB YCbCr) or 'gray'.
        pretrained_model_path (str): The pretrained model path.
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
        img = to_y_channel(img, 255, color_space)

    img = diff_round(img)
    img = img.to(torch.float64)

    if crop_border != 0:
        img = img[..., crop_border:-crop_border, crop_border:-crop_border]

    niqe_result = niqe(img, mu_pris_param, cov_pris_param)

    return niqe_result


def gauDerivative(sigma, in_ch=1, out_ch=1, device=None):
    halfLength = math.ceil(3 * sigma)

    x, y = np.meshgrid(
        np.linspace(-halfLength, halfLength, 2 * halfLength + 1),
        np.linspace(-halfLength, halfLength, 2 * halfLength + 1))

    gauDerX = x * np.exp(-(x**2 + y**2) / 2 / sigma / sigma)
    gauDerY = y * np.exp(-(x**2 + y**2) / 2 / sigma / sigma)

    dx = torch.from_numpy(gauDerX).to(device)
    dy = torch.from_numpy(gauDerY).to(device)
    dx = dx.repeat(out_ch, in_ch, 1, 1)
    dy = dy.repeat(out_ch, in_ch, 1, 1)

    return dx, dy


def ilniqe(img: torch.Tensor,
           mu_pris_param: torch.Tensor,
           cov_pris_param: torch.Tensor,
           principleVectors: torch.Tensor,
           meanOfSampleData: torch.Tensor,
           resize: bool = True,
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

    EPS = 1e-8
    scales = 3
    orientations = 4
    infConst = 10000
    nanConst = 2000

    if resize:
        img = imresize(img, sizes=(normalizedWidth, normalizedWidth))
        img = img.clamp(0.0, 255.0)

    # crop image
    b, c, h, w = img.shape
    num_block_h = math.floor(h / block_size_h)
    num_block_w = math.floor(w / block_size_w)
    img = img[..., 0:num_block_h * block_size_h, 0:num_block_w * block_size_w]
    ospace_weight = torch.tensor([
        [0.3, 0.04, -0.35],
        [0.34, -0.6, 0.17],
        [0.06, 0.63, 0.27],
    ]).to(img)

    O_img = img.permute(0, 2, 3, 1) @ ospace_weight.T
    O_img = O_img.permute(0, 3, 1, 2)

    distparam = []  # dist param is actually the multiscale features
    for scale in (1, 2):  # perform on two scales (1, 2)
        struct_dis = normalize_img_with_guass(O_img[:, [2]], kernel_size=5, sigma=5. / 6, padding='replicate')

        dx, dy = gauDerivative(sigmaForGauDerivative / (scale**scaleFactorForGaussianDer), device=img)

        Ix = conv2d(O_img, dx.repeat(3, 1, 1, 1), groups=3)
        Iy = conv2d(O_img, dy.repeat(3, 1, 1, 1), groups=3)
        GM = torch.sqrt(Ix**2 + Iy**2 + EPS)
        Ixy = torch.stack((Ix, Iy), dim=2).reshape(Ix.shape[0], Ix.shape[1] * 2,
                                                   *Ix.shape[2:])  # reshape to (IxO1, IxO1, IxO2, IyO2, IxO3, IyO3)

        logRGB = torch.log(img + KforLog)
        logRGBMS = logRGB - logRGB.mean(dim=(2, 3), keepdim=True)

        Intensity = logRGBMS.sum(dim=1, keepdim=True) / np.sqrt(3)
        BY = (logRGBMS[:, [0]] + logRGBMS[:, [1]] - 2 * logRGBMS[:, [2]]) / np.sqrt(6)
        RG = (logRGBMS[:, [0]] - logRGBMS[:, [1]]) / np.sqrt(2)

        compositeMat = torch.cat([struct_dis, GM, Intensity, BY, RG, Ixy], dim=1)

        O3 = O_img[:, [2]]
        # gabor filter in shape (b, ori * scale, h, w)
        LGFilters = _construct_filters(
            O3,
            scales=scales,
            orientations=orientations,
            min_length=minWaveLength / (scale**scaleFactorForLoG),
            sigma_f=sigmaOnf,
            mult=mult,
            delta_theta=dThetaOnSigma,
            use_lowpass_filter=False)
        # reformat to scale * ori
        b, _, h, w = LGFilters.shape
        LGFilters = LGFilters.reshape(b, orientations, scales, h, w).transpose(1, 2).reshape(b, -1, h, w)
        # TODO: current filters needs to be transposed to get same results as matlab, find the bug
        LGFilters = LGFilters.transpose(-1, -2)
        fftIm = torch.fft.fft2(O3)

        logResponse = []
        partialDer = []
        GM = []
        for index in range(LGFilters.shape[1]):
            filter = LGFilters[:, [index]]
            response = torch.fft.ifft2(filter * fftIm)
            realRes = torch.real(response)
            imagRes = torch.imag(response)

            partialXReal = conv2d(realRes, dx)
            partialYReal = conv2d(realRes, dy)
            realGM = torch.sqrt(partialXReal**2 + partialYReal**2 + EPS)

            partialXImag = conv2d(imagRes, dx)
            partialYImag = conv2d(imagRes, dy)
            imagGM = torch.sqrt(partialXImag**2 + partialYImag**2 + EPS)

            logResponse.append(realRes)
            logResponse.append(imagRes)
            partialDer.append(partialXReal)
            partialDer.append(partialYReal)
            partialDer.append(partialXImag)
            partialDer.append(partialYImag)
            GM.append(realGM)
            GM.append(imagGM)
        logResponse = torch.cat(logResponse, dim=1)
        partialDer = torch.cat(partialDer, dim=1)
        GM = torch.cat(GM, dim=1)
        compositeMat = torch.cat((compositeMat, logResponse, partialDer, GM), dim=1)

        distparam.append(blockproc(compositeMat, [block_size_h // scale,
                         block_size_w // scale], fun=compute_feature, ilniqe=True))

        gauForDS = fspecial(math.ceil(6 * sigmaForDownsample), sigmaForDownsample).to(img)
        filterResult = imfilter(O_img, gauForDS.repeat(3, 1, 1, 1), padding='replicate', groups=3)
        O_img = filterResult[..., ::2, ::2]
        filterResult = imfilter(img, gauForDS.repeat(3, 1, 1, 1), padding='replicate', groups=3)
        img = filterResult[..., ::2, ::2]

    distparam = torch.cat(distparam, dim=-1)  # b, block_num, feature_num
    distparam[distparam > infConst] = infConst

    # fit a MVG (multivariate Gaussian) model to distorted patch features
    coefficientsViaPCA = torch.bmm(
        principleVectors.transpose(1, 2), (distparam - meanOfSampleData.unsqueeze(1)).transpose(1, 2))
    final_features = coefficientsViaPCA.transpose(1, 2)
    b, blk_num, feat_num = final_features.shape

    # remove block features with nan and compute nonan cov
    cov_distparam = nancov(final_features)

    # replace nan in final features with mu
    mu_final_features = nanmean(final_features, dim=1, keepdim=True)
    final_features_withmu = torch.where(torch.isnan(final_features), mu_final_features, final_features)

    # compute ilniqe quality
    invcov_param = torch.linalg.pinv((cov_pris_param + cov_distparam) / 2)
    diff = final_features_withmu - mu_pris_param.unsqueeze(1)
    quality = (torch.bmm(diff, invcov_param) * diff).sum(dim=-1)
    quality = torch.sqrt(quality).mean(dim=1)

    return quality


def calculate_ilniqe(img: torch.Tensor,
                     crop_border: int = 0,
                     pretrained_model_path: str = None,
                     **kwargs) -> torch.Tensor:
    """Calculate IL-NIQE metric.
    Args:
        img (Tensor): Input image whose quality needs to be computed.
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the metric calculation.
        pretrained_model_path (str): The pretrained model path.
    Returns:
        Tensor: IL-NIQE result.
    """

    params = scipy.io.loadmat(pretrained_model_path)
    img = img * 255.
    img = diff_round(img)
    # float64 precision is critical to be consistent with matlab codes
    img = img.to(torch.float64)

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
        test_y_channel (bool): whether to use y channel on ycbcr.
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the metric calculation.
        pretrained_model_path (str): The pretrained model path.
    References:
        Mittal, Anish, Rajiv Soundararajan, and Alan C. Bovik.
        "Making a “completely blind” image quality analyzer."
        IEEE Signal Processing Letters (SPL) 20.3 (2012): 209-212.
    """

    def __init__(self,
                 channels: int = 1,
                 test_y_channel: bool = True,
                 color_space: str = 'yiq',
                 crop_border: int = 0,
                 pretrained_model_path: str = None) -> None:

        super(NIQE, self).__init__()
        self.channels = channels
        self.test_y_channel = test_y_channel
        self.color_space = color_space
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
        score = calculate_niqe(X, self.crop_border, self.test_y_channel, self.pretrained_model_path, self.color_space)
        return score


@ARCH_REGISTRY.register()
class ILNIQE(torch.nn.Module):
    r"""Args:
        channels (int): Number of processed channel.
        test_y_channel (bool): whether to use y channel on ycbcr.
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the metric calculation.
        pretrained_model_path (str): The pretrained model path.
    References:
        Zhang, Lin, Lei Zhang, and Alan C. Bovik. "A feature-enriched
        completely blind image quality evaluator." IEEE Transactions
        on Image Processing 24.8 (2015): 2579-2591.
    """

    def __init__(self, channels: int = 3, crop_border: int = 0, pretrained_model_path: str = None) -> None:

        super(ILNIQE, self).__init__()
        self.channels = channels
        self.crop_border = crop_border
        if pretrained_model_path is not None:
            self.pretrained_model_path = pretrained_model_path
        else:
            self.pretrained_model_path = load_file_from_url(default_model_urls['ilniqe'])

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        r"""Computation of NIQE metric.
        Args:
            X: An input tensor. Shape :math:`(N, C, H, W)`.
        Returns:
            Value of niqe metric in [0, 1] range.
        """
        score = calculate_ilniqe(X, self.crop_border, self.pretrained_model_path)
        return score
