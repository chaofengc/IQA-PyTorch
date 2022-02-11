r"""NIQE Metric

Created by: https://github.com/photosynthesis-team/piq/blob/master/piq/brisque.py

Modified by: Jiadi Mo (https://github.com/JiadiMo)

Reference:
    MATLAB codes: https://live.ece.utexas.edu/research/Quality/index_algorithms.htm BRISQUE;
    Pretrained model from: https://github.com/photosynthesis-team/piq/releases/download/v0.4.0/brisque_svm_weights.pt
    
"""
from tokenize import String
from typing import Union, Tuple
from xmlrpc.client import Boolean
import torch
import torch.nn.functional as F
from pyiqa.archs.ssim_arch import to_y_channel, fspecial_gauss
from pyiqa.utils.matlab_functions import imresize
from pyiqa.utils.registry import ARCH_REGISTRY
from pyiqa.utils.matlab_functions import imresize


def brisque(x: torch.Tensor,
            kernel_size: int = 7,
            kernel_sigma: float = 7 / 6,
            data_range: Union[int, float] = 1.,
            test_y_channel: Boolean = True,
            pretrained_model_path: String = None) -> torch.Tensor:
    r"""Interface of BRISQUE index.

    Args:
        x: An input tensor. Shape :math:`(N, C, H, W)`.
        kernel_size: The side-length of the sliding window used in comparison. Must be an odd value.
        kernel_sigma: Sigma of normal distribution.
        data_range: Maximum value range of images (usually 1.0 or 255).
        to_y_channel: Whether use the y-channel of YCBCR.
        pretrained_model_path: The model path.

    Returns:
        Value of BRISQUE index.

    References:
        Mittal, Anish, Anush Krishna Moorthy, and Alan Conrad Bovik. 
        "No-reference image quality assessment in the spatial domain." 
        IEEE Transactions on image processing 21, no. 12 (2012): 4695-4708.

    """

    if test_y_channel and x.size(1) == 3:
        x = x / float(data_range)
        x = to_y_channel(x)
    else:
        x = x / float(data_range) * 255

    features = []
    num_of_scales = 2
    for _ in range(num_of_scales):
        features.append(natural_scene_statistics(x, kernel_size, kernel_sigma))
        x = imresize(x / 255., scale=0.5, antialiasing=True) * 255

    features = torch.cat(features, dim=-1)
    scaled_features = scale_features(features)

    if pretrained_model_path:
        sv_coef, sv = torch.load(pretrained_model_path)

    # gamma and rho are SVM model parameters taken from official implementation of BRISQUE on MATLAB
    # Source: https://live.ece.utexas.edu/research/Quality/index_algorithms.htm
    gamma = 0.05
    rho = -153.591
    sv.t_()
    kernel_features = rbf_kernel(features=scaled_features, sv=sv, gamma=gamma)
    score = kernel_features @ sv_coef
    return score - rho


def estimate_ggd_param(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    gamma = torch.arange(0.2, 10 + 0.001, 0.001).to(x)
    r_table = (torch.lgamma(1. / gamma) + torch.lgamma(3. / gamma) -
               2 * torch.lgamma(2. / gamma)).exp()
    r_table = r_table.repeat(x.size(0), 1)

    sigma_sq = x.pow(2).mean(dim=(-1, -2))
    sigma = sigma_sq.sqrt().squeeze(dim=-1)

    assert not torch.isclose(sigma, torch.zeros_like(sigma)).all(), \
        'Expected image with non zero variance of pixel values'

    E = x.abs().mean(dim=(-1, -2))
    rho = sigma_sq / E**2

    indexes = (rho - r_table).abs().argmin(dim=-1)
    solution = gamma[indexes]
    return solution, sigma


def estimate_aggd_param(
        x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gamma = torch.arange(0.2, 10 + 0.001, 0.001).to(x)
    r_table = torch.exp(2 * torch.lgamma(2. / gamma) -
                        torch.lgamma(1. / gamma) - torch.lgamma(3. / gamma))
    r_table = r_table.repeat(x.size(0), 1)

    mask_left = x < 0
    mask_right = x > 0
    count_left = mask_left.sum(dim=(-1, -2), dtype=torch.float32)
    count_right = mask_right.sum(dim=(-1, -2), dtype=torch.float32)

    assert (count_left > 0).all(), 'Expected input tensor (pairwise products of neighboring MSCN coefficients)' \
                                   '  with values below zero to compute parameters of AGGD'
    assert (count_right > 0).all(), 'Expected input tensor (pairwise products of neighboring MSCN coefficients)' \
                                    ' with values above zero to compute parameters of AGGD'

    left_sigma = ((x * mask_left).pow(2).sum(dim=(-1, -2)) / count_left).sqrt()
    right_sigma = ((x * mask_right).pow(2).sum(dim=(-1, -2)) /
                   count_right).sqrt()

    assert (left_sigma > 0).all() and (right_sigma > 0).all(), f'Expected non-zero left and right variances, ' \
                                                               f'got {left_sigma} and {right_sigma}'

    gamma_hat = left_sigma / right_sigma
    ro_hat = x.abs().mean(dim=(-1, -2)).pow(2) / x.pow(2).mean(dim=(-1, -2))
    ro_hat_norm = (ro_hat * (gamma_hat.pow(3) + 1) *
                   (gamma_hat + 1)) / (gamma_hat.pow(2) + 1).pow(2)

    indexes = (ro_hat_norm - r_table).abs().argmin(dim=-1)
    solution = gamma[indexes]
    return solution, left_sigma.squeeze(dim=-1), right_sigma.squeeze(dim=-1)


def natural_scene_statistics(luma: torch.Tensor,
                             kernel_size: int = 7,
                             sigma: float = 7. / 6) -> torch.Tensor:
    kernel = fspecial_gauss(kernel_size, sigma, 1).to(luma)
    C = 1
    mu = F.conv2d(luma, kernel, padding=kernel_size // 2)
    mu_sq = mu**2
    std = F.conv2d(luma**2, kernel, padding=kernel_size // 2)
    std = ((std - mu_sq).abs().sqrt())

    luma_nrmlzd = (luma - mu) / (std + C)
    alpha, sigma = estimate_ggd_param(luma_nrmlzd)
    features = [alpha, sigma.pow(2)]

    shifts = [(0, 1), (1, 0), (1, 1), (-1, 1)]

    for shift in shifts:
        shifted_luma_nrmlzd = torch.roll(luma_nrmlzd,
                                         shifts=shift,
                                         dims=(-2, -1))
        alpha, sigma_l, sigma_r = estimate_aggd_param(luma_nrmlzd *
                                                      shifted_luma_nrmlzd)
        eta = (sigma_r - sigma_l) * torch.exp(
            torch.lgamma(2. / alpha) -
            (torch.lgamma(1. / alpha) + torch.lgamma(3. / alpha)) / 2)
        features.extend((alpha, eta, sigma_l.pow(2), sigma_r.pow(2)))

    return torch.stack(features, dim=-1)


def scale_features(features: torch.Tensor) -> torch.Tensor:
    lower_bound = -1
    upper_bound = 1
    # Feature range is taken from official implementation of BRISQUE on MATLAB.
    # Source: https://live.ece.utexas.edu/research/Quality/index_algorithms.htm
    feature_ranges = torch.tensor([[0.338, 10], [0.017204, 0.806612],
                                   [0.236, 1.642], [-0.123884, 0.20293],
                                   [0.000155, 0.712298], [0.001122, 0.470257],
                                   [0.244, 1.641], [-0.123586, 0.179083],
                                   [0.000152, 0.710456], [0.000975, 0.470984],
                                   [0.249, 1.555], [-0.135687, 0.100858],
                                   [0.000174, 0.684173], [0.000913, 0.534174],
                                   [0.258, 1.561], [-0.143408, 0.100486],
                                   [0.000179, 0.685696], [0.000888, 0.536508],
                                   [0.471, 3.264], [0.012809, 0.703171],
                                   [0.218, 1.046], [-0.094876, 0.187459],
                                   [1.5e-005, 0.442057], [0.001272, 0.40803],
                                   [0.222, 1.042], [-0.115772, 0.162604],
                                   [1.6e-005, 0.444362], [0.001374, 0.40243],
                                   [0.227, 0.996],
                                   [-0.117188, 0.09832299999999999],
                                   [3e-005, 0.531903], [0.001122, 0.369589],
                                   [0.228, 0.99], [-0.12243, 0.098658],
                                   [2.8e-005, 0.530092],
                                   [0.001118, 0.370399]]).to(features)

    scaled_features = lower_bound + (upper_bound - lower_bound) * (
        features - feature_ranges[..., 0]) / (feature_ranges[..., 1] -
                                              feature_ranges[..., 0])

    return scaled_features


def rbf_kernel(features: torch.Tensor,
               sv: torch.Tensor,
               gamma: float = 0.05) -> torch.Tensor:
    dist = (features.unsqueeze(dim=-1) - sv.unsqueeze(dim=0)).pow(2).sum(dim=1)
    return torch.exp(-dist * gamma)


@ARCH_REGISTRY.register()
class BRISQUE(torch.nn.Module):
    r"""Creates a criterion that measures the BRISQUE score.

    Args:
        kernel_size: By default, the mean and covariance of a pixel is obtained
            by convolution with given filter_size. Must be an odd value.
        kernel_sigma: Standard deviation for Gaussian kernel.
        data_range: Maximum value range of images (usually 1.0 or 255).
        to_y_channel: Whether use the y-channel of YCBCR.
        pretrained_model_path: The model path.

    """

    def __init__(self,
                 kernel_size: int = 7,
                 kernel_sigma: float = 7 / 6,
                 data_range: Union[int, float] = 1.,
                 test_y_channel: Boolean = True,
                 pretrained_model_path: String = None) -> None:
        super().__init__()
        self.kernel_size = kernel_size

        # This check might look redundant because kernel size is checked within the brisque function anyway.
        # However, this check allows to fail fast when the loss is being initialised and training has not been started.
        assert kernel_size % 2 == 1, f'Kernel size must be odd, got [{kernel_size}]'

        self.kernel_sigma = kernel_sigma
        self.data_range = data_range
        self.test_y_channel = test_y_channel
        self.pretrained_model_path = pretrained_model_path

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Computation of BRISQUE score as a loss function.

        Args:
            x: An input tensor with (N, C, H, W) shape. RGB channel order for colour images.

        Returns:
            Value of BRISQUE metric.
        """
        return brisque(x,
                       kernel_size=self.kernel_size,
                       kernel_sigma=self.kernel_sigma,
                       data_range=self.data_range,
                       test_y_channel=self.test_y_channel,
                       pretrained_model_path=self.pretrained_model_path)