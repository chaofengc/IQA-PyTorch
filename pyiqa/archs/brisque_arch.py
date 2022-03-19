r"""BRISQUE Metric

Created by: https://github.com/photosynthesis-team/piq/blob/master/piq/brisque.py

Modified by: Jiadi Mo (https://github.com/JiadiMo)

Reference:
    MATLAB codes: https://live.ece.utexas.edu/research/Quality/index_algorithms.htm BRISQUE;
    Pretrained model from: https://github.com/photosynthesis-team/piq/releases/download/v0.4.0/brisque_svm_weights.pt

"""

import torch
from pyiqa.utils.color_util import to_y_channel
from pyiqa.matlab_utils import imresize
from .func_util import estimate_ggd_param, estimate_aggd_param, normalize_img_with_guass
from pyiqa.utils.download_util import load_file_from_url
from pyiqa.utils.registry import ARCH_REGISTRY

default_model_urls = {
    'url': 'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/brisque_svm_weights.pth'
}


def brisque(x: torch.Tensor,
            kernel_size: int = 7,
            kernel_sigma: float = 7 / 6,
            test_y_channel: bool = True,
            pretrained_model_path: str = None) -> torch.Tensor:
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
        x = to_y_channel(x, 255.)
    else:
        x = x * 255

    features = []
    num_of_scales = 2
    for _ in range(num_of_scales):
        features.append(natural_scene_statistics(x, kernel_size, kernel_sigma))
        x = imresize(x, scale=0.5, antialiasing=True)

    features = torch.cat(features, dim=-1)
    scaled_features = scale_features(features)

    if pretrained_model_path:
        sv_coef, sv = torch.load(pretrained_model_path)
        sv_coef = sv_coef.to(x)
        sv = sv.to(x)

    # gamma and rho are SVM model parameters taken from official implementation of BRISQUE on MATLAB
    # Source: https://live.ece.utexas.edu/research/Quality/index_algorithms.htm
    gamma = 0.05
    rho = -153.591
    sv.t_()
    kernel_features = rbf_kernel(features=scaled_features, sv=sv, gamma=gamma)
    score = kernel_features @ sv_coef
    return score - rho


def natural_scene_statistics(luma: torch.Tensor, kernel_size: int = 7, sigma: float = 7. / 6) -> torch.Tensor:

    luma_nrmlzd = normalize_img_with_guass(luma, kernel_size, sigma, padding='same')
    alpha, sigma = estimate_ggd_param(luma_nrmlzd)
    features = [alpha, sigma.pow(2)]

    shifts = [(0, 1), (1, 0), (1, 1), (-1, 1)]

    for shift in shifts:
        shifted_luma_nrmlzd = torch.roll(luma_nrmlzd, shifts=shift, dims=(-2, -1))
        alpha, sigma_l, sigma_r = estimate_aggd_param(luma_nrmlzd * shifted_luma_nrmlzd, return_sigma=True)
        eta = (sigma_r - sigma_l
               ) * torch.exp(torch.lgamma(2. / alpha) - (torch.lgamma(1. / alpha) + torch.lgamma(3. / alpha)) / 2)
        features.extend((alpha, eta, sigma_l.pow(2), sigma_r.pow(2)))

    return torch.stack(features, dim=-1)


def scale_features(features: torch.Tensor) -> torch.Tensor:
    lower_bound = -1
    upper_bound = 1
    # Feature range is taken from official implementation of BRISQUE on MATLAB.
    # Source: https://live.ece.utexas.edu/research/Quality/index_algorithms.htm
    feature_ranges = torch.tensor([[0.338, 10], [0.017204, 0.806612], [0.236, 1.642], [-0.123884, 0.20293],
                                   [0.000155, 0.712298], [0.001122, 0.470257], [0.244, 1.641], [-0.123586, 0.179083],
                                   [0.000152, 0.710456], [0.000975, 0.470984], [0.249, 1.555], [-0.135687, 0.100858],
                                   [0.000174, 0.684173], [0.000913, 0.534174], [0.258, 1.561], [-0.143408, 0.100486],
                                   [0.000179, 0.685696], [0.000888, 0.536508], [0.471, 3.264], [0.012809, 0.703171],
                                   [0.218, 1.046], [-0.094876, 0.187459], [1.5e-005, 0.442057], [0.001272, 0.40803],
                                   [0.222, 1.042], [-0.115772, 0.162604], [1.6e-005, 0.444362], [0.001374, 0.40243],
                                   [0.227, 0.996], [-0.117188, 0.09832299999999999], [3e-005, 0.531903],
                                   [0.001122, 0.369589], [0.228, 0.99], [-0.12243, 0.098658], [2.8e-005, 0.530092],
                                   [0.001118, 0.370399]]).to(features)

    scaled_features = lower_bound + (upper_bound - lower_bound) * (features - feature_ranges[..., 0]) / (
        feature_ranges[..., 1] - feature_ranges[..., 0])

    return scaled_features


def rbf_kernel(features: torch.Tensor, sv: torch.Tensor, gamma: float = 0.05) -> torch.Tensor:
    dist = (features.unsqueeze(dim=-1) - sv.unsqueeze(dim=0)).pow(2).sum(dim=1)
    return torch.exp(-dist * gamma)


@ARCH_REGISTRY.register()
class BRISQUE(torch.nn.Module):
    r"""Creates a criterion that measures the BRISQUE score.

    Args:
        kernel_size (int): By default, the mean and covariance of a pixel is obtained
            by convolution with given filter_size. Must be an odd value.
        kernel_sigma (float): Standard deviation for Gaussian kernel.
        to_y_channel (bool): Whether use the y-channel of YCBCR.
        pretrained_model_path (str): The model path.

    """

    def __init__(self,
                 kernel_size: int = 7,
                 kernel_sigma: float = 7 / 6,
                 test_y_channel: bool = True,
                 pretrained_model_path: str = None) -> None:
        super().__init__()
        self.kernel_size = kernel_size

        # This check might look redundant because kernel size is checked within the brisque function anyway.
        # However, this check allows to fail fast when the loss is being initialised and training has not been started.
        assert kernel_size % 2 == 1, f'Kernel size must be odd, got [{kernel_size}]'

        self.kernel_sigma = kernel_sigma
        self.test_y_channel = test_y_channel
        if pretrained_model_path is not None:
            self.pretrained_model_path = pretrained_model_path
        else:
            self.pretrained_model_path = load_file_from_url(default_model_urls['url'])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Computation of BRISQUE score as a loss function.

        Args:
            x: An input tensor with (N, C, H, W) shape. RGB channel order for colour images.

        Returns:
            Value of BRISQUE metric.

        """
        return brisque(
            x,
            kernel_size=self.kernel_size,
            kernel_sigma=self.kernel_sigma,
            test_y_channel=self.test_y_channel,
            pretrained_model_path=self.pretrained_model_path)
