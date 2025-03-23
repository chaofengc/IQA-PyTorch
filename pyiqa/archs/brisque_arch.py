r"""BRISQUE Metric

Created by: https://github.com/photosynthesis-team/piq/blob/master/piq/brisque.py

Modified by: Jiadi Mo (https://github.com/JiadiMo)

Reference:
    MATLAB codes: https://live.ece.utexas.edu/research/Quality/index_algorithms.htm BRISQUE;
    Pretrained model from: https://github.com/photosynthesis-team/piq/releases/download/v0.4.0/brisque_svm_weights.pt

"""

import scipy
import numpy as np
import torch
from pyiqa.utils.color_util import to_y_channel
from pyiqa.matlab_utils import imresize
from pyiqa.matlab_utils.nss_feature import compute_nss_features
from .func_util import estimate_ggd_param, estimate_aggd_param, normalize_img_with_gauss
from pyiqa.utils.download_util import load_file_from_url
from pyiqa.utils.registry import ARCH_REGISTRY
from pyiqa.archs.arch_util import get_url_from_name

default_model_urls = {
    'url': get_url_from_name('brisque_svm_weights.pth'),
    'brisque_matlab': get_url_from_name('brisque_matlab.mat'),
}


def brisque(
    x: torch.Tensor,
    kernel_size: int = 7,
    kernel_sigma: float = 7 / 6,
    test_y_channel: bool = True,
    sv_coef: torch.Tensor = None,
    sv: torch.Tensor = None,
    gamma: float = 0.05,
    rho: float = -153.591,
    scale: float = 1,
    version: str = 'original',
) -> torch.Tensor:
    r"""Interface of BRISQUE index.

    Args:
        x (torch.Tensor): An input tensor. Shape :math:`(N, C, H, W)`.
        kernel_size (int): The side-length of the sliding window used in comparison. Must be an odd value.
        kernel_sigma (float): Sigma of normal distribution.
        test_y_channel (bool): Whether to use the y-channel of YCBCR.
        sv_coef (torch.Tensor): Support vector coefficients.
        sv (torch.Tensor): Support vectors.
        gamma (float): Gamma parameter for the RBF kernel.
        rho (float): Bias term in the decision function.
        scale (float): Scaling factor for the features.
        version (str): Version of the BRISQUE implementation ('original' or 'matlab').

    Returns:
        torch.Tensor: Value of BRISQUE index.

    References:
        Mittal, Anish, Anush Krishna Moorthy, and Alan Conrad Bovik.
        "No-reference image quality assessment in the spatial domain."
        IEEE Transactions on image processing 21, no. 12 (2012): 4695-4708.
    """
    if test_y_channel and x.size(1) == 3:
        x = to_y_channel(x, 255.0)
    else:
        x = x * 255

    features = []
    num_of_scales = 2
    for _ in range(num_of_scales):
        if version == 'matlab':
            xnorm = normalize_img_with_gauss(
                x, kernel_size, kernel_sigma, padding='replicate'
            )
            features.append(compute_nss_features(xnorm))
        elif version == 'original':
            features.append(natural_scene_statistics(x, kernel_size, kernel_sigma))
        x = imresize(x, scale=0.5, antialiasing=True)
    features = torch.cat(features, dim=-1)

    sv_coef = sv_coef.to(x)
    sv = sv.to(x)

    if version == 'original':
        scaled_features = scale_features(features)
    elif version == 'matlab':
        scaled_features = features / scale

    sv = sv.t()
    kernel_features = rbf_kernel(features=scaled_features, sv=sv, gamma=gamma)
    score = kernel_features @ sv_coef - rho
    return score


def natural_scene_statistics(
    luma: torch.Tensor, kernel_size: int = 7, sigma: float = 7.0 / 6
) -> torch.Tensor:
    """
    Compute natural scene statistics (NSS) features for a given luminance image.

    Args:
        luma (torch.Tensor): Luminance image tensor.
        kernel_size (int): Size of the Gaussian kernel.
        sigma (float): Standard deviation of the Gaussian kernel.

    Returns:
        torch.Tensor: NSS features.
    """
    luma_nrmlzd = normalize_img_with_gauss(luma, kernel_size, sigma, padding='same')
    alpha, sigma = estimate_ggd_param(luma_nrmlzd)
    features = [alpha, sigma.pow(2)]

    shifts = [(0, 1), (1, 0), (1, 1), (-1, 1)]

    for shift in shifts:
        shifted_luma_nrmlzd = torch.roll(luma_nrmlzd, shifts=shift, dims=(-2, -1))
        alpha, sigma_l, sigma_r = estimate_aggd_param(
            luma_nrmlzd * shifted_luma_nrmlzd, return_sigma=True
        )
        eta = (sigma_r - sigma_l) * torch.exp(
            torch.lgamma(2.0 / alpha)
            - (torch.lgamma(1.0 / alpha) + torch.lgamma(3.0 / alpha)) / 2
        )
        features.extend((alpha, eta, sigma_l.pow(2), sigma_r.pow(2)))

    return torch.stack(features, dim=-1)


def scale_features(features: torch.Tensor) -> torch.Tensor:
    """
    Scale features to the range [-1, 1] based on predefined feature ranges.

    Args:
        features (torch.Tensor): Input features.

    Returns:
        torch.Tensor: Scaled features.
    """
    lower_bound = -1
    upper_bound = 1
    # Feature range is taken from official implementation of BRISQUE on MATLAB.
    # Source: https://live.ece.utexas.edu/research/Quality/index_algorithms.htm
    feature_ranges = torch.tensor(
        [
            [0.338, 10],
            [0.017204, 0.806612],
            [0.236, 1.642],
            [-0.123884, 0.20293],
            [0.000155, 0.712298],
            [0.001122, 0.470257],
            [0.244, 1.641],
            [-0.123586, 0.179083],
            [0.000152, 0.710456],
            [0.000975, 0.470984],
            [0.249, 1.555],
            [-0.135687, 0.100858],
            [0.000174, 0.684173],
            [0.000913, 0.534174],
            [0.258, 1.561],
            [-0.143408, 0.100486],
            [0.000179, 0.685696],
            [0.000888, 0.536508],
            [0.471, 3.264],
            [0.012809, 0.703171],
            [0.218, 1.046],
            [-0.094876, 0.187459],
            [1.5e-005, 0.442057],
            [0.001272, 0.40803],
            [0.222, 1.042],
            [-0.115772, 0.162604],
            [1.6e-005, 0.444362],
            [0.001374, 0.40243],
            [0.227, 0.996],
            [-0.117188, 0.09832299999999999],
            [3e-005, 0.531903],
            [0.001122, 0.369589],
            [0.228, 0.99],
            [-0.12243, 0.098658],
            [2.8e-005, 0.530092],
            [0.001118, 0.370399],
        ]
    ).to(features)

    scaled_features = lower_bound + (upper_bound - lower_bound) * (
        features - feature_ranges[..., 0]
    ) / (feature_ranges[..., 1] - feature_ranges[..., 0])

    return scaled_features


def rbf_kernel(
    features: torch.Tensor, sv: torch.Tensor, gamma: float = 0.05
) -> torch.Tensor:
    """
    Compute the Radial Basis Function (RBF) kernel between features and support vectors.

    Args:
        features (torch.Tensor): Input features.
        sv (torch.Tensor): Support vectors.
        gamma (float): Gamma parameter for the RBF kernel.

    Returns:
        torch.Tensor: RBF kernel values.
    """
    dist = (features.unsqueeze(dim=-1) - sv.unsqueeze(dim=0)).pow(2).sum(dim=1)
    return torch.exp(-dist * gamma)


@ARCH_REGISTRY.register()
class BRISQUE(torch.nn.Module):
    r"""Creates a criterion that measures the BRISQUE score.

    Args:
        kernel_size (int): By default, the mean and covariance of a pixel is obtained
                           by convolution with given filter_size. Must be an odd value.
        kernel_sigma (float): Standard deviation for Gaussian kernel.
        test_y_channel (bool): Whether to use the y-channel of YCBCR.
        version (str): Version of the BRISQUE implementation ('original' or 'matlab').
        pretrained_model_path (str, optional): The model path.

    Attributes:
        kernel_size (int): The side-length of the sliding window used in comparison.
        kernel_sigma (float): Sigma of normal distribution.
        test_y_channel (bool): Whether to use the y-channel of YCBCR.
        sv_coef (torch.Tensor): Support vector coefficients.
        sv (torch.Tensor): Support vectors.
        gamma (float): Gamma parameter for the RBF kernel.
        rho (float): Bias term in the decision function.
        scale (float): Scaling factor for the features.
        version (str): Version of the BRISQUE implementation ('original' or 'matlab').
    """

    def __init__(
        self,
        kernel_size: int = 7,
        kernel_sigma: float = 7 / 6,
        test_y_channel: bool = True,
        version: str = 'original',
        pretrained_model_path: str = None,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size

        # This check might look redundant because kernel size is checked within the brisque function anyway.
        # However, this check allows to fail fast when the loss is being initialised and training has not been started.
        assert kernel_size % 2 == 1, f'Kernel size must be odd, got [{kernel_size}]'
        assert test_y_channel, (
            f'Only [test_y_channel=True] is supported for current BRISQUE model, which is taken directly from official codes: https://github.com/utlive/BRISQUE.'
        )

        self.kernel_sigma = kernel_sigma
        self.test_y_channel = test_y_channel

        if pretrained_model_path is not None:
            self.sv_coef, self.sv = torch.load(
                pretrained_model_path, weights_only=False
            )
        elif version == 'original':
            # gamma and rho are SVM model parameters taken from official implementation of BRISQUE on MATLAB
            # Source: https://live.ece.utexas.edu/research/Quality/index_algorithms.htm
            pretrained_model_path = load_file_from_url(default_model_urls['url'])
            self.sv_coef, self.sv = torch.load(
                pretrained_model_path, weights_only=False
            )
            self.gamma = 0.05
            self.rho = -153.591
            self.scale = 1
        elif version == 'matlab':
            pretrained_model_path = load_file_from_url(
                default_model_urls['brisque_matlab']
            )
            self.gamma = 1
            self.rho = -43.4582
            self.scale = 0.3210

            params = scipy.io.loadmat(pretrained_model_path)
            sv = params['sv']
            sv_coef = np.ravel(params['sv_coef'])
            sv = torch.from_numpy(sv)
            self.sv_coef = torch.from_numpy(sv_coef)
            self.sv = sv / self.scale

        self.version = version

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Computation of BRISQUE score as a loss function.

        Args:
            x (torch.Tensor): An input tensor with (N, C, H, W) shape. RGB channel order for colour images.

        Returns:
            torch.Tensor: Value of BRISQUE metric.
        """
        return brisque(
            x,
            kernel_size=self.kernel_size,
            kernel_sigma=self.kernel_sigma,
            test_y_channel=self.test_y_channel,
            sv_coef=self.sv_coef,
            sv=self.sv,
            gamma=self.gamma,
            rho=self.rho,
            scale=self.scale,
            version=self.version,
        )
