r"""FSIM Metric

Created by: https://github.com/photosynthesis-team/piq/blob/master/piq/fsim.py

Modified by: Jiadi Mo (https://github.com/JiadiMo)

Refer to:
    Official matlab code from https://www4.comp.polyu.edu.hk/~cslzhang/IQA/FSIM/Files/FeatureSIM.m
    PIQA from https://github.com/francois-rozet/piqa/blob/master/piqa/fsim.py

"""

import math
import functools
from typing import Tuple
import torch.nn as nn
import torch

from pyiqa.utils.registry import ARCH_REGISTRY
from pyiqa.utils.color_util import rgb2yiq
from .func_util import gradient_map, similarity_map, ifftshift, get_meshgrid


def fsim(x: torch.Tensor,
         y: torch.Tensor,
         chromatic: bool = True,
         scales: int = 4,
         orientations: int = 4,
         min_length: int = 6,
         mult: int = 2,
         sigma_f: float = 0.55,
         delta_theta: float = 1.2,
         k: float = 2.0) -> torch.Tensor:
    r"""Compute Feature Similarity Index Measure for a batch of images.
    Args:
        x: An input tensor. Shape :math:`(N, C, H, W)`.
        y: A target tensor. Shape :math:`(N, C, H, W)`.
        chromatic: Flag to compute FSIMc, which also takes into account chromatic components
        scales: Number of wavelets used for computation of phase congruensy maps
        orientations: Number of filter orientations used for computation of phase congruensy maps
        min_length: Wavelength of smallest scale filter
        mult: Scaling factor between successive filters
        sigma_f: Ratio of the standard deviation of the Gaussian describing the log Gabor filter's
            transfer function in the frequency domain to the filter center frequency.
        delta_theta: Ratio of angular interval between filter orientations and the standard deviation
            of the angular Gaussian function used to construct filters in the frequency plane.
        k: No of standard deviations of the noise energy beyond the mean at which we set the noise
            threshold  point, below which phase congruency values get penalized.

    Returns:
        Index of similarity betwen two images. Usually in [0, 1] interval.
        Can be bigger than 1 for predicted :math:`x` images with higher contrast than the original ones.
    References:
        L. Zhang, L. Zhang, X. Mou and D. Zhang, "FSIM: A Feature Similarity Index for Image Quality Assessment,"
        IEEE Transactions on Image Processing, vol. 20, no. 8, pp. 2378-2386, Aug. 2011, doi: 10.1109/TIP.2011.2109730.
        https://ieeexplore.ieee.org/document/5705575

    """

    # Rescale to [0, 255] range, because all constant are calculated for this factor
    x = x / float(1.0) * 255
    y = y / float(1.0) * 255

    # Apply average pooling
    kernel_size = max(1, round(min(x.shape[-2:]) / 256))
    x = torch.nn.functional.avg_pool2d(x, kernel_size)
    y = torch.nn.functional.avg_pool2d(y, kernel_size)

    num_channels = x.size(1)

    # Convert RGB to YIQ color space
    if num_channels == 3:
        x_yiq = rgb2yiq(x)
        y_yiq = rgb2yiq(y)

        x_lum = x_yiq[:, :1]
        y_lum = y_yiq[:, :1]

        x_i = x_yiq[:, 1:2]
        y_i = y_yiq[:, 1:2]
        x_q = x_yiq[:, 2:]
        y_q = y_yiq[:, 2:]

    else:
        x_lum = x
        y_lum = y

    # Compute phase congruency maps
    pc_x = _phase_congruency(
        x_lum,
        scales=scales,
        orientations=orientations,
        min_length=min_length,
        mult=mult,
        sigma_f=sigma_f,
        delta_theta=delta_theta,
        k=k)
    pc_y = _phase_congruency(
        y_lum,
        scales=scales,
        orientations=orientations,
        min_length=min_length,
        mult=mult,
        sigma_f=sigma_f,
        delta_theta=delta_theta,
        k=k)

    # Gradient maps
    scharr_filter = torch.tensor([[[-3., 0., 3.], [-10., 0., 10.], [-3., 0., 3.]]]) / 16

    kernels = torch.stack([scharr_filter, scharr_filter.transpose(-1, -2)])
    grad_map_x = gradient_map(x_lum, kernels)
    grad_map_y = gradient_map(y_lum, kernels)

    # Constants from the paper
    T1, T2, T3, T4, lmbda = 0.85, 160, 200, 200, 0.03

    # Compute FSIM
    PC = similarity_map(pc_x, pc_y, T1)
    GM = similarity_map(grad_map_x, grad_map_y, T2)
    pc_max = torch.where(pc_x > pc_y, pc_x, pc_y)
    score = GM * PC * pc_max  # torch.sum(score)/torch.sum(pc_max)

    if chromatic:
        assert num_channels == 3, 'Chromatic component can be computed only for RGB images!'
        S_I = similarity_map(x_i, y_i, T3)
        S_Q = similarity_map(x_q, y_q, T4)
        score = score * torch.abs(S_I * S_Q)**lmbda
        # Complex gradients will work in PyTorch 1.6.0
        # score = score * torch.real((S_I * S_Q).to(torch.complex64) ** lmbda)

    result = score.sum(dim=[1, 2, 3]) / pc_max.sum(dim=[1, 2, 3])

    return result


def _construct_filters(x: torch.Tensor,
                       scales: int = 4,
                       orientations: int = 4,
                       min_length: int = 6,
                       mult: int = 2,
                       sigma_f: float = 0.55,
                       delta_theta: float = 1.2,
                       k: float = 2.0,
                       use_lowpass_filter=True):
    """Creates a stack of filters used for computation of phase congruensy maps

    Args:
        x: Tensor. Shape :math:`(N, 1, H, W)`.
        scales: Number of wavelets
        orientations: Number of filter orientations
        min_length: Wavelength of smallest scale filter
        mult: Scaling factor between successive filters
        sigma_f: Ratio of the standard deviation of the Gaussian
            describing the log Gabor filter's transfer function
            in the frequency domain to the filter center frequency.
        delta_theta: Ratio of angular interval between filter orientations
            and the standard deviation of the angular Gaussian function
            used to construct filters in the freq. plane.
        k: No of standard deviations of the noise energy beyond the mean
            at which we set the noise threshold point, below which phase
            congruency values get penalized.
        """
    N, _, H, W = x.shape

    # Calculate the standard deviation of the angular Gaussian function
    # used to construct filters in the freq. plane.
    theta_sigma = math.pi / (orientations * delta_theta)

    # Pre-compute some stuff to speed up filter construction
    grid_x, grid_y = get_meshgrid((H, W))
    radius = torch.sqrt(grid_x**2 + grid_y**2)
    theta = torch.atan2(-grid_y, grid_x)

    # Quadrant shift radius and theta so that filters are constructed with 0 frequency at the corners.
    # Get rid of the 0 radius value at the 0 frequency point (now at top-left corner)
    # so that taking the log of the radius will not cause trouble.
    radius = ifftshift(radius)
    theta = ifftshift(theta)
    radius[0, 0] = 1

    sintheta = torch.sin(theta)
    costheta = torch.cos(theta)

    # Filters are constructed in terms of two components.
    # 1) The radial component, which controls the frequency band that the filter responds to
    # 2) The angular component, which controls the orientation that the filter responds to.
    # The two components are multiplied together to construct the overall filter.

    # First construct a low-pass filter that is as large as possible, yet falls
    # away to zero at the boundaries.  All log Gabor filters are multiplied by
    # this to ensure no extra frequencies at the 'corners' of the FFT are
    # incorporated as this seems to upset the normalisation process when
    lp = _lowpassfilter(size=(H, W), cutoff=.45, n=15)

    # Construct the radial filter components...
    log_gabor = []
    for s in range(scales):
        wavelength = min_length * mult**s
        omega_0 = 1.0 / wavelength
        gabor_filter = torch.exp((-torch.log(radius / omega_0)**2) / (2 * math.log(sigma_f)**2))
        if use_lowpass_filter:
            gabor_filter = gabor_filter * lp
        gabor_filter[0, 0] = 0
        log_gabor.append(gabor_filter)

    # Then construct the angular filter components...
    spread = []
    for o in range(orientations):
        angl = o * math.pi / orientations

        # For each point in the filter matrix calculate the angular distance from
        # the specified filter orientation.  To overcome the angular wrap-around
        # problem sine difference and cosine difference values are first computed
        # and then the atan2 function is used to determine angular distance.
        ds = sintheta * math.cos(angl) - costheta * math.sin(angl)  # Difference in sine.
        dc = costheta * math.cos(angl) + sintheta * math.sin(angl)  # Difference in cosine.
        dtheta = torch.abs(torch.atan2(ds, dc))
        spread.append(torch.exp((-dtheta**2) / (2 * theta_sigma**2)))

    spread = torch.stack(spread)
    log_gabor = torch.stack(log_gabor)

    # Multiply, add batch dimension and transfer to correct device.
    filters = (spread.repeat_interleave(scales, dim=0) * log_gabor.repeat(orientations, 1, 1)).unsqueeze(0).to(x)
    return filters


def _phase_congruency(x: torch.Tensor,
                      scales: int = 4,
                      orientations: int = 4,
                      min_length: int = 6,
                      mult: int = 2,
                      sigma_f: float = 0.55,
                      delta_theta: float = 1.2,
                      k: float = 2.0) -> torch.Tensor:
    r"""Compute Phase Congruence for a batch of greyscale images
    Args:
        x: Tensor. Shape :math:`(N, 1, H, W)`.
        scales: Number of wavelet scales
        orientations: Number of filter orientations
        min_length: Wavelength of smallest scale filter
        mult: Scaling factor between successive filters
        sigma_f: Ratio of the standard deviation of the Gaussian
            describing the log Gabor filter's transfer function
            in the frequency domain to the filter center frequency.
        delta_theta: Ratio of angular interval between filter orientations
            and the standard deviation of the angular Gaussian function
            used to construct filters in the freq. plane.
        k: No of standard deviations of the noise energy beyond the mean
            at which we set the noise threshold point, below which phase
            congruency values get penalized.
    Returns:
        Phase Congruency map with shape :math:`(N, H, W)`
    """
    EPS = torch.finfo(x.dtype).eps

    N, _, H, W = x.shape

    # Fourier transform
    filters = _construct_filters(x, scales, orientations, min_length, mult, sigma_f, delta_theta, k)

    imagefft = torch.fft.fft2(x)
    filters_ifft = torch.fft.ifft2(filters)
    filters_ifft = filters_ifft.real * math.sqrt(H * W)
    even_odd = torch.view_as_real(torch.fft.ifft2(imagefft * filters)).view(N, orientations, scales, H, W, 2)

    # Amplitude of even & odd filter response. An = sqrt(real^2 + imag^2)
    an = torch.sqrt(torch.sum(even_odd**2, dim=-1))

    # Take filter at scale 0 and sum spatially
    # Record mean squared filter value at smallest scale.
    # This is used for noise estimation.
    em_n = (filters.view(1, orientations, scales, H, W)[:, :, :1, ...]**2).sum(dim=[-2, -1], keepdims=True)

    # Sum of even filter convolution results.
    sum_e = even_odd[..., 0].sum(dim=2, keepdims=True)

    # Sum of odd filter convolution results.
    sum_o = even_odd[..., 1].sum(dim=2, keepdims=True)

    # Get weighted mean filter response vector, this gives the weighted mean phase angle.
    x_energy = torch.sqrt(sum_e**2 + sum_o**2) + EPS

    mean_e = sum_e / x_energy
    mean_o = sum_o / x_energy

    # Now calculate An(cos(phase_deviation) - | sin(phase_deviation)) | by
    # using dot and cross products between the weighted mean filter response
    # vector and the individual filter response vectors at each scale.
    # This quantity is phase congruency multiplied by An, which we call energy.

    # Extract even and odd convolution results.
    even = even_odd[..., 0]
    odd = even_odd[..., 1]

    energy = (even * mean_e + odd * mean_o - torch.abs(even * mean_o - odd * mean_e)).sum(dim=2, keepdim=True)

    # Compensate for noise
    # We estimate the noise power from the energy squared response at the
    # smallest scale.  If the noise is Gaussian the energy squared will have a
    # Chi-squared 2DOF pdf.  We calculate the median energy squared response
    # as this is a robust statistic.  From this we estimate the mean.
    # The estimate of noise power is obtained by dividing the mean squared
    # energy value by the mean squared filter value

    abs_eo = torch.sqrt(torch.sum(even_odd[:, :, :1, ...]**2, dim=-1)).reshape(N, orientations, 1, 1, H * W)
    median_e2n = torch.median(abs_eo**2, dim=-1, keepdim=True).values

    mean_e2n = -median_e2n / math.log(0.5)

    # Estimate of noise power.
    noise_power = mean_e2n / em_n

    # Now estimate the total energy^2 due to noise
    # Estimate for sum(An^2) + sum(Ai.*Aj.*(cphi.*cphj + sphi.*sphj))
    filters_ifft = filters_ifft.view(1, orientations, scales, H, W)

    sum_an2 = torch.sum(filters_ifft**2, dim=-3, keepdim=True)

    sum_ai_aj = torch.zeros(N, orientations, 1, H, W).to(x)
    for s in range(scales - 1):
        sum_ai_aj = sum_ai_aj + (filters_ifft[:, :, s:s + 1] * filters_ifft[:, :, s + 1:]).sum(dim=-3, keepdim=True)

    sum_an2 = torch.sum(sum_an2, dim=[-1, -2], keepdim=True)
    sum_ai_aj = torch.sum(sum_ai_aj, dim=[-1, -2], keepdim=True)

    noise_energy2 = 2 * noise_power * sum_an2 + 4 * noise_power * sum_ai_aj

    # Rayleigh parameter
    tau = torch.sqrt(noise_energy2 / 2)

    # Expected value of noise energy
    noise_energy = tau * math.sqrt(math.pi / 2)
    moise_energy_sigma = torch.sqrt((2 - math.pi / 2) * tau**2)

    # Noise threshold
    T = noise_energy + k * moise_energy_sigma

    # The estimated noise effect calculated above is only valid for the PC_1 measure.
    # The PC_2 measure does not lend itself readily to the same analysis.  However
    # empirically it seems that the noise effect is overestimated roughly by a factor
    # of 1.7 for the filter parameters used here.

    # Empirical rescaling of the estimated noise effect to suit the PC_2 phase congruency measure
    T = T / 1.7

    # Apply noise threshold
    energy = torch.max(energy - T, torch.zeros_like(T))

    eps = torch.finfo(energy.dtype).eps
    energy_all = energy.sum(dim=[1, 2]) + eps
    an_all = an.sum(dim=[1, 2]) + eps
    result_pc = energy_all / an_all
    return result_pc.unsqueeze(1)


def _lowpassfilter(size: Tuple[int, int], cutoff: float, n: int) -> torch.Tensor:
    r"""
    Constructs a low-pass Butterworth filter.
    Args:
        size: Tuple with heigth and width of filter to construct
        cutoff: Cutoff frequency of the filter in (0, 0.5()
        n: Filter order. Higher `n` means sharper transition.
            Note that `n` is doubled so that it is always an even integer.

    Returns:
        f = 1 / (1 + w/cutoff) ^ 2n

    """
    assert 0 < cutoff <= 0.5, 'Cutoff frequency must be between 0 and 0.5'
    assert n > 1 and int(n) == n, 'n must be an integer >= 1'

    grid_x, grid_y = get_meshgrid(size)

    # A matrix with every pixel = radius relative to centre.
    radius = torch.sqrt(grid_x**2 + grid_y**2)

    return ifftshift(1. / (1.0 + (radius / cutoff)**(2 * n)))


@ARCH_REGISTRY.register()
class FSIM(nn.Module):
    r"""Args:
        chromatic: Flag to compute FSIMc, which also takes into account chromatic components
        scales: Number of wavelets used for computation of phase congruensy maps
        orientations: Number of filter orientations used for computation of phase congruensy maps
        min_length: Wavelength of smallest scale filter
        mult: Scaling factor between successive filters
        sigma_f: Ratio of the standard deviation of the Gaussian describing the log Gabor filter's
            transfer function in the frequency domain to the filter center frequency.
        delta_theta: Ratio of angular interval between filter orientations and the standard deviation
            of the angular Gaussian function used to construct filters in the frequency plane.
        k: No of standard deviations of the noise energy beyond the mean at which we set the noise
            threshold  point, below which phase congruency values get penalized.
    References:
        L. Zhang, L. Zhang, X. Mou and D. Zhang, "FSIM: A Feature Similarity Index for Image Quality Assessment,"
        IEEE Transactions on Image Processing, vol. 20, no. 8, pp. 2378-2386, Aug. 2011, doi: 10.1109/TIP.2011.2109730.
        https://ieeexplore.ieee.org/document/5705575
    """

    def __init__(self,
                 chromatic: bool = True,
                 scales: int = 4,
                 orientations: int = 4,
                 min_length: int = 6,
                 mult: int = 2,
                 sigma_f: float = 0.55,
                 delta_theta: float = 1.2,
                 k: float = 2.0) -> None:
        super().__init__()

        # Save function with predefined parameters, rather than parameters themself
        self.fsim = functools.partial(
            fsim,
            chromatic=chromatic,
            scales=scales,
            orientations=orientations,
            min_length=min_length,
            mult=mult,
            sigma_f=sigma_f,
            delta_theta=delta_theta,
            k=k,
        )

    def forward(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
    ) -> torch.Tensor:
        r"""Computation of FSIM as a loss function.
        Args:
            x: An input tensor. Shape :math:`(N, C, H, W)`.
            y: A target tensor. Shape :math:`(N, C, H, W)`.
        Returns:
            Value of FSIM loss to be minimized in [0, 1] range.
        """

        assert X.shape == Y.shape, f'Input and reference images should have the same shape, but got {X.shape} and {Y.shape}'
        score = self.fsim(X, Y)
        return score
