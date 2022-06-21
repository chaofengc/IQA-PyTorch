r"""Color space conversion functions

Created by: https://github.com/photosynthesis-team/piq/blob/master/piq/functional/colour_conversion.py

Modified by: Chaofeng Chen (https://github.com/chaofengc)

"""
from typing import Union, Dict
import torch


def safe_frac_pow(x: torch.Tensor, p) -> torch.Tensor:
    EPS = torch.finfo(x.dtype).eps
    return torch.sign(x) * torch.abs(x + EPS).pow(p)


def to_y_channel(img: torch.Tensor, out_data_range: float = 1., color_space: str = 'yiq') -> torch.Tensor:
    r"""Change to Y channel
    Args:
        image tensor: tensor with shape (N, 3, H, W) in range [0, 1].
    Returns:
        image tensor: Y channel of the input tensor
    """
    assert img.ndim == 4 and img.shape[1] == 3, 'input image tensor should be RGB image batches with shape (N, 3, H, W)'
    color_space = color_space.lower()
    if color_space == 'yiq':
        img = rgb2yiq(img)
    elif color_space == 'ycbcr':
        img = rgb2ycbcr(img)
    elif color_space == 'lhm':
        img = rgb2lhm(img)
    out_img = img[:, [0], :, :] * out_data_range
    if out_data_range >= 255:
        # differentiable round with pytorch
        out_img = out_img - out_img.detach() + out_img.round()
    return out_img


def rgb2ycbcr(x: torch.Tensor) -> torch.Tensor:
    r"""Convert a batch of RGB images to a batch of YCbCr images

    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    Args:
        x: Batch of images with shape (N, 3, H, W). RGB color space, range [0, 1].

    Returns:
        Batch of images with shape (N, 3, H, W). YCbCr color space.
    """
    weights_rgb_to_ycbcr = torch.tensor([[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                                         [24.966, 112.0, -18.214]]).to(x)
    bias_rgb_to_ycbcr = torch.tensor([16, 128, 128]).view(1, 3, 1, 1).to(x)
    x_ycbcr = torch.matmul(x.permute(0, 2, 3, 1), weights_rgb_to_ycbcr).permute(0, 3, 1, 2) \
            + bias_rgb_to_ycbcr
    x_ycbcr = x_ycbcr / 255.
    return x_ycbcr


def ycbcr2rgb(x: torch.Tensor) -> torch.Tensor:
    r"""Convert a batch of YCbCr images to a batch of RGB images

    It implements the inversion of the above rgb2ycbcr function.

    Args:
        x: Batch of images with shape (N, 3, H, W). YCbCr color space, range [0, 1].

    Returns:
        Batch of images with shape (N, 3, H, W). RGB color space.
    """
    x = x * 255.
    weights_ycbcr_to_rgb = 255. * torch.tensor([[0.00456621, 0.00456621, 0.00456621], [0, -0.00153632, 0.00791071],
                                                [0.00625893, -0.00318811, 0]]).to(x)
    bias_ycbcr_to_rgb = torch.tensor([-222.921, 135.576, -276.836]).view(1, 3, 1, 1).to(x)
    x_rgb = torch.matmul(x.permute(0, 2, 3, 1), weights_ycbcr_to_rgb).permute(0, 3, 1, 2) \
            + bias_ycbcr_to_rgb
    x_rgb = x_rgb / 255.
    return x_rgb


def rgb2lmn(x: torch.Tensor) -> torch.Tensor:
    r"""Convert a batch of RGB images to a batch of LMN images

    Args:
        x: Batch of images with shape (N, 3, H, W). RGB colour space.

    Returns:
        Batch of images with shape (N, 3, H, W). LMN colour space.
    """
    weights_rgb_to_lmn = torch.tensor([[0.06, 0.63, 0.27], [0.30, 0.04, -0.35], [0.34, -0.6, 0.17]]).t().to(x)
    x_lmn = torch.matmul(x.permute(0, 2, 3, 1), weights_rgb_to_lmn).permute(0, 3, 1, 2)
    return x_lmn


def rgb2xyz(x: torch.Tensor) -> torch.Tensor:
    r"""Convert a batch of RGB images to a batch of XYZ images

    Args:
        x: Batch of images with shape (N, 3, H, W). RGB colour space.

    Returns:
        Batch of images with shape (N, 3, H, W). XYZ colour space.
    """
    mask_below = (x <= 0.04045).to(x)
    mask_above = (x > 0.04045).to(x)

    tmp = x / 12.92 * mask_below + torch.pow((x + 0.055) / 1.055, 2.4) * mask_above

    weights_rgb_to_xyz = torch.tensor([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750],
                                       [0.0193339, 0.1191920, 0.9503041]]).to(x)

    x_xyz = torch.matmul(tmp.permute(0, 2, 3, 1), weights_rgb_to_xyz.t()).permute(0, 3, 1, 2)
    return x_xyz


def xyz2lab(x: torch.Tensor, illuminant: str = 'D50', observer: str = '2') -> torch.Tensor:
    r"""Convert a batch of XYZ images to a batch of LAB images

    Args:
        x: Batch of images with shape (N, 3, H, W). XYZ colour space.
        illuminant: {“A”, “D50”, “D55”, “D65”, “D75”, “E”}, optional. The name of the illuminant.
        observer: {“2”, “10”}, optional. The aperture angle of the observer.

    Returns:
        Batch of images with shape (N, 3, H, W). LAB colour space.
    """
    epsilon = 0.008856
    kappa = 903.3
    illuminants: Dict[str, Dict] = \
        {'A': {'2': (1.098466069456375, 1, 0.3558228003436005),
               '10': (1.111420406956693, 1, 0.3519978321919493)},
         'D50': {'2': (0.9642119944211994, 1, 0.8251882845188288),
                 '10': (0.9672062750333777, 1, 0.8142801513128616)},
         'D55': {'2': (0.956797052643698, 1, 0.9214805860173273),
                 '10': (0.9579665682254781, 1, 0.9092525159847462)},
         'D65': {'2': (0.95047, 1., 1.08883),  # This was: `lab_ref_white`
                 '10': (0.94809667673716, 1, 1.0730513595166162)},
         'D75': {'2': (0.9497220898840717, 1, 1.226393520724154),
                 '10': (0.9441713925645873, 1, 1.2064272211720228)},
         'E': {'2': (1.0, 1.0, 1.0),
               '10': (1.0, 1.0, 1.0)}}

    illuminants_to_use = torch.tensor(illuminants[illuminant][observer]).to(x).view(1, 3, 1, 1)

    tmp = x / illuminants_to_use

    mask_below = tmp <= epsilon
    mask_above = tmp > epsilon
    tmp = safe_frac_pow(tmp, 1. / 3.) * mask_above + (kappa * tmp + 16.) / 116. * mask_below

    weights_xyz_to_lab = torch.tensor([[0, 116., 0], [500., -500., 0], [0, 200., -200.]]).to(x)
    bias_xyz_to_lab = torch.tensor([-16., 0., 0.]).to(x).view(1, 3, 1, 1)

    x_lab = torch.matmul(tmp.permute(0, 2, 3, 1), weights_xyz_to_lab.t()).permute(0, 3, 1, 2) + bias_xyz_to_lab
    return x_lab


def rgb2lab(x: torch.Tensor, data_range: Union[int, float] = 255) -> torch.Tensor:
    r"""Convert a batch of RGB images to a batch of LAB images

    Args:
        x: Batch of images with shape (N, 3, H, W). RGB colour space.
        data_range: dynamic range of the input image.

    Returns:
        Batch of images with shape (N, 3, H, W). LAB colour space.
    """
    return xyz2lab(rgb2xyz(x / float(data_range)))


def rgb2yiq(x: torch.Tensor) -> torch.Tensor:
    r"""Convert a batch of RGB images to a batch of YIQ images

    Args:
        x: Batch of images with shape (N, 3, H, W). RGB colour space.

    Returns:
        Batch of images with shape (N, 3, H, W). YIQ colour space.
    """
    yiq_weights = torch.tensor([[0.299, 0.587, 0.114], [0.5959, -0.2746, -0.3213], [0.2115, -0.5227, 0.3112]]).t().to(x)
    x_yiq = torch.matmul(x.permute(0, 2, 3, 1), yiq_weights).permute(0, 3, 1, 2)
    return x_yiq


def rgb2lhm(x: torch.Tensor) -> torch.Tensor:
    r"""Convert a batch of RGB images to a batch of LHM images

    Args:
        x: Batch of images with shape (N, 3, H, W). RGB colour space.

    Returns:
        Batch of images with shape (N, 3, H, W). LHM colour space.

    Reference:
        https://arxiv.org/pdf/1608.07433.pdf
    """
    lhm_weights = torch.tensor([[0.2989, 0.587, 0.114], [0.3, 0.04, -0.35], [0.34, -0.6, 0.17]]).t().to(x)
    x_lhm = torch.matmul(x.permute(0, 2, 3, 1), lhm_weights).permute(0, 3, 1, 2)
    return x_lhm
