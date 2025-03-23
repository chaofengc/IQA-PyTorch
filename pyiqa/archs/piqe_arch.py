r"""PIQE metric implementation.

Paper:
    N. Venkatanath, D. Praneeth, Bh. M. Chandrasekhar, S. S. Channappayya, and S. S. Medasani. "Blind Image Quality Evaluation Using Perception Based Features", In Proceedings of the 21st National Conference on Communications (NCC). Piscataway, NJ: IEEE, 2015.

References:
    - Matlab: https://www.mathworks.com/help/images/ref/piqe.html
    - Python: https://github.com/michael-rutherford/pypiqe

This PyTorch implementation by: Chaofeng Chen (https://github.com/chaofengc)
"""

import torch
import torch.nn.functional as F

from pyiqa.utils.color_util import to_y_channel
from pyiqa.matlab_utils import symm_pad
from pyiqa.archs.func_util import normalize_img_with_gauss
from pyiqa.utils.registry import ARCH_REGISTRY


def piqe(
    img: torch.Tensor,
    block_size: int = 16,
    activity_threshold: float = 0.1,
    block_impaired_threshold: float = 0.1,
    window_size: int = 6,
) -> torch.Tensor:
    """
    Calculates the Perceptual Image Quality Estimator (PIQE) score for an input image.
    Args:
        - img (torch.Tensor): The input image tensor.
        - block_size (int, optional): The size of the blocks used for processing. Defaults to 16.
        - activity_threshold (float, optional): The threshold for considering a block as active. Defaults to 0.1.
        - block_impaired_threshold (float, optional): The threshold for considering a block as impaired. Defaults to 0.1.
        - window_size (int, optional): The size of the window used for block analysis. Defaults to 6.
    Returns:
        - torch.Tensor: The PIQE score for the input image.
    """

    # RGB to Gray Conversion
    if img.shape[1] == 3:
        img = to_y_channel(img, out_data_range=1, color_space='yiq')

    # Convert input image to double and scaled to the range 0-255
    img = torch.round(
        255
        * (img / torch.max(img.flatten(1), dim=-1)[0].reshape(img.shape[0], 1, 1, 1))
    )

    # Symmetric pad if image size is not divisible by block_size.
    bsz, _, height, width = img.shape
    col_pad, row_pad = width % block_size, height % block_size
    img = symm_pad(img, (0, col_pad, 0, row_pad))
    new_height, new_width = img.shape[2], img.shape[3]

    # Normalize image to zero mean and ~unit std
    img_normalized = normalize_img_with_gauss(img, padding='replicate')

    # Create blocks
    blocks = img_normalized.unfold(2, block_size, block_size).unfold(
        3, block_size, block_size
    )
    blocks = blocks.contiguous().view(bsz, -1, block_size, block_size)

    # Compute block variance
    block_var = torch.var(blocks, dim=[2, 3], unbiased=True)

    # Considering spatially prominent blocks
    active_blocks = block_var > activity_threshold

    # Analyze blocks for noticeable artifacts and Gaussian noise distortions
    block_sigma, block_beta = noise_criterion(blocks, block_size - 1, block_var)
    noise_mask = block_sigma > 2 * block_beta

    block_impaired = notice_dist_criterion(
        blocks, window_size, block_impaired_threshold, block_size
    )

    # Pooling/ distortion assignment
    WHSA = active_blocks.float()
    WNDC = block_impaired.float()
    WNC = noise_mask.float()
    dist_block_scores = WHSA * WNDC * (1 - block_var) + WHSA * WNC * block_var

    # Quality score computation
    NHSA = active_blocks.sum(dim=1)
    dist_block_scores = dist_block_scores.sum(dim=1)
    C = 1
    score = ((dist_block_scores + C) / (C + NHSA)) * 100

    noticeable_artifacts_mask = block_impaired.view(
        bsz, 1, new_height // block_size, new_width // block_size
    )
    noticeable_artifacts_mask = F.interpolate(
        noticeable_artifacts_mask.float(), scale_factor=block_size, mode='nearest'
    )[..., :height, :width]

    noise_mask = noise_mask.view(
        bsz, 1, new_height // block_size, new_width // block_size
    )
    noise_mask = F.interpolate(
        noise_mask.float(), scale_factor=block_size, mode='nearest'
    )[..., :height, :width]

    activity_mask = active_blocks.view(
        bsz, 1, new_height // block_size, new_width // block_size
    )
    activity_mask = F.interpolate(
        activity_mask.float(), scale_factor=block_size, mode='nearest'
    )[..., :height, :width]

    return score, noticeable_artifacts_mask, noise_mask, activity_mask


def noise_criterion(block, block_size, block_var):
    """Function to analyze block for Gaussian noise distortions."""
    # Compute block standard deviation
    block_sigma = torch.sqrt(block_var)
    # Compute ratio of center and surround standard deviation
    cen_sur_dev = cal_center_sur_dev(block, block_size)
    # Relation between center-surround deviation and the block standard deviation
    block_beta = torch.abs(block_sigma - cen_sur_dev) / torch.max(
        block_sigma, cen_sur_dev
    )
    return block_sigma, block_beta


def cal_center_sur_dev(block, block_size):
    """Function to compute center surround Deviation of a block."""
    # block center
    center1 = (block_size + 1) // 2
    center2 = center1 + 1
    center = torch.stack((block[..., center1 - 1], block[..., center2 - 1]), dim=3)

    # block surround
    block = torch.cat((block[..., : center1 - 1], block[..., center1:]), dim=-1)
    block = torch.cat((block[..., : center2 - 1], block[..., center2:]), dim=-1)

    # Compute standard deviation of block center and block surround
    center_std = torch.std(center, dim=[2, 3], unbiased=True)
    surround_std = torch.std(block, dim=[2, 3], unbiased=True)
    # Ratio of center and surround standard deviation
    cen_sur_dev = center_std / surround_std
    # Check for nan's
    cen_sur_dev = torch.nan_to_num(cen_sur_dev)
    return cen_sur_dev


def notice_dist_criterion(blocks, window_size, block_impaired_threshold, N):
    """
    Analyze blocks for noticeable artifacts and Gaussian noise distortions.

    Args:
        blocks (torch.Tensor): Tensor of shape (b, num_blocks, block_size, block_size).
        window_size (int): Size of the window for segment analysis.
        block_impaired_threshold (float): Threshold for considering a block as impaired.
        N (int): Size of the blocks (same as block_size).

    Returns:
        torch.Tensor: Tensor indicating impaired blocks.
    """

    top_edge = blocks[:, :, 0, :]
    seg_top_edge = top_edge.unfold(-1, window_size, 1)

    right_side_edge = blocks[:, :, :, N - 1]
    seg_right_side_edge = right_side_edge.unfold(-1, window_size, 1)

    down_side_edge = blocks[:, :, N - 1, :]
    seg_down_side_edge = down_side_edge.unfold(-1, window_size, 1)

    left_side_edge = blocks[:, :, :, 0]
    seg_left_side_edge = left_side_edge.unfold(-1, window_size, 1)

    seg_top_edge_std_dev = torch.std(seg_top_edge, dim=-1, unbiased=True)
    seg_right_side_edge_std_dev = torch.std(seg_right_side_edge, dim=-1, unbiased=True)
    seg_down_side_edge_std_dev = torch.std(seg_down_side_edge, dim=-1, unbiased=True)
    seg_left_side_edge_std_dev = torch.std(seg_left_side_edge, dim=-1, unbiased=True)

    block_impaired = (
        (seg_top_edge_std_dev < block_impaired_threshold).sum(dim=2)
        + (seg_right_side_edge_std_dev < block_impaired_threshold).sum(dim=2)
        + (seg_down_side_edge_std_dev < block_impaired_threshold).sum(dim=2)
        + (seg_left_side_edge_std_dev < block_impaired_threshold).sum(dim=2)
    ) > 0

    return block_impaired


@ARCH_REGISTRY.register()
class PIQE(torch.nn.Module):
    """
    PIQE module.

    Args:
        x (torch.Tensor): Input tensor of shape (B, C, H, W).

    Returns:
        torch.Tensor: PIQE score.
    """

    def get_masks(
        self,
    ):
        assert self.results is not None, 'Please calculate the piqe score first.'
        return {
            'noticeable_artifacts_mask': self.results[1],
            'noise_mask': self.results[2],
            'activity_mask': self.results[3],
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.results = piqe(x)
        return self.results[0]
