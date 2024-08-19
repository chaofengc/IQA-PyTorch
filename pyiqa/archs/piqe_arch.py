r"""PIQE metric implementation.

Paper: 
    N. Venkatanath, D. Praneeth, Bh. M. Chandrasekhar, S. S. Channappayya, and S. S. Medasani. "Blind Image Quality Evaluation Using Perception Based Features", In Proceedings of the 21st National Conference on Communications (NCC). Piscataway, NJ: IEEE, 2015.

References:
    - Matlab: https://www.mathworks.com/help/images/ref/piqe.html
    - Python: https://github.com/michael-rutherford/pypiqe

This PyTorch implementation by: Chaofeng Chen (https://github.com/chaofengc)
"""

import torch

from pyiqa.utils.color_util import to_y_channel
from pyiqa.utils import scandir_images, imread2tensor
from pyiqa.matlab_utils import symm_pad
from pyiqa.archs.func_util import normalize_img_with_guass
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
    img = torch.round(255 * (img / torch.max(img.flatten(1), dim=-1)[0].reshape(img.shape[0], 1, 1, 1)))

    # Symmetric pad if image size is not divisible by block_size.
    bsz, _, height, width = img.shape
    col_pad, row_pad = width % block_size, height % block_size
    img = symm_pad(img, (0, col_pad, 0, row_pad))

    # Normalize image to zero mean and ~unit std
    # used circularly-symmetric Gaussian weighting function sampled out 
    # to 3 standard deviations.
    img_normalized = normalize_img_with_guass(img, padding='replicate')

    # Preallocation for masks
    noticeable_artifacts_mask = torch.zeros_like(img_normalized, dtype=bool)
    noise_mask = torch.zeros_like(img_normalized, dtype=bool)
    activity_mask = torch.zeros_like(img_normalized, dtype=bool)
    score = torch.zeros(bsz)

    nsegments = block_size - window_size + 1
    # Start of block by block processing
    for b in range(0, bsz):
        NHSA = 0
        dist_block_scores = 0
        for i in range(0, height, block_size):
            for j in range(0, width, block_size):

                # Weights Initialization
                WNDC = WNC = 0

                # Compute block variance
                block = img_normalized[b, 0, i:i + block_size, j:j + block_size]
                block_var = torch.var(block, unbiased=True)

                # Considering spatially prominent blocks 
                if block_var > activity_threshold:
                    activity_mask[b, 0, i:i + block_size, j:j + block_size] = True
                    WHSA = 1
                    NHSA += 1

                    # Analyze Block for noticeable artifacts
                    block_impaired = notice_dist_criterion(block, nsegments, block_size - 1, window_size, block_impaired_threshold, block_size)

                    if block_impaired:
                        WNDC = 1
                        noticeable_artifacts_mask[b, 0, i:i + block_size, j:j + block_size] = True

                    # Analyze Block for Gaussian noise distortions
                    block_sigma, block_beta = noise_criterion(block, block_size - 1, block_var)

                    if block_sigma > 2 * block_beta:
                        WNC = 1
                        noise_mask[b, 0, i:i + block_size, j:j + block_size] = True

                    # Pooling/ distortion assignment
                    dist_block_scores += WHSA * WNDC * (1 - block_var) + WHSA * WNC * block_var

        # Quality score computation
        # C is a positive constant, it is included to prevent numerical instability
        C = 1
        score[b] = ((dist_block_scores + C) / (C + NHSA)) * 100

    noticeable_artifacts_mask = noticeable_artifacts_mask[..., :height, :width]
    noise_mask = noise_mask[..., :height, :width]
    activity_mask = activity_mask[..., :height, :width]

    return score, noticeable_artifacts_mask, noise_mask, activity_mask 


def noise_criterion(block, block_size, block_var):
    """Function to analyze block for Gaussian noise distortions.
    """
    # Compute block standard deviation
    block_sigma = torch.sqrt(block_var)
    # Compute ratio of center and surround standard deviation
    cen_sur_dev = cal_center_sur_dev(block, block_size)
    # Relation between center-surround deviation and the block standard deviation
    block_beta = torch.abs(block_sigma - cen_sur_dev) / torch.max(block_sigma, cen_sur_dev)
    return block_sigma, block_beta


def cal_center_sur_dev(block, block_size):
    """Function to compute center surround Deviation of a block.
    """
    # block center
    center1 = (block_size + 1) // 2
    center2 = center1 + 1
    center = torch.cat((block[..., center1 - 1], block[..., center2 - 1]), dim=0)

    # block surround
    block = torch.cat((block[..., :center1 - 1], block[..., center1:]), dim=-1)
    block = torch.cat((block[..., :center2 - 1], block[..., center2:]), dim=-1)

    # Compute standard deviation of block center and block surround
    center_std = torch.std(center, unbiased=True)
    surround_std = torch.std(block, unbiased=True)
    # Ratio of center and surround standard deviation
    cen_sur_dev = center_std / surround_std
    # Check for nan's
    if torch.isnan(cen_sur_dev):
        cen_sur_dev = 0
    return cen_sur_dev


def notice_dist_criterion(block, nsegments, block_size, window_size, block_impaired_threshold, N):
    # Top edge of block
    top_edge = block[0, :]
    seg_top_edge = segment_edge(top_edge, nsegments, block_size, window_size)

    # Right side edge of block
    right_side_edge = block[:, N - 1]
    seg_right_side_edge = segment_edge(right_side_edge, nsegments, block_size, window_size)

    # Down side edge of block
    down_side_edge = block[N - 1, :]
    seg_down_side_edge = segment_edge(down_side_edge, nsegments, block_size, window_size)

    # Left side edge of block
    left_side_edge = block[:, 0]
    seg_left_side_edge = segment_edge(left_side_edge, nsegments, block_size, window_size)

    # Compute standard deviation of segments in left, right, top and down side edges of a block
    seg_top_edge_std_dev = torch.std(seg_top_edge, dim=1, unbiased=True)
    seg_right_side_edge_std_dev = torch.std(seg_right_side_edge, dim=1, unbiased=True)
    seg_down_side_edge_std_dev = torch.std(seg_down_side_edge, dim=1, unbiased=True)
    seg_left_side_edge_std_dev = torch.std(seg_left_side_edge, dim=1, unbiased=True)

    # Check for segment in block exhibits impairedness, if the standard deviation of the segment is less than block_impaired_threshold.
    block_impaired = 0
    for seg_index in range(seg_top_edge.shape[0]):
        if (
            (seg_top_edge_std_dev[seg_index] < block_impaired_threshold)
            or (seg_right_side_edge_std_dev[seg_index] < block_impaired_threshold)
            or (seg_down_side_edge_std_dev[seg_index] < block_impaired_threshold)
            or (seg_left_side_edge_std_dev[seg_index] < block_impaired_threshold)
        ):
            block_impaired = 1
            break
    
    return block_impaired


def segment_edge(block_edge, nsegments, block_size, window_size):
    # Segment is defined as a collection of 6 contiguous pixels in a block edge
    segments = torch.zeros(nsegments, window_size)
    for i in range(nsegments):
        segments[i, :] = block_edge[i: window_size]
        if window_size <= (block_size + 1):
            window_size += 1
    return segments


@ARCH_REGISTRY.register()
class PIQE(torch.nn.Module):
    """
    PIQE module.

    Args:
        x (torch.Tensor): Input tensor of shape (B, C, H, W).

    Returns:
        torch.Tensor: PIQE score.
    """
    def get_masks(self,):
        assert self.results is not None, "Please calculate the piqe score first."
        return {
            'noticeable_artifacts_mask': self.results[1],
            'noise_mask': self.results[2],
            'activity_mask': self.results[3], 
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.results = piqe(x)
        return self.results[0] 
