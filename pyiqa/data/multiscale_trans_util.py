r"""Preprocessing utils for Multiscale Transformer

Reference: https://github.com/google-research/google-research/blob/5c622d523c/musiq/model/preprocessing.py

Modified: Chaofeng Chen (https://github.com/chaofengc)
"""

from unittest.mock import patch
import numpy as np
import math
from os import path as osp

import torch
from torch.nn import functional as F


def extract_image_patches(x, kernel, stride=1, dilation=1):
    """
    Ref: https://stackoverflow.com/a/65886666
    """
    # Do TF 'SAME' Padding
    b, c, h, w = x.shape
    h2 = math.ceil(h / stride)
    w2 = math.ceil(w / stride)
    pad_row = (h2 - 1) * stride + (kernel - 1) * dilation + 1 - h
    pad_col = (w2 - 1) * stride + (kernel - 1) * dilation + 1 - w
    x = F.pad(x, (pad_col // 2, pad_col - pad_col // 2, pad_row // 2, pad_row - pad_row // 2))

    # Extract patches
    patches = F.unfold(x, kernel, dilation, stride=stride)
    return patches


def _ceil_divide_int(x, y):
    """Returns ceil(x / y) as int"""
    return int(math.ceil(x / y))


def resize_preserve_aspect_ratio(image, h, w, longer_side_length):
    """Aspect-ratio-preserving resizing with tf.image.ResizeMethod.GAUSSIAN.
    Args:
      image: The image tensor (n_crops, c, h, w).
      h: Height of the input image.
      w: Width of the input image.
      longer_side_length: The length of the longer side after resizing.
    Returns:
      A tuple of [Image after resizing, Resized height, Resized width].
    """
    # Computes the height and width after aspect-ratio-preserving resizing.
    ratio = longer_side_length / max(h, w)
    rh = round(h * ratio)
    rw = round(w * ratio)

    resized = F.interpolate(image, (rh, rw), mode='bicubic', align_corners=False)
    return resized, rh, rw


def _pad_or_cut_to_max_seq_len(x, max_seq_len):
    """Pads (or cuts) patch tensor `max_seq_len`.
    Args:
        x: input tensor of shape (n_crops, c, num_patches).
        max_seq_len: max sequence length.
    Returns:
        The padded or cropped tensor of shape (n_crops, c, max_seq_len).
    """
    # Shape of x (n_crops, c, num_patches)
    # Padding makes sure that # patches > max_seq_length. Note that it also
    # makes the input mask zero for shorter input.
    n_crops, c, num_patches = x.shape
    paddings = torch.zeros((n_crops, c, max_seq_len)).to(x)
    x = torch.cat([x, paddings], dim=-1)
    x = x[:, :, :max_seq_len]
    return x


def get_hashed_spatial_pos_emb_index(grid_size, count_h, count_w):
    """Get hased spatial pos embedding index for each patch.
    The size H x W is hashed to grid_size x grid_size.
    Args:
      grid_size: grid size G for the hashed-based spatial positional embedding.
      count_h: number of patches in each row for the image.
      count_w: number of patches in each column for the image.
    Returns:
      hashed position of shape (1, HxW). Each value corresponded to the hashed
      position index in [0, grid_size x grid_size).
    """
    pos_emb_grid = torch.arange(grid_size).float()

    pos_emb_hash_w = pos_emb_grid.reshape(1, 1, grid_size)
    pos_emb_hash_w = F.interpolate(pos_emb_hash_w, (count_w), mode='nearest')
    pos_emb_hash_w = pos_emb_hash_w.repeat(1, count_h, 1)

    pos_emb_hash_h = pos_emb_grid.reshape(1, 1, grid_size)
    pos_emb_hash_h = F.interpolate(pos_emb_hash_h, (count_h), mode='nearest')
    pos_emb_hash_h = pos_emb_hash_h.transpose(1, 2)
    pos_emb_hash_h = pos_emb_hash_h.repeat(1, 1, count_w)

    pos_emb_hash = pos_emb_hash_h * grid_size + pos_emb_hash_w

    pos_emb_hash = pos_emb_hash.reshape(1, -1)
    return pos_emb_hash


def _extract_patches_and_positions_from_image(image, patch_size, patch_stride, hse_grid_size, n_crops, h, w, c,
                                              scale_id, max_seq_len):
    """Extracts patches and positional embedding lookup indexes for a given image.
    Args:
      image: the input image of shape [n_crops, c, h, w]
      patch_size: the extracted patch size.
      patch_stride: stride for extracting patches.
      hse_grid_size: grid size for hash-based spatial positional embedding.
      n_crops: number of crops from the input image.
      h: height of the image.
      w: width of the image.
      c: number of channels for the image.
      scale_id: the scale id for the image in the multi-scale representation.
      max_seq_len: maximum sequence length for the number of patches. If
        max_seq_len = 0, no patch is returned. If max_seq_len < 0 then we return
        all the patches.
    Returns:
      A concatenating vector of (patches, HSE, SCE, input mask). The tensor shape
      is (n_crops, num_patches, patch_size * patch_size * c + 3).
    """
    n_crops, c, h, w = image.shape
    p = extract_image_patches(image, patch_size, patch_stride)
    assert p.shape[1] == c * patch_size**2

    count_h = _ceil_divide_int(h, patch_stride)
    count_w = _ceil_divide_int(w, patch_stride)

    # Shape (1, num_patches)
    spatial_p = get_hashed_spatial_pos_emb_index(hse_grid_size, count_h, count_w)
    # Shape (n_crops, 1, num_patches)
    spatial_p = spatial_p.unsqueeze(1).repeat(n_crops, 1, 1)
    scale_p = torch.ones_like(spatial_p) * scale_id
    mask_p = torch.ones_like(spatial_p)

    # Concatenating is a hacky way to pass both patches, positions and input
    # mask to the model.
    # Shape (n_crops, c * patch_size * patch_size + 3, num_patches)
    out = torch.cat([p, spatial_p.to(p), scale_p.to(p), mask_p.to(p)], dim=1)
    if max_seq_len >= 0:
        out = _pad_or_cut_to_max_seq_len(out, max_seq_len)
    return out


def get_multiscale_patches(image,
                           patch_size=32,
                           patch_stride=32,
                           hse_grid_size=10,
                           longer_side_lengths=[224, 384],
                           max_seq_len_from_original_res=None):
    """Extracts image patches from multi-scale representation.
    Args:
      image: input image tensor with shape [n_crops, 3, h, w]
      patch_size: patch size.
      patch_stride: patch stride.
      hse_grid_size: Hash-based positional embedding grid size.
      longer_side_lengths: List of longer-side lengths for each scale in the
        multi-scale representation.
      max_seq_len_from_original_res: Maximum number of patches extracted from
        original resolution. <0 means use all the patches from the original
        resolution. None means we don't use original resolution input.
    Returns:
      A concatenating vector of (patches, HSE, SCE, input mask). The tensor shape
      is (n_crops, num_patches, patch_size * patch_size * c + 3).
    """
    # Sorting the list to ensure a deterministic encoding of the scale position.
    longer_side_lengths = sorted(longer_side_lengths)

    if len(image.shape) == 3:
        image = image.unsqueeze(0)

    n_crops, c, h, w = image.shape

    outputs = []
    for scale_id, longer_size in enumerate(longer_side_lengths):
        resized_image, rh, rw = resize_preserve_aspect_ratio(image, h, w, longer_size)

        max_seq_len = int(np.ceil(longer_size / patch_stride)**2)
        out = _extract_patches_and_positions_from_image(resized_image, patch_size, patch_stride, hse_grid_size, n_crops,
                                                        rh, rw, c, scale_id, max_seq_len)
        outputs.append(out)

    if max_seq_len_from_original_res is not None:
        out = _extract_patches_and_positions_from_image(image, patch_size, patch_stride, hse_grid_size, n_crops, h, w,
                                                        c, len(longer_side_lengths), max_seq_len_from_original_res)
        outputs.append(out)

    outputs = torch.cat(outputs, dim=-1)
    return outputs.transpose(1, 2)
