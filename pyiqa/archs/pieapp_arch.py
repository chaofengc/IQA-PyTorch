r"""PieAPP metric, proposed by

Prashnani, Ekta, Hong Cai, Yasamin Mostofi, and Pradeep Sen.
"Pieapp: Perceptual image-error assessment through pairwise preference."
In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 1808-1817. 2018.

Ref url: https://github.com/prashnani/PerceptualImageError
Modified by: Chaofeng Chen (https://github.com/chaofengc)

!!! Important Note: to keep simple test process and fair comparison with other methods,
                    we use zero padding and extract subpatches only once
                    rather than from multiple subimages as the original codes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyiqa.utils.registry import ARCH_REGISTRY
from pyiqa.archs.arch_util import load_pretrained_network
from .func_util import extract_2d_patches
from pyiqa.archs.arch_util import get_url_from_name

default_model_urls = {'url': get_url_from_name('PieAPPv0.1-0937b014.pth')}


class CompactLinear(nn.Module):
    """Compact learned linear regressor used to predict the PieAPP quality score.

    """
    def __init__(self):
        """Initialize the compact linear and configure its layers, parameters, and optional pretrained state.

        """
        super().__init__()
        self.weight = nn.parameter.Parameter(torch.randn(1))
        self.bias = nn.parameter.Parameter(torch.randn(1))

    def forward(self, x):
        """Compute the PieAPP prediction for the supplied image or image pair.

        Args:
            x: Input tensor or activation; image entry points generally use ``(B, C, H, W)`` layout, while internal layers may use other layouts.

        Returns:
            Quality score tensor, generally one score per input image or image pair.
        """
        return x * self.weight + self.bias


@ARCH_REGISTRY.register()
class PieAPP(nn.Module):
    r"""
    PieAPP model implementation.

    Args:
        - patch_size (int): Size of the patches to extract from the images.
        - stride (int): Stride to use when extracting patches.
        - patch_batch_size (int or None): Maximum number of patches to process
          together during inference. ``None`` processes all patches at once.
        - pretrained (bool): Whether to use a pretrained model or not.
        - pretrained_model_path (str): Path to the pretrained model.

    Methods:
        - flatten(matrix): Takes NxCxHxW input and outputs NxHWC.
        compute_features(input): Computes the features of the input image.
        - preprocess(x): Preprocesses the input image.
        forward(dist, ref): Computes the PieAPP score between the distorted and reference images.
    """

    def __init__(
        self,
        patch_size=64,
        stride=27,
        pretrained=True,
        pretrained_model_path=None,
        patch_batch_size=64,
    ):
        """Initialize the pie app and configure its layers, parameters, and optional pretrained state.

        Args:
            patch_size: Spatial size of each non-overlapping image patch or attention window.
            stride: Spatial stride for the convolutional stage.
            pretrained: Whether to initialize or load pretrained weights.
            pretrained_model_path: Optional local checkpoint path; ``None`` selects the implementation default.
            patch_batch_size: Maximum number of patches evaluated together
                during inference. Positive values stream patches in bounded
                batches. ``None`` restores the original all-at-once evaluation.
        """
        super(PieAPP, self).__init__()
        if (
            patch_batch_size is not None
            and (
                isinstance(patch_batch_size, bool)
                or not isinstance(patch_batch_size, int)
                or patch_batch_size <= 0
            )
        ):
            raise ValueError('patch_batch_size must be a positive integer or None.')

        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool6 = nn.MaxPool2d(2, 2)
        self.conv7 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv8 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool8 = nn.MaxPool2d(2, 2)
        self.conv9 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv10 = nn.Conv2d(256, 512, 3, padding=1)
        self.pool10 = nn.MaxPool2d(2, 2)
        self.conv11 = nn.Conv2d(512, 512, 3, padding=1)
        self.fc1_score = nn.Linear(120832, 512)
        self.fc2_score = nn.Linear(512, 1)
        self.fc1_weight = nn.Linear(2048, 512)
        self.fc2_weight = nn.Linear(512, 1)
        self.ref_score_subtract = CompactLinear()

        self.patch_size = patch_size
        self.stride = stride
        self.patch_batch_size = patch_batch_size

        if pretrained_model_path is not None:
            load_pretrained_network(self, pretrained_model_path)
        elif pretrained:
            load_pretrained_network(self, default_model_urls['url'])

        self.pretrained = pretrained

    def flatten(self, matrix):  # takes NxCxHxW input and outputs NxHWC
        """Perform the flatten operation for pie app.

        Args:
            matrix: Feature matrix with batch, channel, and spatial dimensions.

        Returns:
            Computed result; type and shape follow the supplied inputs and model configuration.
        """
        return torch.flatten(matrix, 1)

    def compute_features(self, input):
        # conv1 -> relu -> conv2 -> relu -> pool2 -> conv3 -> relu
        """Compute and return features from the supplied inputs.

        Args:
            input: Image or feature tensor supplied to the feature extractor.

        Returns:
            Computed result; type and shape follow the supplied inputs and model configuration.
        """
        x3 = F.relu(
            self.conv3(self.pool2(F.relu(self.conv2(F.relu(self.conv1(input))))))
        )
        # conv4 -> relu -> pool4 -> conv5 -> relu
        x5 = F.relu(self.conv5(self.pool4(F.relu(self.conv4(x3)))))
        # conv6 -> relu -> pool6 -> conv7 -> relu
        x7 = F.relu(self.conv7(self.pool6(F.relu(self.conv6(x5)))))
        # conv8 -> relu -> pool8 -> conv9 -> relu
        x9 = F.relu(self.conv9(self.pool8(F.relu(self.conv8(x7)))))
        # conv10 -> relu -> pool10 -> conv11 -> relU
        x11 = self.flatten(F.relu(self.conv11(self.pool10(F.relu(self.conv10(x9))))))
        # flatten and concatenate
        feature_ms = torch.cat(
            (
                self.flatten(x3),
                self.flatten(x5),
                self.flatten(x7),
                self.flatten(x9),
                x11,
            ),
            1,
        )
        return feature_ms, x11

    def preprocess(self, x):
        """Default BGR in [0, 255] in original codes"""
        x = x[:, [2, 1, 0]] * 255.0
        return x

    def _compute_patch_outputs(self, dist_patches, ref_patches):
        """Compute scores and weights for one batch of paired patches."""
        dist_multi_scale, dist_coarse = self.compute_features(dist_patches)
        ref_multi_scale, ref_coarse = self.compute_features(ref_patches)
        diff_ms = ref_multi_scale - dist_multi_scale
        diff_coarse = ref_coarse - dist_coarse

        per_patch_score = self.ref_score_subtract(
            0.01 * self.fc2_score(F.relu(self.fc1_score(diff_ms)))
        )
        per_patch_weight = (
            self.fc2_weight(F.relu(self.fc1_weight(diff_coarse))) + 1e-6
        )
        return per_patch_score, per_patch_weight

    def _extract_patch_grid(self, image):
        """Return a zero-copy grid view over all valid image patches."""
        return image.unfold(2, self.patch_size, self.stride).unfold(
            3, self.patch_size, self.stride
        )

    def _get_patch_batch(self, patch_grid, start, end):
        """Materialize a contiguous batch from a flattened patch-grid range."""
        patch_rows = patch_grid.shape[2]
        patch_columns = patch_grid.shape[3]
        patches_per_image = patch_rows * patch_columns
        flat_indices = torch.arange(start, end, device=patch_grid.device)
        image_indices = flat_indices // patches_per_image
        patch_indices = flat_indices % patches_per_image
        row_indices = patch_indices // patch_columns
        column_indices = patch_indices % patch_columns
        return patch_grid[image_indices, :, row_indices, column_indices]

    def forward(self, dist, ref):
        """Compute the PieAPP prediction for the supplied image or image pair.

        Args:
            dist: Distorted-image tensor for full-reference quality assessment.
            ref: Reference-image tensor for full-reference quality assessment.

        Returns:
            Quality score tensor, generally one score per input image or image pair.
        """
        assert dist.shape == ref.shape, (
            f'Input and reference images should have the same shape, but got {dist.shape}'
        )
        f' and {ref.shape}'

        dist = self.preprocess(dist)
        ref = self.preprocess(ref)

        if not self.training and self.patch_batch_size is not None:
            image_A_patch_grid = self._extract_patch_grid(dist)
            image_ref_patch_grid = self._extract_patch_grid(ref)
            bsz = image_A_patch_grid.shape[0]
            num_patches = (
                image_A_patch_grid.shape[2] * image_A_patch_grid.shape[3]
            )
            total_patches = bsz * num_patches
            patch_batch_size = self.patch_batch_size
            patch_pairs = (
                (
                    self._get_patch_batch(
                        image_A_patch_grid,
                        start,
                        min(start + patch_batch_size, total_patches),
                    ),
                    self._get_patch_batch(
                        image_ref_patch_grid,
                        start,
                        min(start + patch_batch_size, total_patches),
                    ),
                )
                for start in range(0, total_patches, patch_batch_size)
            )
        else:
            if not self.training:
                image_A_patches = extract_2d_patches(
                    dist, self.patch_size, self.stride, padding='none'
                )
                image_ref_patches = extract_2d_patches(
                    ref, self.patch_size, self.stride, padding='none'
                )
            else:
                image_A_patches, image_ref_patches = dist, ref
                image_A_patches = image_A_patches.unsqueeze(1)
                image_ref_patches = image_ref_patches.unsqueeze(1)

            bsz, num_patches, c, patch_height, patch_width = image_A_patches.shape
            image_A_patches = image_A_patches.reshape(
                bsz * num_patches, c, patch_height, patch_width
            )
            image_ref_patches = image_ref_patches.reshape(
                bsz * num_patches, c, patch_height, patch_width
            )
            total_patches = bsz * num_patches
            patch_batch_size = total_patches
            patch_pairs = (
                (
                    image_A_patches[start : start + patch_batch_size],
                    image_ref_patches[start : start + patch_batch_size],
                )
                for start in range(0, total_patches, patch_batch_size)
            )

        score_chunks = []
        weight_chunks = []
        for image_A_patch_batch, image_ref_patch_batch in patch_pairs:
            chunk_score, chunk_weight = self._compute_patch_outputs(
                image_A_patch_batch, image_ref_patch_batch
            )
            score_chunks.append(chunk_score)
            weight_chunks.append(chunk_weight)

        if len(score_chunks) == 1:
            per_patch_score = score_chunks[0]
            per_patch_weight = weight_chunks[0]
        else:
            per_patch_score = torch.cat(score_chunks)
            per_patch_weight = torch.cat(weight_chunks)

        per_patch_score = per_patch_score.view((-1, num_patches))
        per_patch_weight = per_patch_weight.view((-1, num_patches))

        score = (per_patch_weight * per_patch_score).sum(dim=-1) / per_patch_weight.sum(
            dim=-1
        )
        return score.reshape(bsz, 1)
