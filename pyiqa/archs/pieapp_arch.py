r"""PieAPP metric, proposed by

Prashnani, Ekta, Hong Cai, Yasamin Mostofi, and Pradeep Sen.
"Pieapp: Perceptual image-error assessment through pairwise preference."
In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 1808-1817. 2018.

Ref url: https://github.com/prashnani/PerceptualImageError/blob/master/model/PieAPPv0pt1_PT.py
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

default_model_urls = {
    'url': 'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/PieAPPv0.1-0937b014.pth'
}


class CompactLinear(nn.Module):

    def __init__(self):
        super().__init__()
        self.weight = nn.parameter.Parameter(torch.randn(1))
        self.bias = nn.parameter.Parameter(torch.randn(1))

    def forward(self, x):
        return x * self.weight + self.bias


@ARCH_REGISTRY.register()
class PieAPP(nn.Module):

    def __init__(self, patch_size=64, stride=27, pretrained=True, pretrained_model_path=None):
        super(PieAPP, self).__init__()
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

        if pretrained_model_path is not None:
            load_pretrained_network(self, pretrained_model_path)
        elif pretrained:
            load_pretrained_network(self, default_model_urls['url'])

        self.pretrained = pretrained

    def flatten(self, matrix):  # takes NxCxHxW input and outputs NxHWC
        return torch.flatten(matrix, 1)

    def compute_features(self, input):
        # conv1 -> relu -> conv2 -> relu -> pool2 -> conv3 -> relu
        x3 = F.relu(self.conv3(self.pool2(F.relu(self.conv2(F.relu(self.conv1(input)))))))
        # conv4 -> relu -> pool4 -> conv5 -> relu
        x5 = F.relu(self.conv5(self.pool4(F.relu(self.conv4(x3)))))
        # conv6 -> relu -> pool6 -> conv7 -> relu
        x7 = F.relu(self.conv7(self.pool6(F.relu(self.conv6(x5)))))
        # conv8 -> relu -> pool8 -> conv9 -> relu
        x9 = F.relu(self.conv9(self.pool8(F.relu(self.conv8(x7)))))
        # conv10 -> relu -> pool10 -> conv11 -> relU
        x11 = self.flatten(F.relu(self.conv11(self.pool10(F.relu(self.conv10(x9))))))
        # flatten and concatenate
        feature_ms = torch.cat((self.flatten(x3), self.flatten(x5), self.flatten(x7), self.flatten(x9), x11), 1)
        return feature_ms, x11

    def preprocess(self, x):
        """Default BGR in [0, 255] in original codes
        """
        x = x[:, [2, 1, 0]] * 255.
        return x

    def forward(self, dist, ref):
        assert dist.shape == ref.shape, f'Input and reference images should have the same shape, but got {dist.shape}'
        f' and {ref.shape}'

        if self.pretrained:
            dist = self.preprocess(dist)
            ref = self.preprocess(ref)

            image_A_patches = extract_2d_patches(dist, self.patch_size, self.stride, padding='none')
            image_ref_patches = extract_2d_patches(ref, self.patch_size, self.stride, padding='none')

        bsz, num_patches, c, psz, psz = image_A_patches.shape
        image_A_patches = image_A_patches.reshape(bsz * num_patches, c, psz, psz)
        image_ref_patches = image_ref_patches.reshape(bsz * num_patches, c, psz, psz)

        A_multi_scale, A_coarse = self.compute_features(image_A_patches)
        ref_multi_scale, ref_coarse = self.compute_features(image_ref_patches)
        diff_ms = ref_multi_scale - A_multi_scale
        diff_coarse = ref_coarse - A_coarse
        # per patch score: fc1_score -> relu -> fc2_score
        per_patch_score = self.ref_score_subtract(0.01 * self.fc2_score(F.relu(self.fc1_score(diff_ms))))
        per_patch_score = per_patch_score.view((-1, num_patches))
        # per patch weight: fc1_weight -> relu -> fc2_weight
        per_patch_weight = self.fc2_weight(F.relu(self.fc1_weight(diff_coarse))) + 1e-6
        per_patch_weight = per_patch_weight.view((-1, num_patches))

        score = (per_patch_weight * per_patch_score).sum(dim=-1) / per_patch_weight.sum(dim=-1)
        return score.squeeze()
