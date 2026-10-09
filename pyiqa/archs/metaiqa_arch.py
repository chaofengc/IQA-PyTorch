r"""MetaIQA Model wrapper for pyiqa.
Reference: Zhu, Hancheng, et al. "MetaIQA: Deep Meta-Learning for No-Reference Image Quality Assessment." CVPR 2020.
Re-implemented and optimized for pyiqa ecosystem.
"""

import os
import sys
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image

from pyiqa.utils.registry import ARCH_REGISTRY
from pyiqa.archs.arch_util import load_pretrained_network, get_url_from_name, load_file_from_url

default_model_urls = {
    'meta-train-seed': get_url_from_name('Metaiqa_prior.pth'),
    'meta-infer-ready': get_url_from_name('Metaiqa_livec.pth'),
}


class BaselineModel1(nn.Module):
    """Convolutional regression baseline used by the MetaIQA meta-learning model.

    """
    def __init__(self, num_classes=1, keep_probability=0.5, inputsize=1000):
        """Initialize the baseline model1 and configure its layers, parameters, and optional pretrained state.

        Args:
            num_classes: Number of items, stages, tokens, or channels configured for this operation.
            keep_probability: Probability of retaining activations during dropout.
            inputsize: Expected spatial size of the model input.
        """
        super(BaselineModel1, self).__init__()
        self.fc1 = nn.Linear(inputsize, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.drop_prob = (1 - keep_probability)
        self.relu1 = nn.PReLU()
        self.drop1 = nn.Dropout(self.drop_prob)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.relu2 = nn.PReLU()
        self.drop2 = nn.Dropout(p=self.drop_prob)
        self.fc3 = nn.Linear(512, num_classes)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        """Compute the MetaIQA prediction for the supplied image or image pair.

        Args:
            x: Input tensor or activation; image entry points generally use ``(B, C, H, W)`` layout, while internal layers may use other layouts.

        Returns:
            Quality score tensor, generally one score per input image or image pair.
        """
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.drop1(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.drop2(out)
        out = self.fc3(out)
        out = self.sig(out)
        return out


@ARCH_REGISTRY.register()
class MetaIQA(nn.Module):
    """Meta-learning model that predicts image quality from query/reference image features.

    """
    def __init__(self, pretrained=True, pretrained_model_path=None, **kwargs):
        """Initialize the meta iqa and configure its layers, parameters, and optional pretrained state.

        Args:
            pretrained: Whether to initialize or load pretrained weights.
            pretrained_model_path: Optional local checkpoint path; ``None`` selects the implementation default.
            **kwargs: kwargs value used to configure or compute this operation.
        """
        super(MetaIQA, self).__init__()

        self.metric_mode = 'NR'
        self.lower_better = False

        self.resnet_layer = models.resnet18(weights=None)
        self.head = BaselineModel1(num_classes=1, keep_probability=0.5, inputsize=1000)

        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        if pretrained_model_path is not None:
            load_pretrained_network(self, pretrained_model_path, strict=False, weight_keys=None)
        elif pretrained:
            main_script = os.path.basename(sys.argv[0])
            is_train_env = (
                    'train.py' in main_script or
                    'train_nsplits.py' in main_script
            )

            if not is_train_env:
                ckpt_path = load_file_from_url(default_model_urls['meta-infer-ready'])
                print(f'Loading pretrained model from {ckpt_path} for inference...')
                state_dict = torch.load(ckpt_path, map_location='cpu')
                if 'params' in state_dict:
                    state_dict = state_dict['params']
                elif 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']

                clean_dict = {}
                for k, v in state_dict.items():
                    new_k = k.replace('net_g.', '').replace('module.', '').replace('net.', 'head.')
                    clean_dict[new_k] = v

                self.load_state_dict(clean_dict, strict=False)
            else:
                load_pretrained_network(self, default_model_urls['meta-train-seed'], strict=False, weight_keys=None)

    def preprocess(self, x):
        """Perform the preprocess operation for meta iqa.

        Args:
            x: Input tensor or activation; image entry points generally use ``(B, C, H, W)`` layout, while internal layers may use other layouts.

        Returns:
            Computed result; type and shape follow the supplied inputs and model configuration.
        """
        if x.shape[2:] != (224, 224):
            x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = (x - self.mean) / self.std
        return x

    def forward(self, x, ref=None):
        """Compute the MetaIQA prediction for the supplied image or image pair.

        Args:
            x: Input tensor or activation; image entry points generally use ``(B, C, H, W)`` layout, while internal layers may use other layouts.
            ref: Reference-image tensor for full-reference quality assessment.

        Returns:
            Quality score tensor, generally one score per input image or image pair.
        """
        x = self.preprocess(x)

        feat = self.resnet_layer(x)
        score = self.head(feat)

        return score.view(-1, 1)

