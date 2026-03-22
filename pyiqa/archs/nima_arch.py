r"""NIMA model.
Reference:
    Talebi, Hossein, and Peyman Milanfar. "NIMA: Neural image assessment."
    IEEE transactions on image processing 27, no. 8 (2018): 3998-4011.

Created by: https://github.com/yunxiaoshi/Neural-IMage-Assessment/blob/master/model/model.py

Modified by: Chaofeng Chen (https://github.com/chaofengc)

"""

import torch
import torch.nn as nn
import timm
from pyiqa.utils.registry import ARCH_REGISTRY
from pyiqa.archs.arch_util import dist_to_mos, load_pretrained_network
from pyiqa.archs.arch_util import get_url_from_name

import torchvision.transforms as T

default_model_urls = {
    'vgg16-ava': get_url_from_name('NIMA_VGG16_ava-dc4e8265.pth'),
    'inception_resnet_v2-ava': get_url_from_name('NIMA_InceptionV2_ava-b0c77c00.pth'),
    'inception_resnet_v2-koniq': get_url_from_name('NIMA_koniq-250367ae.pth'),
    'inception_resnet_v2-spaq': get_url_from_name('NIMA-spaq-46a7fcb7.pth'),
}


@ARCH_REGISTRY.register()
class NIMA(nn.Module):
    """Neural IMage Assessment (NIMA) model.

    Modification:
        - for simplicity, we use global average pool for all models
        - we remove the dropout, because parameters with avg pool is much less.

    Args:
        base_model_name (str): Backbone model name supported by :mod:`timm`.
        train_dataset (str): Dataset tag used to select default checkpoint.
        num_classes (int): Number of output classes for score distribution.
        dropout_rate (float): Dropout ratio in classifier head.
        pretrained (bool): Whether to load pretrained NIMA weights.
        pretrained_model_path (str | None): Optional local checkpoint path.

    Notes:
        Typical input sizes are ``224`` (VGG/MobileNet) or ``299``
        (Inception-family backbones).
    """

    def __init__(
        self,
        base_model_name='vgg16',
        train_dataset='ava',
        num_classes=10,
        dropout_rate=0.0,
        pretrained=True,
        pretrained_model_path=None,
    ):
        super(NIMA, self).__init__()
        self.base_model = timm.create_model(
            base_model_name, pretrained=True, features_only=True
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        in_ch = self.base_model.feature_info.channels()[-1]
        self.num_classes = num_classes

        self.classifier = [
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=in_ch, out_features=num_classes),
        ]
        if num_classes > 1:
            self.classifier.append(nn.Softmax(dim=-1))
        self.classifier = nn.Sequential(*self.classifier)

        default_mean = self.base_model.pretrained_cfg['mean']
        default_std = self.base_model.pretrained_cfg['std']
        self.default_mean = torch.Tensor(default_mean).view(1, 3, 1, 1)
        self.default_std = torch.Tensor(default_std).view(1, 3, 1, 1)

        if pretrained and pretrained_model_path is None:
            url_key = f'{base_model_name}-{train_dataset}'
            load_pretrained_network(
                self, default_model_urls[url_key], True, weight_keys='params'
            )
        elif pretrained_model_path is not None:
            load_pretrained_network(
                self, pretrained_model_path, True, weight_keys='params'
            )

    def preprocess(self, x):
        """Resize/crop (eval mode) and normalize input image tensor."""
        if not self.training:
            x = T.functional.resize(x, self.base_model.default_cfg['input_size'][-1])
            x = T.functional.center_crop(
                x, self.base_model.default_cfg['input_size'][-1]
            )

        x = (x - self.default_mean.to(x)) / self.default_std.to(x)
        return x

    def forward(self, x, return_mos=True, return_dist=False):
        r"""Compute image quality using NIMA.

        Args:
            x (torch.Tensor): Input tensor with shape ``(N, 3, H, W)``.
            return_mos (bool): Whether to return MOS prediction.
            return_dist (bool): Whether to return score distribution.

        Returns:
            torch.Tensor | list[torch.Tensor]: One tensor when a single output is
            requested, otherwise ``[mos, dist]``.
        """
        # imagenet normalization of input is hard coded
        x = self.preprocess(x)
        x = self.base_model(x)[-1]
        x = self.global_pool(x)
        dist = self.classifier(x)
        mos = dist_to_mos(dist)
        return_list = []
        if return_mos:
            return_list.append(mos)
        if return_dist:
            return_list.append(dist)

        if len(return_list) > 1:
            return return_list
        else:
            return return_list[0]
