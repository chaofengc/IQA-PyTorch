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

default_model_urls = {
    'ava': 'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/NIMA_VGG16_ava-dc4e8265.pth'
}


@ARCH_REGISTRY.register()
class NIMA(nn.Module):
    """Neural IMage Assessment model.

    Modification:
        - for simplicity, we use global average pool for all models
        - we remove the dropout, because parameters with avg pool is much less.

    Args:
        base_model_name: pretrained model to extract features, can be any models supported by timm.
                         Models used in the paper: vgg16, inception_resnet_v2, mobilenetv2_100

        default input shape:
            - vgg and mobilenet: (N, 3, 224, 224)
            - inception: (N, 3, 299, 299)
    """

    def __init__(
        self,
        base_model_name='vgg16',
        num_classes=10,
        dropout_rate=0.,
        pretrained=True,
        pretrained_model_path=None,
        default_mean=[0.485, 0.456, 0.406],
        default_std=[0.229, 0.224, 0.225],
    ):
        super(NIMA, self).__init__()
        self.base_model = timm.create_model(base_model_name, pretrained=True, features_only=True)

        if pretrained_model_path is None and pretrained:
            url_key = 'ava' if isinstance(pretrained, bool) else pretrained
            num_class = 10 if url_key == 'ava' else num_class
            pretrained_model_path = default_model_urls[url_key]

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        in_ch = self.base_model.feature_info.channels()[-1]
        self.num_classes = num_classes

        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(p=dropout_rate),
                                        nn.Linear(in_features=in_ch, out_features=num_classes), nn.Softmax(dim=1))

        self.default_mean = torch.Tensor(default_mean).view(1, 3, 1, 1)
        self.default_std = torch.Tensor(default_std).view(1, 3, 1, 1)

        if pretrained_model_path is not None:
            load_pretrained_network(self, pretrained_model_path, True, weight_keys='params')

    def preprocess(self, x):
        x = (x - self.default_mean.to(x)) / self.default_std.to(x)
        return x

    def forward(self, x, return_mos=True, return_dist=False):
        r"""Computation image quality using NIMA.
        Args:
            x: An input tensor. Shape :math:`(N, C, H, W)`.
            return_mos: Whether to return mos_score.
            retuen_dist: Whether to return dist_score.

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
