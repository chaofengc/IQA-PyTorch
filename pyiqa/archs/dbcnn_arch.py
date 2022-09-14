r"""DBCNN Metric

Created by: https://github.com/zwx8981/DBCNN-PyTorch/blob/master/DBCNN.py

Modified by: Chaofeng Chen (https://github.com/chaofengc)

"""

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from pyiqa.utils.registry import ARCH_REGISTRY
from pyiqa.archs.arch_util import load_pretrained_network

default_model_urls = {
    'csiq': 'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/DBCNN_CSIQ-8677d071.pth',
    'tid2008': 'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/DBCNN_TID2008-4b47c5d1.pth',
    'tid2013': 'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/DBCNN_TID2013-485d021d.pth',
    'live': 'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/DBCNN_LIVE-97262bf4.pth',
    'livec': 'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/DBCNN_LIVEC-83f6dad3.pth',
    'livem': 'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/DBCNN_LIVEM-698474e3.pth',
    'koniq': 'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/DBCNN_KonIQ10k-254e8241.pth',
}


class SCNN(nn.Module):
    """Network branch for synthetic distortions.
    Args:
        use_bn (Boolean): Whether to use batch normalization.

    Modified from https://github.com/zwx8981/DBCNN-PyTorch/blob/master/SCNN.py

    """

    def __init__(self, use_bn=True):
        super(SCNN, self).__init__()

        self.num_class = 39
        self.use_bn = use_bn
        self.features = nn.Sequential(
            *self._make_layers(3, 48, 3, 1, 1),
            *self._make_layers(48, 48, 3, 2, 1),
            *self._make_layers(48, 64, 3, 1, 1),
            *self._make_layers(64, 64, 3, 2, 1),
            *self._make_layers(64, 64, 3, 1, 1),
            *self._make_layers(64, 64, 3, 2, 1),
            *self._make_layers(64, 128, 3, 1, 1),
            *self._make_layers(128, 128, 3, 1, 1),
            *self._make_layers(128, 128, 3, 2, 1),
        )

        self.pooling = nn.AdaptiveAvgPool2d(1)

        self.projection = nn.Sequential(
            *self._make_layers(128, 256, 1, 1, 0),
            *self._make_layers(256, 256, 1, 1, 0),
        )

        self.classifier = nn.Linear(256, self.num_class)

    def _make_layers(self, in_ch, out_ch, ksz, stride, pad):
        if self.use_bn:
            layers = [
                nn.Conv2d(in_ch, out_ch, ksz, stride, pad),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(True),
            ]
        else:
            layers = [
                nn.Conv2d(in_ch, out_ch, ksz, stride, pad),
                nn.ReLU(True),
            ]

        return layers

    def forward(self, X):
        X = self.features(X)
        X = self.pooling(X)
        X = self.projection(X)
        X = X.view(X.shape[0], -1)
        X = self.classifier(X)

        return X


@ARCH_REGISTRY.register()
class DBCNN(nn.Module):
    """Full DBCNN network.
    Args:
        fc (Boolean): Whether initialize the fc layers.
        use_bn (Boolean): Whether use batch normalization.
        pretrained_scnn_path (String): Pretrained scnn path.
        default_mean (list): Default mean value.
        default_std (list): Default std value.

    Reference:
        Zhang, Weixia, et al. "Blind image quality assessment using
        a deep bilinear convolutional neural network." IEEE Transactions
        on Circuits and Systems for Video Technology 30.1 (2018): 36-47.

        """

    def __init__(
        self,
        fc=True,
        use_bn=True,
        pretrained_scnn_path=None,
        pretrained=True,
        pretrained_model_path=None,
        default_mean=[0.485, 0.456, 0.406],
        default_std=[0.229, 0.224, 0.225],
    ):
        super(DBCNN, self).__init__()

        # Convolution and pooling layers of VGG-16.
        self.features1 = torchvision.models.vgg16(pretrained=True).features
        self.features1 = nn.Sequential(*list(self.features1.children())[:-1])
        scnn = SCNN(use_bn=use_bn)
        if pretrained_scnn_path is not None:
            load_pretrained_network(scnn, pretrained_scnn_path)

        self.features2 = scnn.features

        # Linear classifier.
        self.fc = torch.nn.Linear(512 * 128, 1)

        self.default_mean = torch.Tensor(default_mean).view(1, 3, 1, 1)
        self.default_std = torch.Tensor(default_std).view(1, 3, 1, 1)

        if fc:
            # Freeze all previous layers.
            for param in self.features1.parameters():
                param.requires_grad = False
            for param in scnn.parameters():
                param.requires_grad = False
            # Initialize the fc layers.
            nn.init.kaiming_normal_(self.fc.weight.data)
            if self.fc.bias is not None:
                nn.init.constant_(self.fc.bias.data, val=0)

        if pretrained_model_path is None and pretrained:
            url_key = 'koniq' if isinstance(pretrained, bool) else pretrained
            pretrained_model_path = default_model_urls[url_key]
        if pretrained_model_path is not None:
            load_pretrained_network(self, pretrained_model_path, True, 'params')

    def preprocess(self, x):
        x = (x - self.default_mean.to(x)) / self.default_std.to(x)
        return x

    def forward(self, X):
        r"""Compute IQA using DBCNN model.

        Args:
            X: An input tensor with (N, C, H, W) shape. RGB channel order for colour images.

        Returns:
            Value of DBCNN model.

        """
        X = self.preprocess(X)

        X1 = self.features1(X)
        X2 = self.features2(X)

        N, _, H, W = X1.shape
        N, _, H2, W2 = X2.shape

        if (H != H2) or (W != W2):
            X2 = F.interpolate(X2, (H, W), mode='bilinear', align_corners=True)

        X1 = X1.view(N, 512, H * W)
        X2 = X2.view(N, 128, H * W)
        X = torch.bmm(X1, torch.transpose(X2, 1, 2)) / (H * W)  # Bilinear
        X = X.view(N, 512 * 128)
        X = torch.sqrt(X + 1e-8)
        X = torch.nn.functional.normalize(X)
        X = self.fc(X)

        return X
