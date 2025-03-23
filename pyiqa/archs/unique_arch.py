"""LIQE Model

github repo link: https://github.com/zwx8981/UNIQUE

Cite as:
@article{zhang2021uncertainty,
  title   = {Uncertainty-aware blind image quality assessment in the laboratory and wild},
  author  = {Zhang, Weixia and Ma, Kede and Zhai, Guangtao and Yang, Xiaokang},
  journal = {IEEE Transactions on Image Processing},
  volume  = {30},
  pages   = {3474--3486},
  month   = {Mar.},
  year    = {2021}
}

"""

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from pyiqa.utils.registry import ARCH_REGISTRY
from pyiqa.archs.arch_util import load_pretrained_network
from pyiqa.archs.arch_util import get_url_from_name

default_model_urls = {
    'mix': get_url_from_name('UNIQUE.pt'),
}


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)

    def forward(self, x):
        return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[
            None, :, None, None
        ]


class BCNN(nn.Module):
    def __init__(self, thresh=1e-8, is_vec=True, input_dim=512):
        super(BCNN, self).__init__()
        self.thresh = thresh
        self.is_vec = is_vec
        self.output_dim = input_dim * input_dim

    def _bilinearpool(self, x):
        batchSize, dim, h, w = x.data.shape
        x = x.reshape(batchSize, dim, h * w)
        x = 1.0 / (h * w) * x.bmm(x.transpose(1, 2))
        return x

    def _signed_sqrt(self, x):
        x = torch.mul(x.sign(), torch.sqrt(x.abs() + self.thresh))
        return x

    def _l2norm(self, x):
        x = nn.functional.normalize(x)
        return x

    def forward(self, x):
        x = self._bilinearpool(x)
        x = self._signed_sqrt(x)
        if self.is_vec:
            x = x.view(x.size(0), -1)
        x = self._l2norm(x)
        return x


@ARCH_REGISTRY.register()
class UNIQUE(nn.Module):
    """Full UNIQUE network.
    Args:
        - default_mean (list): Default mean value.
        - default_std (list): Default std value.

    """

    def __init__(self):
        super(UNIQUE, self).__init__()

        self.backbone = torchvision.models.resnet34(pretrained=True)
        outdim = 2
        self.representation = BCNN()
        self.fc = nn.Linear(512 * 512, outdim)
        self.preprocess = Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        pretrained_model_path = default_model_urls['mix']
        load_pretrained_network(self, pretrained_model_path, True)

    def forward(self, x):
        r"""Compute IQA using UNIQUE model.

        Args:
            X: An input tensor with (N, C, H, W) shape. RGB channel order for colour images.

        Returns:
            Value of UNIQUE model.

        """
        x = self.preprocess(x)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.representation(x)
        x = self.fc(x)

        mean = x[:, 0]

        return mean
