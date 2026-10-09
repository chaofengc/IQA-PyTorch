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
from pyiqa.utils.registry import ARCH_REGISTRY
from pyiqa.archs.arch_util import load_pretrained_network
from pyiqa.archs.arch_util import get_url_from_name

try:
    from torchvision.models import ResNet34_Weights
except ImportError:
    ResNet34_Weights = None

default_model_urls = {
    'mix': get_url_from_name('UNIQUE.pt'),
}


class Normalize(nn.Module):
    """Channel-wise normalization module."""

    def __init__(self, mean, std):
        """Initialize the normalize and configure its layers, parameters, and optional pretrained state.

        Args:
            mean: Per-channel input normalization means.
            std: Per-channel input normalization standard deviations.
        """
        super(Normalize, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)

    def forward(self, x):
        """Apply the normalize computation to the provided activations.

        Args:
            x: Input tensor or activation; image entry points generally use ``(B, C, H, W)`` layout, while internal layers may use other layouts.

        Returns:
            Transformed tensor or feature representation; shape follows the layer configuration.
        """
        return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[
            None, :, None, None
        ]


class BCNN(nn.Module):
    """Bilinear CNN pooling block used in UNIQUE."""

    def __init__(self, thresh=1e-8, is_vec=True, input_dim=512):
        """Initialize the bcnn and configure its layers, parameters, and optional pretrained state.

        Args:
            thresh: Threshold used to select or binarize the prediction.
            is_vec: Whether the input is already represented as a vector.
            input_dim: Input feature dimension expected by the projection layer.
        """
        super(BCNN, self).__init__()
        self.thresh = thresh
        self.is_vec = is_vec
        self.output_dim = input_dim * input_dim

    def _bilinearpool(self, x):
        """Perform the internal bilinearpool operation used by bcnn.

        Args:
            x: Input tensor or activation; image entry points generally use ``(B, C, H, W)`` layout, while internal layers may use other layouts.

        Returns:
            Computed result; type and shape follow the supplied inputs and model configuration.
        """
        batchSize, dim, h, w = x.data.shape
        x = x.reshape(batchSize, dim, h * w)
        x = 1.0 / (h * w) * x.bmm(x.transpose(1, 2))
        return x

    def _signed_sqrt(self, x):
        """Perform the internal signed sqrt operation used by bcnn.

        Args:
            x: Input tensor or activation; image entry points generally use ``(B, C, H, W)`` layout, while internal layers may use other layouts.

        Returns:
            Computed result; type and shape follow the supplied inputs and model configuration.
        """
        x = torch.mul(x.sign(), torch.sqrt(x.abs() + self.thresh))
        return x

    def _l2norm(self, x):
        """Perform the internal l2norm operation used by bcnn.

        Args:
            x: Input tensor or activation; image entry points generally use ``(B, C, H, W)`` layout, while internal layers may use other layouts.

        Returns:
            Computed result; type and shape follow the supplied inputs and model configuration.
        """
        x = nn.functional.normalize(x)
        return x

    def forward(self, x):
        """Compute the UNIQUE prediction for the supplied image or image pair.

        Args:
            x: Input tensor or activation; image entry points generally use ``(B, C, H, W)`` layout, while internal layers may use other layouts.

        Returns:
            Quality score tensor, generally one score per input image or image pair.
        """
        x = self._bilinearpool(x)
        x = self._signed_sqrt(x)
        if self.is_vec:
            x = x.view(x.size(0), -1)
        x = self._l2norm(x)
        return x


@ARCH_REGISTRY.register()
class UNIQUE(nn.Module):
    """UNIQUE no-reference image quality model.

    Args:
        No runtime arguments. The model loads the default pretrained
        ``'mix'`` checkpoint.
    """

    def __init__(self):
        """Initialize the unique and configure its layers, parameters, and optional pretrained state.

        """
        super(UNIQUE, self).__init__()

        if ResNet34_Weights is not None:
            self.backbone = torchvision.models.resnet34(
                weights=ResNet34_Weights.IMAGENET1K_V1
            )
        else:
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
        r"""Predict quality score using UNIQUE.

        Args:
            x (torch.Tensor): Input tensor with shape ``(N, 3, H, W)``.

        Returns:
            torch.Tensor: Predicted mean quality score with shape ``(N,)``.

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
