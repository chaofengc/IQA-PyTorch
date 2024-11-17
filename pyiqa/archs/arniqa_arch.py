r"""ARNIQA: Learning Distortion Manifold for Image Quality Assessment

@inproceedings{agnolucci2024arniqa,
  title={ARNIQA: Learning Distortion Manifold for Image Quality Assessment},
  author={Agnolucci, Lorenzo and Galteri, Leonardo and Bertini, Marco and Del Bimbo, Alberto},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={189--198},
  year={2024}
}

Reference:
    - Arxiv link: https://www.arxiv.org/abs/2310.14918
    - Official Github: https://github.com/miccunifi/ARNIQA
"""

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models
from typing import Tuple
import warnings
from collections import OrderedDict

from .constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from pyiqa.utils.registry import ARCH_REGISTRY
from pyiqa.archs.arch_util import get_url_from_name
from pyiqa.api_helpers import get_dataset_info

# Avoid warning related to loading a jit model from torch.hub
warnings.filterwarnings("ignore", category=UserWarning, module="torch.serialization")

DATASET_INFO = get_dataset_info()
DATASET_INFO["clive"] = DATASET_INFO["livec"]
DATASET_INFO["tid"] = DATASET_INFO["tid2013"]
DATASET_INFO["koniq"] = DATASET_INFO["koniq10k"]
DATASET_INFO["kadid"] = DATASET_INFO["kadid10k"]

default_model_urls = {
    "ARNIQA": get_url_from_name(name="ARNIQA.pth"),
    "live": get_url_from_name(name="regressor_live.pth"),
    "csiq": get_url_from_name(name="regressor_csiq.pth"),
    "tid": get_url_from_name(name="regressor_tid2013.pth"),
    "kadid": get_url_from_name(name="regressor_kadid10k.pth"),
    "koniq": get_url_from_name(name="regressor_koniq10k.pth"),
    "clive": get_url_from_name(name="regressor_clive.pth"),
    "flive": get_url_from_name(name="regressor_flive.pth"),
    "spaq": get_url_from_name(name="regressor_spaq.pth")
}


@ARCH_REGISTRY.register()
class ARNIQA(nn.Module):
    """
    ARNIQA model implementation.

    This class implements the ARNIQA model for image quality assessment, which combines a ResNet50 encoder
    with a regressor network for predicting image quality scores.

    Args:
        regressor_dataset (str, optional): The dataset to use for the regressor. Default is "koniq".

    Attributes:
        regressor_dataset (str): The dataset to use for the regressor.
        encoder (nn.Module): The ResNet50 encoder.
        feat_dim (int): The feature dimension of the encoder.
        regressor (nn.Module): The regressor network.
        default_mean (torch.Tensor): The mean values for normalization.
        default_std (torch.Tensor): The standard deviation values for normalization.
    """
    def __init__(self, regressor_dataset: str = "koniq"):
        super().__init__()

        self.regressor_dataset = regressor_dataset

        self.encoder = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1
        )  # V1 weights work better than V2
        self.feat_dim = self.encoder.fc.in_features
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])

        encoder_state_dict = torch.hub.load_state_dict_from_url(
            default_model_urls["ARNIQA"], progress=True, map_location="cpu"
        )
        cleaned_encoder_state_dict = OrderedDict()
        for key, value in encoder_state_dict.items():
            # Remove the prefix
            if key.startswith("model."):
                new_key = key[6:]
                cleaned_encoder_state_dict[new_key] = value

        self.encoder.load_state_dict(cleaned_encoder_state_dict)
        self.encoder.eval()

        self.regressor: nn.Module = torch.hub.load_state_dict_from_url(
            default_model_urls[self.regressor_dataset], progress=True, map_location="cpu"
        )  # Load regressor from torch.hub as JIT model
        self.regressor.eval()

        self.default_mean = torch.Tensor(IMAGENET_DEFAULT_MEAN).view(1, 3, 1, 1)
        self.default_std = torch.Tensor(IMAGENET_DEFAULT_STD).view(1, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> float:
        """
        Forward pass of the ARNIQA model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            float: The predicted quality score.
        """
        x, x_ds = self._preprocess(x)

        f = F.normalize(self.encoder(x), dim=1)
        f_ds = F.normalize(self.encoder(x_ds), dim=1)
        f_combined = torch.hstack((f, f_ds)).view(-1, self.feat_dim * 2)

        score = self.regressor(f_combined)
        score = self._scale_score(score)

        return score

    def _preprocess(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Downsample the input image with a factor of 2 and normalize the original and downsampled images.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The normalized original and downsampled tensors.
        """
        x_ds = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)
        x = (x - self.default_mean.to(x)) / self.default_std.to(x)
        x_ds = (x_ds - self.default_mean.to(x_ds)) / self.default_std.to(x_ds)
        return x, x_ds

    def _scale_score(self, score: torch.Tensor) -> torch.Tensor:
        """
        Scale the score in the range [0, 1], where higher is better.

        Args:
            score (torch.Tensor): The predicted score.

        Returns:
            torch.Tensor: The scaled score.
        """
        new_range = (0., 1.)

        # Compute scaling factors
        original_range = (
            DATASET_INFO[self.regressor_dataset]["mos_range"][0], 
            DATASET_INFO[self.regressor_dataset]["mos_range"][1], 
        )
        original_width = original_range[1] - original_range[0]
        new_width = new_range[1] - new_range[0]
        scaling_factor = new_width / original_width

        # Scale score
        scaled_score = new_range[0] + (score - original_range[0]) * scaling_factor

        # Invert the scale if needed
        if DATASET_INFO[self.regressor_dataset]["lower_better"]:
            scaled_score = new_range[1] - scaled_score

        return scaled_score
