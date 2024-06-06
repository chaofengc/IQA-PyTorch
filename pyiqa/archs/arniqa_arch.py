r"""ARNIQA: Learning Distortion Manifold for Image Quality Assessment

@inproceedings{agnolucci2024arniqa,
  title={ARNIQA: Learning Distortion Manifold for Image Quality Assessment},
  author={Agnolucci, Lorenzo and Galteri, Leonardo and Bertini, Marco and Del Bimbo, Alberto},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={189--198},
  year={2024}
}

Reference:
    - Arxiv link: https://arxiv.org/abs/2310.14918
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
from pyiqa.archs.arch_util import load_pretrained_network

# Avoid warning related to loading a jit model from torch.hub
warnings.filterwarnings("ignore", category=UserWarning, module="torch.serialization")

available_datasets_ranges = {
    "live": (1, 100),
    "csiq": (0, 1),
    "tid": (0, 9),
    "kadid": (1, 5),
    "koniq": (1, 100),
    "clive": (1, 100),
    "flive": (1, 100),
    "spaq": (1, 100)
}

available_datasets_mos_types = {
    "live": "dmos",
    "csiq": "dmos",
    "tid": "mos",
    "kadid": "mos",
    "koniq": "mos",
    "clive": "mos",
    "flive": "mos",
    "spaq": "mos"
}

base_url = "https://github.com/miccunifi/ARNIQA/releases/download/weights"

default_model_urls = {
    "ARNIQA": f"{base_url}/ARNIQA.pth",
    "live": f"{base_url}/regressor_live.pth",
    "csiq": f"{base_url}/regressor_csiq.pth",
    "tid": f"{base_url}/regressor_tid2013.pth",
    "kadid": f"{base_url}/regressor_kadid10k.pth",
    "koniq": f"{base_url}/regressor_koniq10k.pth",
    "clive": f"{base_url}/regressor_clive.pth",
    "flive": f"{base_url}/regressor_flive.pth",
    "spaq": f"{base_url}/regressor_spaq.pth"
}


@ARCH_REGISTRY.register()
class ARNIQA(nn.Module):
    def __init__(self,
                 regressor_dataset: str = "koniq",
                 ):
        super().__init__()

        self.regressor_dataset = regressor_dataset

        self.encoder = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)  # V1 weights work better than V2
        self.feat_dim = self.encoder.fc.in_features
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])

        encoder_state_dict = torch.hub.load_state_dict_from_url(default_model_urls["ARNIQA"], progress=True,
                                                               map_location="cpu")
        cleaned_encoder_state_dict = OrderedDict()
        for key, value in encoder_state_dict.items():
            # Remove the prefix
            if key.startswith("model."):
                new_key = key[6:]
                cleaned_encoder_state_dict[new_key] = value

        self.encoder.load_state_dict(cleaned_encoder_state_dict)
        self.encoder.eval()

        self.regressor: nn.Module = torch.hub.load_state_dict_from_url(default_model_urls[self.regressor_dataset],
                                                                       progress=True,
                                                                       map_location="cpu")  # Load regressor from torch.hub as JIT model
        self.regressor.eval()

        self.default_mean = torch.Tensor(IMAGENET_DEFAULT_MEAN).view(1, 3, 1, 1)
        self.default_std = torch.Tensor(IMAGENET_DEFAULT_STD).view(1, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> float:
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
        """
        x_ds = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)
        x = (x - self.default_mean.to(x)) / self.default_std.to(x)
        x_ds = (x_ds - self.default_mean.to(x_ds)) / self.default_std.to(x_ds)
        return x, x_ds

    def _scale_score(self, score: torch.Tensor) -> torch.Tensor:
        """
        Scale the score in the range [0, 1], where higher is better.
        """
        new_range = (0., 1.)

        # Compute scaling factors
        original_range = (
            available_datasets_ranges[self.regressor_dataset][0], available_datasets_ranges[self.regressor_dataset][1])
        original_width = original_range[1] - original_range[0]
        new_width = new_range[1] - new_range[0]
        scaling_factor = new_width / original_width

        # Scale score
        scaled_score = new_range[0] + (score - original_range[0]) * scaling_factor

        # Invert the scale if needed
        if available_datasets_mos_types[self.regressor_dataset] == "dmos":
            scaled_score = new_range[1] - scaled_score

        return scaled_score
