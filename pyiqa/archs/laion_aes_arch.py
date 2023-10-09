r"""LAION-Aesthetics Predictor

Introduced by: https://github.com/christophschuhmann/improved-aesthetic-predictor
"""
import torch
import torch.nn as nn

import clip
from pyiqa.utils.registry import ARCH_REGISTRY
from pyiqa.archs.arch_util import load_pretrained_network, clip_preprocess_tensor


default_model_urls = {
    "url": "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac%2Blogos%2Bava1-l14-linearMSE.pth"
}


class MLP(nn.Module):
    def __init__(self, input_size, xcol="emb", ycol="avg_rating"):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            # nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            # nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


@ARCH_REGISTRY.register()
class LAIONAes(nn.Module):
    """
    LAIONAes is a class that implements a neural network architecture for image quality assessment.

    The architecture is based on the ViT-L/14 model from the OpenAI CLIP library, and uses an MLP to predict image quality scores.

    Args:
        None

    Returns:
        A tensor representing the predicted image quality scores.
    """
    def __init__(self) -> None:
        super().__init__()

        self.clip_model, _ = clip.load("ViT-L/14")
        self.mlp = MLP(self.clip_model.visual.output_dim)
        load_pretrained_network(self.mlp, default_model_urls["url"])

    def forward(self, x):
        img = clip_preprocess_tensor(x, self.clip_model)

        img_emb = self.clip_model.encode_image(img)

        img_emb = nn.functional.normalize(img_emb.float(), p=2, dim=-1)
        score = self.mlp(img_emb)
        return score
