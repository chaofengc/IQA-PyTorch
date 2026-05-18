r"""FGResQ metric implementation.

Reference:
    Sheng, X., Pan, X., Yang, Z., Chen, P., and Li, L.
    Fine-grained Image Quality Assessment for Perceptual Image Restoration.
    AAAI 2026.

Reference URL:
    https://github.com/sxfly99/FGResQ
"""

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel
from torchvision.transforms import functional as TF

from pyiqa.archs.arch_util import clean_state_dict, get_url_from_name
from pyiqa.utils.download_util import load_file_from_url
from pyiqa.utils.registry import ARCH_REGISTRY

from .constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

default_model_urls = {
    'fgresq': get_url_from_name('FGResQ.pth'),
    'degradation': get_url_from_name('FGResQ_Degradation.pth'),
}


def load_checkpoint(model_path):
    """Load and normalize a checkpoint state dict.

    Args:
        model_path (str): Path to the checkpoint file.

    Returns:
        dict: Cleaned state dict.
    """
    checkpoint = torch.load(
        model_path,
        map_location=torch.device('cpu'),
        weights_only=False,
    )

    if isinstance(checkpoint, dict):
        if 'model' in checkpoint and isinstance(checkpoint['model'], dict):
            checkpoint = checkpoint['model']
        elif 'state_dict' in checkpoint and isinstance(checkpoint['state_dict'], dict):
            checkpoint = checkpoint['state_dict']

    if not isinstance(checkpoint, dict):
        raise TypeError('Checkpoint does not contain a valid state dict.')

    return clean_state_dict(checkpoint)


def get_pooler_output(model, x):
    """Extract pooled CLIP visual features.

    Args:
        model (CLIPVisionModel): CLIP vision backbone.
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Pooled feature tensor.
    """
    outputs = model(pixel_values=x)
    if hasattr(outputs, 'pooler_output'):
        return outputs.pooler_output
    return outputs['pooler_output']


@ARCH_REGISTRY.register()
class FGResQ(nn.Module):
    """FGResQ no-reference image quality model.

    Args:
        clip_model (str): HuggingFace CLIP vision backbone id.
        task_clip_model (str): CLIP backbone for the degradation-aware branch.
        clip_freeze (bool): Whether to freeze the main CLIP backbone.
        pretrained (bool): Whether to load official pretrained weights.
        pretrained_model_path (str | None): Optional local checkpoint path for
            the main FGResQ weights.
        degradation_model_path (str | None): Optional local checkpoint path for
            the degradation branch weights.
        input_size (int): Final center crop size.
        resize_size (int): Resize size before center crop.
        The network returns a single quality score when only ``x0`` is given,
        and returns ``quality0``, ``quality1``, ``rank``, and ``rank_prob``
        when both ``x0`` and ``x1`` are provided.
        default_mean (tuple[float, float, float]): Input normalization mean.
        default_std (tuple[float, float, float]): Input normalization std.
        score_scale (float): Scale factor used before the sigmoid output head.
    """

    def __init__(
        self,
        clip_model='openai/clip-vit-base-patch16',
        task_clip_model='openai/clip-vit-base-patch16',
        clip_freeze=True,
        pretrained=True,
        pretrained_model_path=None,
        degradation_model_path=None,
        input_size=224,
        resize_size=256,
        default_mean=OPENAI_CLIP_MEAN,
        default_std=OPENAI_CLIP_STD,
        score_scale=0.3,
    ):
        super().__init__()

        self.input_size = input_size
        self.resize_size = resize_size
        self.score_scale = score_scale

        self.clip_model = CLIPVisionModel.from_pretrained(clip_model)
        self.task_cls_clip = CLIPVisionModel.from_pretrained(task_clip_model)

        hidden_size = self.clip_model.config.hidden_size
        task_hidden_size = self.task_cls_clip.config.hidden_size
        if hidden_size != task_hidden_size:
            raise ValueError(
                'FGResQ requires matching CLIP hidden sizes for both branches, '
                f'but got {hidden_size} and {task_hidden_size}.'
            )

        if clip_freeze:
            for param in self.clip_model.parameters():
                param.requires_grad = False

        self.head = nn.Linear(hidden_size * 3, 1)
        self.compare_head = nn.Linear(hidden_size * 6, 3)

        self.prompt = nn.Parameter(torch.rand(1, hidden_size))
        self.task_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(False),
            nn.Linear(hidden_size, hidden_size),
        )
        self.prompt_mlp = nn.Linear(hidden_size, hidden_size)

        with torch.no_grad():
            self.task_mlp[0].weight.zero_()
            self.task_mlp[0].bias.zero_()
            self.task_mlp[2].weight.zero_()
            self.task_mlp[2].bias.zero_()
            self.prompt_mlp.weight.zero_()
            self.prompt_mlp.bias.zero_()

        for param in self.task_cls_clip.parameters():
            param.requires_grad = False
        for layer in self.task_cls_clip.vision_model.encoder.layers[-2:]:
            for param in layer.parameters():
                param.requires_grad = True

        self.default_mean = torch.Tensor(default_mean).view(1, 3, 1, 1)
        self.default_std = torch.Tensor(default_std).view(1, 3, 1, 1)

        if degradation_model_path is not None:
            self.load_degradation_weights(degradation_model_path)
        elif pretrained:
            self.load_degradation_weights(
                load_file_from_url(default_model_urls['degradation'])
            )

        if pretrained_model_path is not None:
            self.load_pretrained_weights(pretrained_model_path)
        elif pretrained:
            self.load_pretrained_weights(load_file_from_url(default_model_urls['fgresq']))

    def load_degradation_weights(self, model_path):
        """Load degradation-branch weights.

        Args:
            model_path (str): Path to the degradation checkpoint.
        """
        state_dict = load_checkpoint(model_path)
        clip_state_dict = {
            key.replace('clip_model.', '', 1): value
            for key, value in state_dict.items()
            if key.startswith('clip_model.')
        }
        if not clip_state_dict:
            clip_state_dict = state_dict

        missing, unexpected = self.task_cls_clip.load_state_dict(
            clip_state_dict,
            strict=False,
        )
        if missing or unexpected:
            warnings.warn(
                f'FGResQ degradation weights loaded with missing={missing}, '
                f'unexpected={unexpected}',
                RuntimeWarning,
            )

    def load_pretrained_weights(self, model_path):
        """Load main FGResQ weights.

        Args:
            model_path (str): Path to the FGResQ checkpoint.
        """
        state_dict = load_checkpoint(model_path)
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            warnings.warn(
                f'FGResQ weights loaded with missing={missing}, unexpected={unexpected}',
                RuntimeWarning,
            )

    def preprocess(self, x):
        """Preprocess an input image tensor.

        Args:
            x (torch.Tensor): Input tensor with shape ``(N, C, H, W)``.

        Returns:
            torch.Tensor: Preprocessed tensor.
        """
        if x.dim() != 4:
            raise ValueError(
                f'FGResQ expects a 4D tensor, but got shape {tuple(x.shape)}.'
            )
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] != 3:
            raise ValueError(
                f'FGResQ expects 1 or 3 channels, but got {x.shape[1]}.'
            )

        x = F.interpolate(
            x,
            size=(self.resize_size, self.resize_size),
            mode='bilinear',
            align_corners=False,
        )
        x = TF.center_crop(x, self.input_size)
        x = (x - self.default_mean.to(x)) / self.default_std.to(x)
        return x

    def get_quality_features(self, x):
        """Extract FGResQ quality features.

        Args:
            x (torch.Tensor): Preprocessed image tensor.

        Returns:
            torch.Tensor: Concatenated quality features.
        """
        features = get_pooler_output(self.clip_model, x)
        task_features = get_pooler_output(self.task_cls_clip, x)

        task_embedding = torch.softmax(self.task_mlp(task_features), dim=1) * self.prompt
        task_embedding = self.prompt_mlp(task_embedding)
        return torch.cat(
            [features, task_embedding, features + task_embedding],
            dim=1,
        )

    def forward_single(self, x):
        """Predict single-image quality.

        Args:
            x (torch.Tensor): Preprocessed image tensor.

        Returns:
            torch.Tensor: Predicted quality score.
        """
        features = self.get_quality_features(x)
        return torch.sigmoid(self.head(features) * self.score_scale)

    def forward_pair(self, x0, x1):
        """Predict pairwise quality and comparison logits.

        Args:
            x0 (torch.Tensor): First preprocessed image tensor.
            x1 (torch.Tensor): Second preprocessed image tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Single-image scores
            for both inputs and pairwise comparison logits.
        """
        features0 = self.get_quality_features(x0)
        features1 = self.get_quality_features(x1)

        quality0 = torch.sigmoid(self.head(features0) * self.score_scale)
        quality1 = torch.sigmoid(self.head(features1) * self.score_scale)
        compare_logits = self.compare_head(torch.cat([features0, features1], dim=1))
        return quality0, quality1, compare_logits

    def get_pair_rank(self, compare_logits):
        """Convert comparison logits to a discrete rank label.

        Args:
            compare_logits (torch.Tensor): Pairwise comparison logits.

        Returns:
            torch.Tensor: Rank tensor with shape ``(N, 1)`` where
            ``0=image2_better``, ``1=image1_better``, and
            ``2=similar_quality``.
        """
        compare_probs = torch.softmax(compare_logits, dim=-1)
        return compare_probs.argmax(dim=-1, keepdim=True)

    def get_pair_result(self, quality0, quality1, compare_logits):
        """Format pairwise prediction outputs.

        Args:
            quality0 (torch.Tensor): Quality of the first image.
            quality1 (torch.Tensor): Quality of the second image.
            compare_logits (torch.Tensor): Pairwise comparison logits.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                ``quality0``, ``quality1``, ``rank``, and ``rank_prob``.
        """
        compare_probs = torch.softmax(compare_logits, dim=-1)
        rank = self.get_pair_rank(compare_logits)
        rank_prob = compare_probs.gather(dim=-1, index=rank)
        return quality0, quality1, rank, rank_prob

    def forward(self, x0, x1=None):
        """Forward pass for FGResQ.

        Args:
            x0 (torch.Tensor): First input tensor.
            x1 (torch.Tensor | None): Optional second input tensor.

        Returns:
            torch.Tensor | tuple: Output depends on whether ``x1`` is given.
        """
        x0 = self.preprocess(x0)

        if x1 is None:
            return self.forward_single(x0)

        x1 = self.preprocess(x1)
        quality0, quality1, compare_logits = self.forward_pair(x0, x1)
        return self.get_pair_result(quality0, quality1, compare_logits)
