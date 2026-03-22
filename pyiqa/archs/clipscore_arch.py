r"""CLIPScore for no reference image caption matching.

Reference:
    @inproceedings{hessel2021clipscore,
    title={{CLIPScore:} A Reference-free Evaluation Metric for Image Captioning},
    author={Hessel, Jack and Holtzman, Ari and Forbes, Maxwell and Bras, Ronan Le and Choi, Yejin},
    booktitle={EMNLP},
    year={2021}
    }

Reference url: https://github.com/jmhessel/clipscore
Re-implemented by: Chaofeng Chen (https://github.com/chaofengc)
"""

import torch
import torch.nn as nn

import clip
from pyiqa.utils.registry import ARCH_REGISTRY
from .arch_util import clip_preprocess_tensor


@ARCH_REGISTRY.register()
class CLIPScore(nn.Module):
    """Compute CLIPScore between an image and one or more captions.

    The implementation follows the original CLIPScore formulation and returns a
    non-negative image-text similarity score:

    .. math::

        s = w \cdot \max(\cos(f_{img}, f_{txt}), 0)

    Args:
        backbone (str): CLIP backbone name accepted by :mod:`clip`, for example
            ``"ViT-B/32"``.
        w (float): Multiplicative scaling factor applied to cosine similarity.
        prefix (str): Text prefix prepended to each caption before tokenization.

    Example:
        >>> metric = CLIPScore(backbone='ViT-B/32')
        >>> img = torch.rand(2, 3, 224, 224)
        >>> score = metric(img, ['a dog on grass', 'a city street'])
        >>> score.shape
        torch.Size([2])
    """

    def __init__(self, backbone='ViT-B/32', w=2.5, prefix='A photo depicts') -> None:
        super().__init__()

        self.clip_model, _ = clip.load(backbone)
        self.prefix = prefix
        self.w = w

    def forward(self, img, caption_list=None):
        """Compute CLIPScore for each image-caption pair.

        Args:
            img (torch.Tensor): Input tensor with shape ``(N, 3, H, W)``.
            caption_list (list[str] | None): List of length ``N`` containing
                captions paired with each image.

        Returns:
            torch.Tensor: Score tensor with shape ``(N,)``.

        Raises:
            AssertionError: If ``caption_list`` is not provided.
        """
        assert caption_list is not None, 'caption_list is None'
        text = clip.tokenize(
            [self.prefix + ' ' + caption for caption in caption_list], truncate=True
        ).to(img.device)

        img_features = self.clip_model.encode_image(
            clip_preprocess_tensor(img, self.clip_model)
        )
        text_features = self.clip_model.encode_text(text)
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        score = self.w * torch.relu((img_features * text_features).sum(dim=-1))
        return score
