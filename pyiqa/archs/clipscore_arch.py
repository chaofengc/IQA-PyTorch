r"""CLIPScore for no reference image caption matching.

Reference:
    @inproceedings{hessel2021clipscore,
    title={{CLIPScore:} A Reference-free Evaluation Metric for Image Captioning},
    author={Hessel, Jack and Holtzman, Ari and Forbes, Maxwell and Bras, Ronan Le and Choi, Yejin},
    booktitle={EMNLP},
    year={2021}
    }

Reference url: https://github.com/jmhessel/clipscore
Re-implmented by: Chaofeng Chen (https://github.com/chaofengc) 
"""
import torch
import torch.nn as nn

import clip
from pyiqa.utils.registry import ARCH_REGISTRY
from .arch_util import clip_preprocess_tensor


@ARCH_REGISTRY.register()
class CLIPScore(nn.Module):
    """
    A PyTorch module for computing image-text similarity scores using the CLIP model.

    Args:
        - backbone (str): The name of the CLIP model backbone to use. Default is 'ViT-B/32'.

    Attributes:
        - clip_model (CLIP): The CLIP model used for computing image and text features.
        - prefix (str): The prefix to add to each caption when computing text features.
        - w (float): The weight to apply to the similarity score.

    Methods:
        - forward(img, caption_list): Computes the similarity score between the input image and a list of captions.
    """
    def __init__(self,
                 backbone='ViT-B/32',
                 w = 2.5,
                 prefix = 'A photo depicts'
                 ) -> None:
        super().__init__()

        self.clip_model, _ = clip.load(backbone)
        self.prefix = prefix
        self.w = w 
    
    def forward(self, img, caption_list=None):
        assert caption_list is not None, f'caption_list is None'
        text = clip.tokenize([self.prefix + ' ' + caption for caption in caption_list], truncate=True).to(img.device)

        img_features = self.clip_model.encode_image(clip_preprocess_tensor(img, self.clip_model))
        text_features = self.clip_model.encode_text(text)
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        score = self.w * torch.relu((img_features * text_features).sum(dim=-1))
        return score


