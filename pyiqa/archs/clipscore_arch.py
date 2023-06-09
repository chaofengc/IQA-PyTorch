r"""CLIPScore for no reference image caption matching.

Reference:
    @inproceedings{hessel2021clipscore,
    title={{CLIPScore:} A Reference-free Evaluation Metric for Image Captioning},
    author={Hessel, Jack and Holtzman, Ari and Forbes, Maxwell and Bras, Ronan Le and Choi, Yejin},
    booktitle={EMNLP},
    year={2021}
    }

Reference code: https://github.com/jmhessel/clipscore
"""
import torch
import torch.nn as nn
import torchvision.transforms as T

import clip
from .constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from pyiqa.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class CLIPScore(nn.Module):
    def __init__(self,
                 backbone='ViT-B/32',
                 ) -> None:
        super().__init__()

        self.clip_model, _ = clip.load(backbone)
        self.prefix = 'A photo depicts'
        self.w = 2.5
    
    def preprocess(self, x):
        # Bicubic interpolation
        x = T.functional.resize(x, self.clip_model.visual.input_resolution, interpolation=T.InterpolationMode.BICUBIC)
        # Center crop
        x = T.functional.center_crop(x, self.clip_model.visual.input_resolution)
        x = T.functional.normalize(x, OPENAI_CLIP_MEAN, OPENAI_CLIP_STD)
        return x
    
    def forward(self, img, caption_list=None):
        assert caption_list is not None, f'caption_list is None'
        text = clip.tokenize([self.prefix + ' ' + caption for caption in caption_list]).to(img.device)

        img_features = self.clip_model.encode_image(self.preprocess(img))
        text_features = self.clip_model.encode_text(text)
        img_features /= img_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        score = self.w * torch.relu((img_features * text_features).sum(dim=-1))
        return score


