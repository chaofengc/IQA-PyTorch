"""LIQE Model

github repo link: https://github.com/zwx8981/LIQE

Cite as:
@inproceedings{zhang2023liqe,
  title={Blind Image Quality Assessment via Vision-Language Correspondence: A Multitask Learning Perspective},
  author={Zhang, Weixia and Zhai, Guangtao and Wei, Ying and Yang, Xiaokang and Ma, Kede},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={14071--14081},
  year={2023}
}

"""


import torch
import torch.nn as nn

from .constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

from pyiqa.utils.registry import ARCH_REGISTRY
from pyiqa.archs.arch_util import load_pretrained_network

import clip
from .clip_model import load
import torch.nn.functional as F
from itertools import product

qualitys = ['bad', 'poor', 'fair', 'good', 'perfect']
scenes = ['animal', 'cityscape', 'human', 'indoor', 'landscape', 'night', 'plant', 'still_life', 'others']
dists_map = ['jpeg2000 compression', 'jpeg compression', 'noise', 'blur', 'color', 'contrast', 'overexposure',
            'underexposure', 'spatial', 'quantization', 'other']

default_model_urls = {'koniq': 'https://github.com/zwx8981/IQA-PyTorch/releases/download/Weights/liqe_koniq.pt',
                      'mix': 'https://github.com/zwx8981/IQA-PyTorch/releases/download/Weights/liqe_mix.pt'}

@ARCH_REGISTRY.register()
class LIQE(nn.Module):
    def __init__(self,
                 model_type='liqe',
                 backbone = 'ViT-B/32',
                 step = 32,
                 num_patch = 15,
                 pretrained=True,
                 pretrained_model_path=None,
                 mtl = False,
                 ) -> None:
        super().__init__()
        assert backbone == 'ViT-B/32', 'Only support ViT-B/32 now'
        self.backbone = backbone
        self.clip_model = load(self.backbone, 'cpu')  # avoid saving clip weights
        self.model_type = model_type

        self.default_mean = torch.Tensor(OPENAI_CLIP_MEAN).view(1, 3, 1, 1)
        self.default_std = torch.Tensor(OPENAI_CLIP_STD).view(1, 3, 1, 1)

        self.clip_model.logit_scale.requires_grad = False

        self.step = step
        self.num_patch = num_patch

        if pretrained_model_path is None and pretrained:
            url_key = 'koniq' if isinstance(pretrained, bool) else pretrained
            pretrained_model_path = default_model_urls[url_key]
        if pretrained_model_path is not None:
            load_pretrained_network(self, pretrained_model_path, True, 'params')

        if pretrained == 'mix':
            self.mtl = True
        else:
            self.mtl = mtl

        if self.mtl:
            self.joint_texts = torch.cat(
                [clip.tokenize(f"a photo of a {c} with {d} artifacts, which is of {q} quality") for q, c, d
                 in product(qualitys, scenes, dists_map)])
        else:
            self.joint_texts = torch.cat([clip.tokenize(f"a photo with {c} quality") for c in qualitys])

    def forward(self, x):
        bs = x.size(0)
        h = x.size(2)
        w = x.size(3)

        assert (h >= 224) & (w >= 224), 'Short side is less than 224, try upsampling the original image'
        # preprocess image
        x = (x - self.default_mean.to(x)) / self.default_std.to(x)

        x = x.unfold(2, 224, self.step).unfold(3, 224, self.step).permute(2, 3, 0, 1, 4, 5).reshape(bs, -1, 3,
                                                                                                    224, 224)

        if x.size(1) < self.num_patch:
            num_patch = x.size(1)
        else:
            num_patch = self.num_patch

        if self.training:
            sel = torch.randint(low=0, high=x.size(0), size=(num_patch, ))
        else:
            sel_step = x.size(1) // self.num_patch
            sel = torch.zeros(num_patch)
            for i in range(num_patch):
                sel[i] = sel_step * i
            sel = sel.long()

        x = x[:, sel, ...]

        text_features = self.clip_model.encode_text(self.joint_texts.to(x.device))
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        x = x.view(bs*x.size(1), x.size(2), x.size(3), x.size(4))
        image_features = self.clip_model.encode_image(x, pos_embedding=True)
        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        # cosine similarity as logits
        logit_scale = self.clip_model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()

        logits_per_image = logits_per_image.view(bs, self.num_patch, -1)
        logits_per_image = logits_per_image.mean(1)
        logits_per_image = F.softmax(logits_per_image, dim=1)

        if self.mtl:
            logits_per_image = logits_per_image.view(-1, len(qualitys), len(scenes), len(dists_map))
            logits_quality = logits_per_image.sum(3).sum(2)
        else:
            logits_per_image = logits_per_image.view(-1, len(qualitys))
            logits_quality = logits_per_image

        quality = 1 * logits_quality[:, 0] + 2 * logits_quality[:, 1] + 3 * logits_quality[:, 2] + \
                             4 * logits_quality[:, 3] + 5 * logits_quality[:, 4]

        return quality