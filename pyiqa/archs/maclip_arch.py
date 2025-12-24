r"""Beyond Cosine Similarity: Magnitude-Aware CLIP for No-Reference Image Quality Assessment

@article{liao2025beyond,
  title={Beyond Cosine Similarity Magnitude-Aware CLIP for No-Reference Image Quality Assessment},
  author={Liao, Zhicheng and Wu, Dongxu and Shi, Zhenshan and Mai, Sijie and Zhu, Hanwei and Zhu, Lingyu and Jiang, Yuncheng and Chen, Baoliang},
  journal={arXiv preprint arXiv:2511.09948},
  year={2025}
}

Accepted by AAAI 2026.

Reference:
    - Arxiv link: https://arxiv.org/abs/2511.09948
    - Official Github: https://github.com/zhix000/MA-CLIP
"""

import torch
import torch.nn as nn
import clip
import torch.nn.functional as F
import torch.nn as nn
from torchvision.transforms import ToTensor, Normalize
from PIL import Image
import torchvision
import torch.nn.functional as F
import torch

from pyiqa.utils.registry import ARCH_REGISTRY
from pyiqa.archs.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from pyiqa.archs.clip_model import load


class CustomCLIP(nn.Module):
    def __init__(self, backbone: str, device="cpu"):
        super().__init__()

        self.clip_model = load(backbone, device)
        self.encode_image = self.clip_model.encode_image
        self.encode_text = self.clip_model.encode_text
        self.logit_scale = self.clip_model.logit_scale

    def forward(self, image, text, pos_embedding=False, text_features=None):
        image_features_org = self.encode_image(image, pos_embedding)
        if text_features is None:
            text_features = self.encode_text(text)

        # L2 normalize
        image_features_nrm = image_features_org.norm(dim=-1, keepdim=True)
        image_features = image_features_org / image_features_nrm
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # cosine similarity
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text, image_features_org


@ARCH_REGISTRY.register()
class MACLIP(nn.Module):
    def __init__(self,
                 model_type='clipiqa',backbone='RN50',pos_embedding=False) -> None:
        '''
        Args:
            backbone: CLIP backbone model (default: `RN50`, optional: `ViT-B/32`, `RN101` etc., from `clip_model.py`).
        '''
        super().__init__()

        self.clip_model = CustomCLIP(backbone=backbone, device='cuda')
        self.prompt_pairs = clip.tokenize([
            'Good image', 'bad image',
            'Sharp image', 'blurry image',
            'sharp edges', 'blurry edges',
            'High resolution image', 'low resolution image',
            'Noise-free image', 'noisy image',
        ])

        self.default_mean = torch.Tensor(OPENAI_CLIP_MEAN).view(1, 3, 1, 1)
        self.default_std = torch.Tensor(OPENAI_CLIP_STD).view(1, 3, 1, 1)

        self.model_type = model_type
        self.pos_embedding = pos_embedding
   
        for p in self.clip_model.parameters():
            p.requires_grad = False
 
    def preprocess(self, img):
        transforms = torchvision.transforms.Compose([
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
        raw_image = transforms(img)
        unfold = nn.Unfold(kernel_size=(224, 224), stride=128)
        img = unfold(raw_image).view(1, 3, 224, 224, -1)[0]
        img = img.permute(3,0,1,2).cuda()
        img_s = F.interpolate(raw_image, size=(224, 224), mode='bilinear', align_corners=False).to('cuda')
        img = torch.cat([img, img_s], dim=0)              
        return img

    def box_cox(self, x, lam=0.5, epsilon=1e-6):
        x = (x) / (x.std(dim=1, keepdim=True) + epsilon)  # [B, D]
        if lam == 0:
            transformed = torch.log(x+1)
        else:
            transformed = ((x + 1) ** lam - 1) / lam

        return transformed

    def fusion(self, cos, norm, base_cos=1.0, base_norm=0.6, alpha=1.0):
        '''
        Args:
            box_lam: Lambda parameter for Box-Cox transformation (default: 0.5)
            base_cos/base_norm: Base weights for fusion of cosine similarity and magnitude cues (default: 1.0/0.6).
            alpha: Fusion coefficient (default: 1.0)
        '''
        d = cos - norm 
        cos_param = base_cos + alpha * d
        norm_param = base_norm - alpha * d
        weights = F.softmax(torch.stack([cos_param, norm_param], dim=-1), dim=-1)  
        w_cos, w_norm = weights.unbind(dim=-1) 
        weighted_metric = w_cos * cos + w_norm * norm
        return weighted_metric, w_cos, w_norm

    def forward(self, x, box_lam=0.5, base_cos=1.0, base_norm=0.6, alpha=1.0):
        x = self.preprocess(x)
        clip_model = self.clip_model.to(x.device)
        prompts = self.prompt_pairs.to(x.device)
        logits_per_image, logits_per_text, image_features_org = clip_model(x, prompts, pos_embedding=self.pos_embedding)
        probs = logits_per_image.reshape(logits_per_image.shape[0], -1, 2).softmax(dim=-1)
        clipiqa = probs[..., 0].mean(dim=1, keepdim=True)

        # Magnitude cue computation
        image_features_org_abs = torch.abs(image_features_org)
        image_features_org_abs_box = self.box_cox(image_features_org_abs, lam=box_lam)
        nrm_score2 = image_features_org_abs_box.mean(dim=-1)
        # Fusion
        comb, w1, w2 = self.fusion(clipiqa.squeeze(1), nrm_score2, base_cos, base_norm, alpha)
        comb = torch.mean(comb)
        
        return comb
    