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
from torchvision.transforms import Normalize
import torchvision

from pyiqa.utils.registry import ARCH_REGISTRY
from pyiqa.archs.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from pyiqa.archs.clip_model import load


class CustomCLIP(nn.Module):
    """Thin wrapper around CLIP image/text encoders used by MACLIP.

    Args:
        backbone (str): CLIP backbone identifier.
        device (str): Device string used when initializing the model.
    """
    def __init__(self, backbone: str, device="cpu"):
        super().__init__()

        self.clip_model = load(backbone, device)
        self.encode_image = self.clip_model.encode_image
        self.encode_text = self.clip_model.encode_text
        self.logit_scale = self.clip_model.logit_scale

    def forward(self, image, text, pos_embedding=False, text_features=None):
        """Encode image/text and return logits and unnormalized image features.

        Args:
            image (torch.Tensor): Image tensor with shape ``(N, 3, H, W)``.
            text (torch.Tensor): Tokenized text tensor.
            pos_embedding (bool): Whether to enable positional embedding branch
                in the custom CLIP visual encoder.
            text_features (torch.Tensor | None): Optional precomputed text
                features.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                ``(logits_per_image, logits_per_text, image_features_org)``.
        """
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
    """Magnitude-Aware CLIP for no-reference image quality assessment.

    Args:
        model_type (str): Output type identifier.
        backbone (str): CLIP backbone name.
        pos_embedding (bool): Whether to enable visual positional embedding in
            CLIP image encoding.

    Notes:
        The current implementation runs on CUDA and is intended for inference.
    """

    def __init__(self,
                 model_type='clipiqa',backbone='RN50',pos_embedding=False) -> None:
        """Initialize MACLIP model."""
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
        """Normalize image and build overlapping 224x224 patch set.

        Args:
            img (torch.Tensor): Input tensor with shape ``(1, 3, H, W)``.

        Returns:
            torch.Tensor: Patch tensor with shape ``(P, 3, 224, 224)``.
        """
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
        """Apply Box-Cox-like transform after per-sample standardization."""
        x = (x) / (x.std(dim=1, keepdim=True) + epsilon)  # [B, D]
        if lam == 0:
            transformed = torch.log(x+1)
        else:
            transformed = ((x + 1) ** lam - 1) / lam

        return transformed

    def fusion(self, cos, norm, base_cos=1.0, base_norm=0.6, alpha=1.0):
        """Fuse cosine and magnitude cues with adaptive softmax weighting.

        Args:
            cos (torch.Tensor): Cosine-similarity based quality scores.
            norm (torch.Tensor): Magnitude-cue scores.
            base_cos (float): Base weight prior for cosine cue.
            base_norm (float): Base weight prior for magnitude cue.
            alpha (float): Adaptive weight adjustment factor.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                Fused score, cosine weight, and magnitude weight.
        """
        d = cos - norm 
        cos_param = base_cos + alpha * d
        norm_param = base_norm - alpha * d
        weights = F.softmax(torch.stack([cos_param, norm_param], dim=-1), dim=-1)  
        w_cos, w_norm = weights.unbind(dim=-1) 
        weighted_metric = w_cos * cos + w_norm * norm
        return weighted_metric, w_cos, w_norm

    def forward(self, x, box_lam=0.5, base_cos=1.0, base_norm=0.6, alpha=1.0):
        """Compute MACLIP score.

        Args:
            x (torch.Tensor): Input image tensor with shape ``(1, 3, H, W)``.
            box_lam (float): Lambda for Box-Cox transform.
            base_cos (float): Base weight for cosine cue.
            base_norm (float): Base weight for magnitude cue.
            alpha (float): Adaptive fusion factor.

        Returns:
            torch.Tensor: Scalar quality score.
        """
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
    