r"""JLD (Jacobian Lens Distance) metric introduced in

@article{saini2026jld,
  title={JLD: Perceptual Distance Through A Jacobian Lens},
  author={Saini, Shreshth and Adsumilli, Balu and Bovik, Alan C.},
  journal={arXiv preprint arXiv:2610.05967},
  year={2026},
  url={https://arxiv.org/abs/2610.05967}
}

Reference implementation: https://github.com/shreshthsaini/jld

The distance between a distorted image ``x`` and its reference ``y`` is

    JLD(x, y) = sqrt(mean_t ||U^T (h_t(x') - h_t(y'))||^2) + w * (1 - cos(c(x'), c(y')))

where ``x'`` and ``y'`` are the images after a chroma low-pass prefilter,
``h_t`` are the block-1 patch tokens of a frozen DINOv2-S/14 encoder, ``U`` is
a fixed 64-direction lens (the top eigenvectors of ``E[J^T J]`` with
``J = dc / dh_t``) fitted once on unlabeled images, and ``c`` is the final CLS
token. JLD uses ``w = 0.5``. JLD-fast sets ``w = 0`` and stops the encoder after
block 1. Scoring needs forward passes only; the lens is loaded, never refitted.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from pyiqa.utils.registry import ARCH_REGISTRY
from pyiqa.archs.arch_util import load_pretrained_network
from pyiqa.archs.arch_util import get_url_from_name

default_model_urls = {'url': get_url_from_name('JLD_dinov2_s14_block1_k64.pth')}

ENCODER_NAME = 'vit_small_patch14_dinov2.lvd142m'

# variant: (chroma prefilter sigma in pixels, CLS term weight)
VARIANTS = {
    'full': (2.0, 0.5),
    'fast': (2.0, 0.0),
}


def _gauss1d(sigma, device, dtype):
    radius = max(1, int(math.ceil(3 * sigma)))
    t = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    return (k / k.sum()).to(dtype)


def gaussian_blur(x, sigma):
    """Separable Gaussian blur with reflect padding.

    The kernel is truncated and renormalized when the image is smaller than
    the kernel radius, so reflect padding stays valid for tiny inputs.

    Args:
        x (torch.Tensor): Input of shape ``(B, C, H, W)``.
        sigma (float): Standard deviation in pixels. Values ``<= 0`` return
            ``x`` unchanged.

    Returns:
        torch.Tensor: Blurred tensor with the same shape as ``x``.
    """
    if sigma <= 0:
        return x
    k = _gauss1d(sigma, x.device, x.dtype)
    r = (k.numel() - 1) // 2
    c = x.shape[1]
    kh = k.view(1, 1, 1, -1).repeat(c, 1, 1, 1)
    kv = k.view(1, 1, -1, 1).repeat(c, 1, 1, 1)
    rw, rh = min(r, x.shape[-1] - 1), min(r, x.shape[-2] - 1)
    kh = kh[..., r - rw : r + rw + 1]
    kh = kh / kh.sum(dim=-1, keepdim=True)
    kv = kv[:, :, r - rh : r + rh + 1]
    kv = kv / kv.sum(dim=2, keepdim=True)
    y = F.conv2d(F.pad(x, (rw, rw, 0, 0), mode='reflect'), kh, groups=c)
    y = F.conv2d(F.pad(y, (0, 0, rh, rh), mode='reflect'), kv, groups=c)
    return y


def chroma_lowpass(x, sigma):
    """Blur the Cb and Cr channels of an RGB image; luminance is untouched.

    Args:
        x (torch.Tensor): RGB image in ``[0, 1]`` of shape ``(B, 3, H, W)``.
        sigma (float): Chroma blur standard deviation in pixels.

    Returns:
        torch.Tensor: Filtered RGB image clamped to ``[0, 1]``.
    """
    if sigma <= 0:
        return x
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    chroma = torch.cat([(b - y) * 0.564 + 0.5, (r - y) * 0.713 + 0.5], 1)
    chroma = gaussian_blur(chroma, sigma) - 0.5
    cb, cr = chroma[:, 0:1], chroma[:, 1:2]
    rgb = torch.cat([y + 1.403 * cr, y - 0.344 * cb - 0.714 * cr, y + 1.773 * cb], 1)
    return rgb.clamp(0, 1)


def center_crop_to_multiple(x, patch):
    """Center-crop ``(..., H, W)`` so that ``H`` and ``W`` are multiples of ``patch``."""
    h, w = x.shape[-2:]
    hc, wc = (h // patch) * patch, (w // patch) * patch
    if hc == 0 or wc == 0:
        raise ValueError(f'JLD needs images of at least {patch}x{patch}, got {h}x{w}.')
    top, left = (h - hc) // 2, (w - wc) // 2
    return x[..., top : top + hc, left : left + wc]


class JLDLens(nn.Module):
    """Container for the fitted lens ``U`` of shape ``(dim, k)``."""

    def __init__(self, dim=384, k=64):
        super().__init__()
        self.register_buffer('U', torch.zeros(dim, k))


@ARCH_REGISTRY.register()
class JLD(nn.Module):
    """Jacobian Lens Distance, a full-reference perceptual distance.

    Images are scored at native resolution after a center crop to a multiple of
    the 14-pixel patch size; no resizing is applied. Lower scores mean the
    distorted image is closer to the reference.

    Args:
        variant (str): ``'full'`` for JLD or ``'fast'`` for JLD-fast.
        pretrained (bool): Load the DINOv2 encoder and the released lens.
        pretrained_model_path (str, optional): Local path or URL of the lens
            checkpoint. Defaults to the released DINOv2-S/14 block-1 k=64 lens.
    """

    def __init__(self, variant='full', pretrained=True, pretrained_model_path=None):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(
                f'variant must be one of {sorted(VARIANTS)}, got {variant!r}.'
            )
        self.chroma_sigma, self.cls_weight = VARIANTS[variant]

        self.encoder = timm.create_model(
            ENCODER_NAME, pretrained=pretrained, dynamic_img_size=True
        )
        self.patch = int(self.encoder.patch_embed.patch_size[0])
        self.n_prefix = int(self.encoder.num_prefix_tokens)
        cfg = self.encoder.pretrained_cfg
        self.register_buffer(
            'mean', torch.tensor(cfg['mean']).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            'std', torch.tensor(cfg['std']).view(1, 3, 1, 1), persistent=False
        )
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        self.lens = JLDLens(dim=self.encoder.embed_dim)
        if pretrained_model_path is not None:
            load_pretrained_network(self.lens, pretrained_model_path)
        elif pretrained:
            load_pretrained_network(self.lens, default_model_urls['url'])
        self.eval()

    def train(self, mode=True):
        # The encoder is frozen and always runs in eval mode, also when JLD is
        # used as a loss inside a training loop.
        super().train(mode)
        self.encoder.eval()
        return self

    def _encode(self, x):
        """Return block-1 patch tokens ``(B, N, dim)`` and the final CLS token or ``None``."""
        if self.chroma_sigma > 0:
            x = chroma_lowpass(x, self.chroma_sigma)
        model = self.encoder
        hidden = model.patch_embed((x - self.mean) / self.std)
        hidden = model._pos_embed(hidden)
        hidden = model.patch_drop(hidden)
        hidden = model.norm_pre(hidden)
        hidden = model.blocks[0](hidden)
        tokens = hidden[:, self.n_prefix :].float()
        if self.cls_weight <= 0:
            return tokens, None
        for block in model.blocks[1:]:
            hidden = block(hidden)
        cls = model.norm(hidden)[:, 0].float()
        return tokens, cls

    def forward(self, x, y):
        """Compute JLD between distorted and reference images.

        Args:
            x (torch.Tensor): Distorted RGB images in ``[0, 1]``, ``(B, 3, H, W)``.
            y (torch.Tensor): Reference RGB images in ``[0, 1]``, ``(B, 3, H, W)``.

        Returns:
            torch.Tensor: One distance per image pair, shape ``(B,)``.
        """
        if x.ndim != 4 or y.ndim != 4 or x.shape[1] != 3 or y.shape[1] != 3:
            raise ValueError('JLD requires RGB tensors with shape (B, 3, H, W).')
        x = center_crop_to_multiple(x, self.patch).float()
        y = center_crop_to_multiple(y, self.patch).float()
        if x.shape != y.shape:
            raise ValueError(
                f'Distorted and reference shapes differ: {tuple(x.shape)} vs {tuple(y.shape)}.'
            )

        b = x.shape[0]
        tokens, cls = self._encode(torch.cat([y, x]))
        delta = (tokens[:b] - tokens[b:]) @ self.lens.U
        score = delta.square().sum(dim=-1).mean(dim=1).clamp_min(0).sqrt()
        if cls is not None:
            cls_term = 1 - F.cosine_similarity(cls[:b], cls[b:], dim=-1)
            score = score + self.cls_weight * cls_term.clamp_min(0)
        return score
