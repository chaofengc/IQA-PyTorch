r"""Adaptive Image Quality Assessment via Teaching Large Multimodal Model to Compare

Reference:
@inproceedings{zhu2024adaptive,
  title={Adaptive Image Quality Assessment via Teaching Large Multimodal Model to Compare},
  author={Zhu, Hanwei and Wu, Haoning and Li, Yixuan and Zhang, Zicheng and Chen, Baoliang and Zhu, Lingyu and Fang, Yuming and Zhai, Guangtao and Lin, Weisi and Wang, Shiqi},
  booktitle={Conference on Neural Information Processing Systems},
  year={2024},
}

Reference url: https://github.com/Q-Future/Compare2Score
"""

import torch
from torch import nn
import warnings
from .q_align.cmp_modelling_mplug_owl2 import MPLUGOwl2LlamaForCausalLM
from transformers import BitsAndBytesConfig

from .constants import OPENAI_CLIP_MEAN
from pyiqa.utils.registry import ARCH_REGISTRY
from transformers import CLIPImageProcessor
import torchvision.transforms.functional as F
from PIL import Image


def expand2square(pil_img):
    background_color = tuple(int(x * 255) for x in OPENAI_CLIP_MEAN)
    width, height = pil_img.size
    maxwh = max(width, height)
    result = Image.new(pil_img.mode, (maxwh, maxwh), background_color)
    result.paste(pil_img, ((maxwh - width) // 2, (maxwh - height) // 2))
    return result


@ARCH_REGISTRY.register()
class Compare2Score(nn.Module):
    def __init__(self, dtype='fp16') -> None:
        super().__init__()

        assert dtype in ['fp16', '4bit', '8bit'], (
            f"Invalid dtype {dtype}. Choose from 'nf4', 'int8', or 'fp16'."
        )

        model_kwargs = {
            'trust_remote_code': False,
            'torch_dtype': torch.float16 if dtype == 'fp16' else None,
        }
        if dtype in ['4bit', '8bit']:
            quant_kwargs = {'load_in_4bit': dtype == '4bit', 'load_in_8bit': dtype == '8bit'}
            if dtype == '4bit':
                quant_kwargs.update(
                    {
                        'bnb_4bit_quant_type': 'nf4',
                        'bnb_4bit_compute_dtype': torch.float16,
                    }
                )
            try:
                model_kwargs['quantization_config'] = BitsAndBytesConfig(**quant_kwargs)
                model_kwargs['torch_dtype'] = torch.float16
            except Exception as err:
                warnings.warn(
                    f"Failed to enable {dtype} quantization ({err}). Falling back to fp16.",
                    RuntimeWarning,
                )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore',
                message=r"The following generation flags are not valid and may be ignored: .*",
            )
            warnings.filterwarnings(
                'ignore',
                message=r"`do_sample` is set to `False`\. However, `temperature` is set to .*",
            )
            warnings.filterwarnings(
                'ignore',
                message=r"`do_sample` is set to `False`\. However, `top_p` is set to .*",
            )
            self.model = MPLUGOwl2LlamaForCausalLM.from_pretrained('q-future/Compare2Score', **model_kwargs)
        if getattr(self.model, 'generation_config', None) is not None:
            gen_cfg = self.model.generation_config
            if not getattr(gen_cfg, 'do_sample', False):
                gen_cfg.temperature = None
                gen_cfg.top_p = None

    def preprocess(self, x):
        assert x.shape[0] == 1, 'Currently, only support batch size 1.'
        images = F.to_pil_image(x[0])
        return images

    def forward(self, x):
        """
        x: str, path to image
        """

        image_tensor = self.preprocess(x)
        score = self.model.score(image_tensor)

        return score
