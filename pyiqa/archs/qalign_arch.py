r"""Q-Align: All-in-one Foundation Model for visual scoring.

Reference:
@article{wu2023qalign,
  title={Q-Align: Teaching LMMs for Visual Scoring via Discrete Text-Defined Levels},
  author={Wu, Haoning and Zhang, Zicheng and Zhang, Weixia and Chen, Chaofeng and Li, Chunyi and Liao, Liang and Wang, Annan and Zhang, Erli and Sun, Wenxiu and Yan, Qiong and Min, Xiongkuo and Zhai, Guangtai and Lin, Weisi},
  journal={arXiv preprint arXiv:2312.17090},
  year={2023},
  institution={Nanyang Technological University and Shanghai Jiao Tong University and Sensetime Research},
  note={Equal Contribution by Wu, Haoning and Zhang, Zicheng. Project Lead by Wu, Haoning. Corresponding Authors: Zhai, Guangtai and Lin, Weisi.}
}

Reference url: https://github.com/Q-Future/Q-Align
"""

import torch
from torch import nn
import warnings
from PIL import Image
import torchvision.transforms.functional as F
from transformers import BitsAndBytesConfig, CLIPImageProcessor

from .constants import OPENAI_CLIP_MEAN
from .q_align.modeling_mplug_owl2 import MPLUGOwl2LlamaForCausalLM
from pyiqa.utils.registry import ARCH_REGISTRY


def expand2square(pil_img):
    background_color = tuple(int(x * 255) for x in OPENAI_CLIP_MEAN)
    width, height = pil_img.size
    maxwh = max(width, height)
    result = Image.new(pil_img.mode, (maxwh, maxwh), background_color)
    result.paste(pil_img, ((maxwh - width) // 2, (maxwh - height) // 2))
    return result


@ARCH_REGISTRY.register()
class QAlign(nn.Module):
    def __init__(self, dtype='fp16') -> None:
        super().__init__()

        assert dtype in ['fp16', '4bit', '8bit'], (
            f"Invalid dtype {dtype}. Choose from 'fp16', '4bit', or '8bit'."
        )

        model_kwargs = {
            'trust_remote_code': False,
            'torch_dtype': torch.float16 if dtype == 'fp16' else None,
        }
        if dtype in ['4bit', '8bit']:
            quant_kwargs = {
                'load_in_4bit': dtype == '4bit',
                'load_in_8bit': dtype == '8bit',
            }
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
            self.model = MPLUGOwl2LlamaForCausalLM.from_pretrained('q-future/one-align', **model_kwargs)
        if getattr(self.model, 'generation_config', None) is not None:
            gen_cfg = self.model.generation_config
            if not getattr(gen_cfg, 'do_sample', False):
                gen_cfg.temperature = None
                gen_cfg.top_p = None
        self.image_processor = CLIPImageProcessor.from_pretrained('q-future/one-align')

    def preprocess(self, x):
        assert x.shape[0] == 1, 'Currently, only support batch size 1.'
        image = expand2square(F.to_pil_image(x[0]))
        image_tensor = self.image_processor.preprocess(image, return_tensors='pt')[
            'pixel_values'
        ].half()
        return image_tensor.to(x.device)

    def forward(self, x, task_='quality', input_='image'):
        """
        task_: str, optional [quality, aesthetic]
        """
        if input_ != 'image':
            raise NotImplementedError(f'Input type {input_} is not supported yet.')

        image_tensor = self.preprocess(x)
        score = self.model.score(
            images=None, image_tensor=image_tensor, task_=task_, input_=input_
        )

        return score
