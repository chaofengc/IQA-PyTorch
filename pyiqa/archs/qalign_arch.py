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
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

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
class QAlign(nn.Module):
    def __init__(self, dtype='fp16') -> None:
        super().__init__()

        assert dtype in ['fp16', '4bit', '8bit'], (
            f"Invalid dtype {dtype}. Choose from 'nf4', 'int8', or 'fp16'."
        )

        # load model
        self.model = AutoModelForCausalLM.from_pretrained(
            'q-future/one-align',
            trust_remote_code=True,
            load_in_4bit=True if dtype == '4bit' else False,
            load_in_8bit=True if dtype == '8bit' else False,
            torch_dtype=torch.float16 if dtype == 'fp16' else None,
        )
        self.image_processor = CLIPImageProcessor.from_pretrained('q-future/one-align')

    def preprocess(self, x):
        assert x.shape[0] == 1, 'Currently, only support batch size 1.'
        images = F.to_pil_image(x[0])
        images = expand2square(images)
        image_tensor = self.image_processor.preprocess(images, return_tensors='pt')[
            'pixel_values'
        ].half()
        return image_tensor.to(x.device)

    def forward(self, x, task_='quality', input_='image'):
        """
        task_: str, optional [quality, aesthetic]
        """
        if input_ == 'image':
            image_tensor = self.preprocess(x)
            score = self.model.score(
                images=None, image_tensor=image_tensor, task_=task_, input_=input_
            )
        else:
            raise NotImplementedError(f'Input type {input_} is not supported yet.')

        return score
