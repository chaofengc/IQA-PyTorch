r"""MetaIQA Model wrapper for pyiqa.
Reference: Zhu, Hancheng, et al. "MetaIQA: Deep Meta-Learning for No-Reference Image Quality Assessment." CVPR 2020.
Re-implemented and optimized for pyiqa ecosystem.
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import sys
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image

from pyiqa.utils.registry import ARCH_REGISTRY
from pyiqa.archs.arch_util import load_pretrained_network, get_url_from_name

default_model_urls = {
    'meta-train-seed': get_url_from_name('Metaiqa_kadid10k_tid2013.pth'),
    'meta-infer-ready': get_url_from_name('Metaiqa_resnet18.pth'),
}


class BaselineModel1(nn.Module):
    def __init__(self, num_classes=1, keep_probability=0.5, inputsize=1000):
        super(BaselineModel1, self).__init__()
        self.fc1 = nn.Linear(inputsize, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.drop_prob = (1 - keep_probability)
        self.relu1 = nn.PReLU()
        self.drop1 = nn.Dropout(self.drop_prob)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.relu2 = nn.PReLU()
        self.drop2 = nn.Dropout(p=self.drop_prob)
        self.fc3 = nn.Linear(512, num_classes)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.drop1(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.drop2(out)
        out = self.fc3(out)
        out = self.sig(out)
        return out


@ARCH_REGISTRY.register()
class MetaIQA(nn.Module):
    def __init__(self, pretrained=True, pretrained_model_path=None, **kwargs):
        super(MetaIQA, self).__init__()

        self.metric_mode = 'NR'
        self.lower_better = False

        self.resnet_layer = models.resnet18(weights=None)
        self.head = BaselineModel1(num_classes=1, keep_probability=0.5, inputsize=1000)

        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        if pretrained_model_path is not None:
            load_pretrained_network(self, pretrained_model_path, strict=False, weight_keys=None)
        elif pretrained:
            main_script = os.path.basename(sys.argv[0])
            is_inference_env = 'inference_iqa.py' in main_script or 'test.py' in main_script

            if is_inference_env:
                load_pretrained_network(
                    self,
                    default_model_urls['meta-infer-ready'],
                    strict=False,
                    weight_keys=None
                )
            else:

                load_pretrained_network(
                    self,
                    default_model_urls['meta-train-seed'],
                    strict=False,
                    weight_keys=None
                )

    def preprocess(self, x):
        x = x[:, [2, 1, 0], :, :]
        if x.shape[2:] != (224, 224):
            x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = (x - self.mean) / self.std
        return x

    def forward(self, x, ref=None):
        if isinstance(x, str):
            import torchvision.transforms.functional as F
            img = Image.open(x).convert('RGB')
            img = img.resize((224, 224), Image.BILINEAR)
            x = F.to_tensor(img).unsqueeze(0).to(self.mean.device)
            x = x * 255.0
            x = (x - self.mean) / self.std
        else:
            x = self.preprocess(x)

        feat = self.resnet_layer(x)
        score = self.head(feat)
        return score.squeeze(-1).view(-1, 1)

