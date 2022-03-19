r"""DISTS metric

Created by: https://github.com/dingkeyan93/DISTS/blob/master/DISTS_pytorch/DISTS_pt.py

Modified by: Jiadi Mo (https://github.com/JiadiMo)

"""

import numpy as np
import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F

from pyiqa.utils.registry import ARCH_REGISTRY
from pyiqa.archs.arch_util import load_pretrained_network

default_model_urls = {
    'url': 'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/DISTS_weights-f5e65c96.pth'
}


class L2pooling(nn.Module):

    def __init__(self, filter_size=5, stride=2, channels=None, pad_off=0):
        super(L2pooling, self).__init__()
        self.padding = (filter_size - 2) // 2
        self.stride = stride
        self.channels = channels
        a = np.hanning(filter_size)[1:-1]
        g = torch.Tensor(a[:, None] * a[None, :])
        g = g / torch.sum(g)
        self.register_buffer('filter', g[None, None, :, :].repeat((self.channels, 1, 1, 1)))

    def forward(self, input):
        input = input**2
        out = F.conv2d(input, self.filter, stride=self.stride, padding=self.padding, groups=input.shape[1])
        return (out + 1e-12).sqrt()


@ARCH_REGISTRY.register()
class DISTS(torch.nn.Module):
    r'''DISTS model.
    Args:
        pretrained_model_path (String): Pretrained model path.

    '''

    def __init__(self, pretrained=True, pretrained_model_path=None, **kwargs):
        """Refer to offical code https://github.com/dingkeyan93/DISTS
        """
        super(DISTS, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        self.stage3 = torch.nn.Sequential()
        self.stage4 = torch.nn.Sequential()
        self.stage5 = torch.nn.Sequential()
        for x in range(0, 4):
            self.stage1.add_module(str(x), vgg_pretrained_features[x])
        self.stage2.add_module(str(4), L2pooling(channels=64))
        for x in range(5, 9):
            self.stage2.add_module(str(x), vgg_pretrained_features[x])
        self.stage3.add_module(str(9), L2pooling(channels=128))
        for x in range(10, 16):
            self.stage3.add_module(str(x), vgg_pretrained_features[x])
        self.stage4.add_module(str(16), L2pooling(channels=256))
        for x in range(17, 23):
            self.stage4.add_module(str(x), vgg_pretrained_features[x])
        self.stage5.add_module(str(23), L2pooling(channels=512))
        for x in range(24, 30):
            self.stage5.add_module(str(x), vgg_pretrained_features[x])

        for param in self.parameters():
            param.requires_grad = False

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1))

        self.chns = [3, 64, 128, 256, 512, 512]
        self.register_parameter('alpha', nn.Parameter(torch.randn(1, sum(self.chns), 1, 1)))
        self.register_parameter('beta', nn.Parameter(torch.randn(1, sum(self.chns), 1, 1)))
        self.alpha.data.normal_(0.1, 0.01)
        self.beta.data.normal_(0.1, 0.01)

        if pretrained_model_path is not None:
            load_pretrained_network(self, pretrained_model_path, False)
        elif pretrained:
            load_pretrained_network(self, default_model_urls['url'], False)

    def forward_once(self, x):
        h = (x - self.mean) / self.std
        h = self.stage1(h)
        h_relu1_2 = h
        h = self.stage2(h)
        h_relu2_2 = h
        h = self.stage3(h)
        h_relu3_3 = h
        h = self.stage4(h)
        h_relu4_3 = h
        h = self.stage5(h)
        h_relu5_3 = h
        return [x, h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]

    def forward(self, x, y):
        r"""Compute IQA using DISTS model.

        Args:
            x: An input tensor with (N, C, H, W) shape. RGB channel order for colour images.
            y: An reference tensor with (N, C, H, W) shape. RGB channel order for colour images.

        Returns:
            Value of DISTS model.

        """
        feats0 = self.forward_once(x)
        feats1 = self.forward_once(y)
        dist1 = 0
        dist2 = 0
        c1 = 1e-6
        c2 = 1e-6

        w_sum = self.alpha.sum() + self.beta.sum()
        alpha = torch.split(self.alpha / w_sum, self.chns, dim=1)
        beta = torch.split(self.beta / w_sum, self.chns, dim=1)
        for k in range(len(self.chns)):
            x_mean = feats0[k].mean([2, 3], keepdim=True)
            y_mean = feats1[k].mean([2, 3], keepdim=True)
            S1 = (2 * x_mean * y_mean + c1) / (x_mean**2 + y_mean**2 + c1)
            dist1 = dist1 + (alpha[k] * S1).sum(1, keepdim=True)

            x_var = ((feats0[k] - x_mean)**2).mean([2, 3], keepdim=True)
            y_var = ((feats1[k] - y_mean)**2).mean([2, 3], keepdim=True)
            xy_cov = (feats0[k] * feats1[k]).mean([2, 3], keepdim=True) - x_mean * y_mean
            S2 = (2 * xy_cov + c2) / (x_var + y_var + c2)
            dist2 = dist2 + (beta[k] * S2).sum(1, keepdim=True)

        score = 1 - (dist1 + dist2).squeeze()

        return score
