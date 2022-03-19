"""CKDN model.

Created by: Chaofeng Chen (https://github.com/chaofengc)

Refer to:
    https://github.com/researchmm/CKDN.

"""

import torch
import torch.nn as nn
import math
import torchvision as tv
from pyiqa.utils.registry import ARCH_REGISTRY
from pyiqa.archs.arch_util import load_pretrained_network

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

default_model_urls = {
    'url': 'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/CKDN_model_best-38b27dc6.pth'
}

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError('Dilation > 1 not supported in BasicBlock')
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 num_classes=1000,
                 zero_init_residual=False,
                 groups=1,
                 width_per_group=64,
                 replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.k = 3
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError('replace_stride_with_dilation should be None '
                             'or a 3-element tuple, got {}'.format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.head = 8
        self.qse_1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.qse_2 = self._make_layer(block, 64, layers[0])
        self.csp = self._make_layer(block, 128, layers[1], stride=2, dilate=False)
        self.inplanes = 64
        self.dte_1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.dte_2 = self._make_layer(block, 64, layers[0])
        self.aux_csp = self._make_layer(block, 128, layers[1], stride=2, dilate=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_ = nn.Sequential(
            nn.Linear((512) * 1 * 1, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, 1),
        )
        self.fc1_ = nn.Sequential(
            nn.Linear((512) * 1 * 1, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, 1),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation,
                  norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, y):

        rest1 = x
        dist1 = y

        rest1 = self.qse_2(self.maxpool(self.qse_1(rest1)))
        dist1 = self.dte_2(self.maxpool(self.dte_1(dist1)))

        x = rest1 - dist1
        x = self.csp(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        dr = torch.sigmoid(self.fc_(x))
        return dr


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        keys = state_dict.keys()
        for key in list(keys):
            if 'conv1' in key:
                state_dict[key.replace('conv1', 'qse_1')] = state_dict[key]
                state_dict[key.replace('conv1', 'dte_1')] = state_dict[key]
            if 'layer1' in key:
                state_dict[key.replace('layer1', 'qse_2')] = state_dict[key]
                state_dict[key.replace('layer1', 'dte_2')] = state_dict[key]
            if 'layer2' in key:
                state_dict[key.replace('layer2', 'csp')] = state_dict[key]
                state_dict[key.replace('layer2', 'aux_csp')] = state_dict[key]
        model.load_state_dict(state_dict, strict=False)
    return model


@ARCH_REGISTRY.register()
class CKDN(nn.Module):
    r"""CKDN metric.

    Args:
        pretrained_model_path (String):  The model path.
        use_default_preprocess (Boolean): Whether use default preprocess, default: True.
        default_mean (tuple): The mean value.
        default_std (tuple): The std value.

    Reference:
        Zheng, Heliang, Huan Yang, Jianlong Fu, Zheng-Jun Zha, and Jiebo Luo.
        "Learning conditional knowledge distillation for degraded-reference image
        quality assessment." In Proceedings of the IEEE/CVF International Conference
        on Computer Vision (ICCV), pp. 10242-10251. 2021.

    """

    def __init__(self,
                 pretrained=True,
                 pretrained_model_path=None,
                 use_default_preprocess=True,
                 default_mean=(0.485, 0.456, 0.406),
                 default_std=(0.229, 0.224, 0.225),
                 **kwargs):
        super().__init__()
        self.net = _resnet('resnet50', Bottleneck, [3, 4, 6, 3], True, True, **kwargs)
        self.use_default_preprocess = use_default_preprocess

        self.default_mean = torch.Tensor(default_mean).view(1, 3, 1, 1)
        self.default_std = torch.Tensor(default_std).view(1, 3, 1, 1)

        if pretrained_model_path is not None:
            load_pretrained_network(self, pretrained_model_path)
        elif pretrained:
            load_pretrained_network(self, default_model_urls['url'])

    def _default_preprocess(self, x, y):
        """default preprocessing of CKDN: https://github.com/researchmm/CKDN
        Useful when using this metric as losses.
        Results are slightly different due to different resize behavior of PIL Image and pytorch interpolate function.

        Args:
            x, y:
              shape, (N, C, H, W) in RGB format;
              value range, 0 ~ 1

        """
        scaled_size = int(math.floor(288 / 0.875))
        x = tv.transforms.functional.resize(x, scaled_size, tv.transforms.InterpolationMode.BICUBIC)
        y = tv.transforms.functional.resize(y, scaled_size, tv.transforms.InterpolationMode.NEAREST)

        x = tv.transforms.functional.center_crop(x, 288)
        y = tv.transforms.functional.center_crop(y, 288)

        x = (x - self.default_mean.to(x)) / self.default_std.to(x)
        y = (y - self.default_mean.to(y)) / self.default_std.to(y)
        return x, y

    def forward(self, x, y):
        r"""Compute IQA using CKDN model.

        Args:
            x: An input tensor with (N, C, H, W) shape. RGB channel order for colour images.
            y: An reference tensor with (N, C, H, W) shape. RGB channel order for colour images.

        Returns:
            Value of CKDN model.

        """
        if self.use_default_preprocess:
            x, y = self._default_preprocess(x, y)
        return self.net(x, y)
