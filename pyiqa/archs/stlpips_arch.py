"""ST-LPIPS Model

github repo link: https://github.com/abhijay9/ShiftTolerant-LPIPS

Cite as:
@inproceedings{ghildyal2022stlpips,
    title={Shift-tolerant Perceptual Similarity Metric},
    author={Abhijay Ghildyal and Feng Liu},
    booktitle={European Conference on Computer Vision},
    year={2022}
}

"""

import torch
from torchvision import models
import torch.nn as nn
from collections import namedtuple
import numpy as np
import torch.nn.functional as F

from pyiqa.utils.registry import ARCH_REGISTRY
from pyiqa.archs.arch_util import load_pretrained_network


default_model_urls = {
    # key "url" is the default
    "alex_shift_tolerant": "https://github.com/abhijay9/ShiftTolerant-LPIPS/raw/main/stlpips/weights/vST0.0/alex_shift_tolerant.pth",
    "vgg_shift_tolerant": "https://github.com/abhijay9/ShiftTolerant-LPIPS/raw/main/stlpips/weights/vST0.0/vgg_shift_tolerant.pth",
}


def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2, 3], keepdim=keepdim)


def upsample(in_tens, out_HW=(64, 64)):  # assumes scale factor is same for H and W
    in_H, in_W = in_tens.shape[2], in_tens.shape[3]
    return nn.Upsample(size=out_HW, mode="bilinear", align_corners=False)(in_tens)


def normalize_tensor(in_feat, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat**2, dim=1, keepdim=True))
    return in_feat / (norm_factor + eps)


@ARCH_REGISTRY.register()
class STLPIPS(nn.Module):
    """ST-LPIPS model.
    Args:
        lpips (Boolean) : Whether to use linear layers on top of base/trunk network.
        pretrained (Boolean): Whether means linear layers are calibrated with human
            perceptual judgments.
        net (String): ['alex','vgg','squeeze'] are the base/trunk networks available.
        pretrained_model_path (String): Petrained model path.

        The following parameters should only be changed if training the network:

        eval_mode (Boolean): choose the mode; True is for test mode (default).
        pnet_tune (Boolean): Whether to tune the base/trunk network.
        use_dropout (Boolean): Whether to use dropout when training linear layers.
    """

    def __init__(
        self,
        pretrained=True,
        net="alex",
        variant="shift_tolerant",
        lpips=True,
        spatial=False,
        pnet_tune=False,
        use_dropout=True,
        pretrained_model_path=None,
        eval_mode=True,
        blur_filter_size=3,
    ):
        super(STLPIPS, self).__init__()

        self.pnet_type = net
        self.pnet_tune = pnet_tune
        self.spatial = spatial
        self.lpips = lpips  # false means baseline of just averaging all layers
        self.scaling_layer = ScalingLayer()

        if self.pnet_type in ["vgg"]:
            net_type = vggnet
            self.chns = [64, 128, 256, 512, 512]
        elif self.pnet_type == "alex":
            net_type = alexnet
            self.chns = [64, 192, 384, 256, 256]

        self.net = net_type(
            requires_grad=self.pnet_tune, variant=variant, filter_size=blur_filter_size
        )

        self.L = len(self.chns)

        if lpips:
            self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
            self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
            self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
            self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
            self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
            self.lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
            if self.pnet_type == "squeeze":  # 7 layers for squeezenet
                self.lin5 = NetLinLayer(self.chns[5], use_dropout=use_dropout)
                self.lin6 = NetLinLayer(self.chns[6], use_dropout=use_dropout)
                self.lins += [self.lin5, self.lin6]
            self.lins = nn.ModuleList(self.lins)

            if pretrained_model_path is not None:
                load_pretrained_network(self, pretrained_model_path, False)
            elif pretrained:
                load_pretrained_network(
                    self, default_model_urls[f"{net}_{variant}"], False
                )

        if eval_mode:
            self.eval()

    def forward(self, in0, in1, retPerLayer=False, normalize=True):
        """Computation IQA using LPIPS.
        Args:
            in1: An input tensor. Shape :math:`(N, C, H, W)`.
            in0: A reference tensor. Shape :math:`(N, C, H, W)`.
            retPerLayer (Boolean): return result contains ressult of
                each layer or not. Default: False.
            normalize (Boolean): Whether to normalize image data range
                in [0,1] to [-1,1]. Default: True.

        Returns:
            Quality score.

        """

        if (normalize):  # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
            in0 = 2 * in0 - 1
            in1 = 2 * in1 - 1

        in0_input, in1_input = (self.scaling_layer(in0), self.scaling_layer(in1))
        outs0, outs1 = self.net.forward(in0_input), self.net.forward(in1_input)
        feats0, feats1, diffs = {}, {}, {}

        for kk in range(self.L):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(
                outs1[kk]
            )
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        if self.lpips:
            if self.spatial:
                res = [
                    upsample(self.lins[kk](diffs[kk]), out_HW=in0.shape[2:])
                    for kk in range(self.L)
                ]
            else:
                res = [
                    spatial_average(self.lins[kk](diffs[kk]), keepdim=True)
                    for kk in range(self.L)
                ]
        else:
            if self.spatial:
                res = [
                    upsample(diffs[kk].sum(dim=1, keepdim=True), out_HW=in0.shape[2:])
                    for kk in range(self.L)
                ]
            else:
                res = [
                    spatial_average(diffs[kk].sum(dim=1, keepdim=True), keepdim=True)
                    for kk in range(self.L)
                ]

        val = res[0]
        for l in range(1, self.L):
            val += res[l]

        if retPerLayer:
            return (val, res)
        else:
            return val


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer(
            "shift", torch.Tensor([-0.030, -0.088, -0.188])[None, :, None, None]
        )
        self.register_buffer(
            "scale", torch.Tensor([0.458, 0.448, 0.450])[None, :, None, None]
        )

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """A single linear layer which does a 1x1 conv"""

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()

        layers = (
            [
                nn.Dropout(),
            ]
            if (use_dropout)
            else []
        )
        layers += [
            nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class alexnet(nn.Module):
    def __init__(self, requires_grad=False, variant="shift_tolerant", filter_size=3):
        super(alexnet, self).__init__()

        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()

        if variant == "vanilla":
            features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
                nn.ReLU(inplace=True),  # 1
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(64, 192, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),  # 4
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),  # 7
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),  # 9
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),  # 11
                nn.MaxPool2d(kernel_size=3, stride=2),
            )
            for x in range(2):
                self.slice1.add_module(str(x), features[x])
            for x in range(2, 5):
                self.slice2.add_module(str(x), features[x])
            for x in range(5, 8):
                self.slice3.add_module(str(x), features[x])
            for x in range(8, 10):
                self.slice4.add_module(str(x), features[x])
            for x in range(10, 12):
                self.slice5.add_module(str(x), features[x])

        elif variant == "antialiased":
            features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=11, stride=2, padding=2),
                nn.ReLU(inplace=True),  # 1
                Downsample(filt_size=filter_size, stride=2, channels=64),
                nn.MaxPool2d(kernel_size=3, stride=1),
                Downsample(filt_size=filter_size, stride=2, channels=64),
                nn.Conv2d(64, 192, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),  # 6
                nn.MaxPool2d(kernel_size=3, stride=1),
                Downsample(filt_size=filter_size, stride=2, channels=192),
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),  # 10
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),  # 12
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),  # 14
                nn.MaxPool2d(kernel_size=3, stride=1),
                Downsample(filt_size=filter_size, stride=2, channels=256),
            )
            for x in range(2):
                self.slice1.add_module(str(x), features[x])
            for x in range(2, 7):
                self.slice2.add_module(str(x), features[x])
            for x in range(7, 11):
                self.slice3.add_module(str(x), features[x])
            for x in range(11, 13):
                self.slice4.add_module(str(x), features[x])
            for x in range(13, 15):
                self.slice5.add_module(str(x), features[x])

        elif (
            variant == "shift_tolerant"
        ):  # antialiased_blurpoolReflectionPad2_conv1stride1_blurAfter
            features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=11, stride=1, padding=2),
                Downsample(filt_size=filter_size, stride=2, channels=64, pad_more=True),
                nn.ReLU(inplace=True),  # 2
                nn.MaxPool2d(kernel_size=3, stride=1),
                Downsample(filt_size=filter_size, stride=2, channels=64, pad_more=True),
                nn.Conv2d(64, 192, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),  # 6
                nn.MaxPool2d(kernel_size=3, stride=1),
                Downsample(
                    filt_size=filter_size, stride=2, channels=192, pad_more=True
                ),
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),  # 10
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),  # 12
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),  # 14
                nn.MaxPool2d(kernel_size=3, stride=1),
                Downsample(
                    filt_size=filter_size, stride=2, channels=256, pad_more=True
                ),
            )
            for x in range(3):
                self.slice1.add_module(str(x), features[x])
            for x in range(3, 7):
                self.slice2.add_module(str(x), features[x])
            for x in range(7, 11):
                self.slice3.add_module(str(x), features[x])
            for x in range(11, 13):
                self.slice4.add_module(str(x), features[x])
            for x in range(13, 15):
                self.slice5.add_module(str(x), features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        alexnet_outputs = namedtuple(
            "AlexnetOutputs", ["relu1", "relu2", "relu3", "relu4", "relu5"]
        )
        out = alexnet_outputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5)

        return out


class vggnet(nn.Module):
    def __init__(self, requires_grad=False, variant="shift_tolerant", filter_size=3):
        super(vggnet, self).__init__()

        filter_size = 3
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        self.N_slices = 5

        if variant == "vanilla":
            vgg_features = tv.vgg16(pretrained=False).features
            for x in range(4):
                self.slice1.add_module(str(x), vgg_features[x])
            for x in range(4, 9):
                self.slice2.add_module(str(x), vgg_features[x])
            for x in range(9, 16):
                self.slice3.add_module(str(x), vgg_features[x])
            for x in range(16, 23):
                self.slice4.add_module(str(x), vgg_features[x])
            for x in range(23, 30):
                self.slice5.add_module(str(x), vgg_features[x])

        elif variant == "shift_tolerant":
            vgg_features = vgg16(filter_size=filter_size, pad_more=True).features
            for x in range(4):
                self.slice1.add_module(str(x), vgg_features[x])
            for x in range(4, 10):
                self.slice2.add_module(str(x), vgg_features[x])
            for x in range(10, 18):
                self.slice3.add_module(str(x), vgg_features[x])
            for x in range(18, 26):
                self.slice4.add_module(str(x), vgg_features[x])
            for x in range(26, 34):
                self.slice5.add_module(str(x), vgg_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple(
            "VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"]
        )
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)

        return out


# Copyright (c) 2019, Adobe Inc. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International Public License. To view a copy of this license, visit
# https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.


class Downsample(nn.Module):
    def __init__(
        self,
        pad_type="reflect",
        filt_size=3,
        stride=2,
        channels=None,
        pad_off=0,
        pad_size="",
        pad_more=False,
    ):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        if pad_size == "2k" or pad_more == True:
            self.pad_sizes = [
                int(1.0 * (filt_size - 1)),
                int(np.ceil(1.0 * (filt_size - 1))),
                int(1.0 * (filt_size - 1)),
                int(np.ceil(1.0 * (filt_size - 1))),
            ]
        elif pad_size == "none":
            self.pad_sizes = [0, 0, 0, 0]
        else:
            self.pad_sizes = [
                int(1.0 * (filt_size - 1) / 2),
                int(np.ceil(1.0 * (filt_size - 1) / 2)),
                int(1.0 * (filt_size - 1) / 2),
                int(np.ceil(1.0 * (filt_size - 1) / 2)),
            ]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.0)
        self.channels = channels

        if self.filt_size == 1:
            a = np.array(
                [
                    1.0,
                ]
            )
        elif self.filt_size == 2:
            a = np.array([1.0, 1.0])
        elif self.filt_size == 3:
            a = np.array([1.0, 2.0, 1.0])
        elif self.filt_size == 4:
            a = np.array([1.0, 3.0, 3.0, 1.0])
        elif self.filt_size == 5:
            a = np.array([1.0, 4.0, 6.0, 4.0, 1.0])
        elif self.filt_size == 6:
            a = np.array([1.0, 5.0, 10.0, 10.0, 5.0, 1.0])
        elif self.filt_size == 7:
            a = np.array([1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0])

        filt = torch.Tensor(a[:, None] * a[None, :])
        filt = filt / torch.sum(filt)
        self.register_buffer(
            "filt", filt[None, None, :, :].repeat((self.channels, 1, 1, 1))
        )

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if self.filt_size == 1:
            if self.pad_off == 0:
                return inp[:, :, :: self.stride, :: self.stride]
            else:
                return self.pad(inp)[:, :, :: self.stride, :: self.stride]
        else:
            return F.conv2d(
                self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1]
            )


def get_pad_layer(pad_type):
    if pad_type in ["refl", "reflect"]:
        PadLayer = nn.ReflectionPad2d
    elif pad_type in ["repl", "replicate"]:
        PadLayer = nn.ReplicationPad2d
    elif pad_type == "zero":
        PadLayer = nn.ZeroPad2d
    else:
        print("Pad type [%s] not recognized" % pad_type)
    return PadLayer


# This code is built from the PyTorch examples repository: https://github.com/pytorch/vision/tree/master/torchvision/models.
# Copyright (c) 2017 Torch Contributors.
# The Pytorch examples are available under the BSD 3-Clause License.
#
# ==========================================================================================
#
# Adobe’s modifications are Copyright 2019 Adobe. All rights reserved.
# Adobe’s modifications are licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International Public License (CC-NC-SA-4.0). To view a copy of the license, visit
# https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.
#
# ==========================================================================================
#
# BSD-3 License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        # print(x.shape)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if (
                    m.in_channels != m.out_channels
                    or m.out_channels != m.groups
                    or m.bias is not None
                ):
                    # don't want to reinitialize downsample layers, code assuming normal conv layers will not have these characteristics
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                else:
                    print("Not initializing")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False, filter_size=1, pad_more=False, fconv=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            # layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            layers += [
                nn.MaxPool2d(kernel_size=2, stride=1),
                Downsample(
                    filt_size=filter_size,
                    stride=2,
                    channels=in_channels,
                    pad_more=pad_more,
                ),
            ]
        else:
            if fconv:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=2)
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M",],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M",],
}


def vgg16(pretrained=False, filter_size=1, pad_more=False, fconv=False, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs["init_weights"] = False
    model = VGG(
        make_layers(cfg["D"], filter_size=filter_size, pad_more=pad_more, fconv=fconv),
        **kwargs,
    )
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    return model
