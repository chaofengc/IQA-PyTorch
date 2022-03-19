r"""HyperNet Metric

Created by: https://github.com/SSL92/hyperIQA

Modified by: Chaofeng Chen (https://github.com/chaofengc)

"""

import torch
import torch.nn as nn
import timm
from pyiqa.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class HyperNet(nn.Module):
    """HyperNet Model.
    Args:
        base_model_name (String): pretrained model to extract features,
            can be any models supported by timm. Default: resnet50.
        pretrained_model_path (String): Pretrained model path.
        default_mean (list): Default mean value.
        default_std (list): Default std value.

    Reference:
        Su, Shaolin, Qingsen Yan, Yu Zhu, Cheng Zhang, Xin Ge,
        Jinqiu Sun, and Yanning Zhang. "Blindly assess image
        quality in the wild guided by a self-adaptive hyper network."
        In Proceedings of the IEEE/CVF Conference on Computer Vision
        and Pattern Recognition (CVPR), pp. 3667-3676. 2020.

    """

    def __init__(
        self,
        base_model_name='resnet50',
        pretrained_model_path=None,
        default_mean=[0.485, 0.456, 0.406],
        default_std=[0.229, 0.224, 0.225],
    ):
        super(HyperNet, self).__init__()
        self.base_model = timm.create_model(base_model_name, pretrained=True, features_only=True)

        lda_out_channels = 16
        hyper_in_channels = 112
        target_in_size = 224
        hyper_fc_channels = [112, 56, 28, 14, 1]
        feature_size = 7  # spatial size of the last features from base model
        self.hyper_fc_channels = hyper_fc_channels

        # local distortion aware module
        self.lda_modules = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, 16, kernel_size=1, stride=1, padding=0, bias=False), nn.AvgPool2d(7, stride=7),
                nn.Flatten(), nn.Linear(16 * 64, lda_out_channels)),
            nn.Sequential(
                nn.Conv2d(512, 32, kernel_size=1, stride=1, padding=0, bias=False),
                nn.AvgPool2d(7, stride=7),
                nn.Flatten(),
                nn.Linear(32 * 16, lda_out_channels),
            ),
            nn.Sequential(
                nn.Conv2d(1024, 64, kernel_size=1, stride=1, padding=0, bias=False),
                nn.AvgPool2d(7, stride=7),
                nn.Flatten(),
                nn.Linear(64 * 4, lda_out_channels),
            ),
            nn.Sequential(
                nn.AvgPool2d(7, stride=7),
                nn.Flatten(),
                nn.Linear(2048, target_in_size - lda_out_channels * 3),
            )
        ])

        # Hyper network part, conv for generating target fc weights, fc for generating target fc biases
        self.fc_w_modules = nn.ModuleList([])
        for i in range(4):
            if i == 0:
                out_ch = int(target_in_size * hyper_fc_channels[i] / feature_size**2)
            else:
                out_ch = int(hyper_fc_channels[i - 1] * hyper_fc_channels[i] / feature_size**2)
            self.fc_w_modules.append(nn.Conv2d(hyper_in_channels, out_ch, 3, padding=(1, 1)), )
        self.fc_w_modules.append(
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(hyper_in_channels, hyper_fc_channels[3]),
            ))

        self.fc_b_modules = nn.ModuleList([])
        for i in range(5):
            self.fc_b_modules.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(hyper_in_channels, hyper_fc_channels[i]),
                ))

        # Conv layers for resnet output features
        self.conv1 = nn.Sequential(
            nn.Conv2d(2048, 1024, 1, padding=(0, 0)), nn.ReLU(inplace=True), nn.Conv2d(1024, 512, 1, padding=(0, 0)),
            nn.ReLU(inplace=True), nn.Conv2d(512, hyper_in_channels, 1, padding=(0, 0)), nn.ReLU(inplace=True))

        self.global_pool = nn.Sequential()

        self.default_mean = torch.Tensor(default_mean).view(1, 3, 1, 1)
        self.default_std = torch.Tensor(default_std).view(1, 3, 1, 1)

    def load_pretrained_network(self, model_path):
        print(f'Loading pretrained model from {model_path}')
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))['state_dict']
        self.net.load_state_dict(state_dict, strict=True)

    def preprocess(self, x):
        # input must have shape of (224, 224) because of network design
        if x.shape[2:] != torch.Size([224, 224]):
            x = nn.functional.interpolate(x, (224, 224), mode='bicubic')
        x = (x - self.default_mean.to(x)) / self.default_std.to(x)
        return x

    def random_crop_test(self, x, sample_num=25):
        b, c, h, w = x.shape
        th = tw = 224
        cropped_x = []
        for s in range(sample_num):
            i = torch.randint(0, h - th + 1, size=(1, )).item()
            j = torch.randint(0, w - tw + 1, size=(1, )).item()
            cropped_x.append(x[:, :, i:i + th, j:j + tw])
        cropped_x = torch.cat(cropped_x, dim=0)
        results = self.forward_patch(cropped_x)
        results = results.reshape(sample_num, b).mean(dim=0)
        return results.unsqueeze(-1)

    def forward_patch(self, x):
        assert x.shape[2:] == torch.Size([224, 224]), f'Input patch size must be (224, 224), but got {x.shape[2:]}'
        x = self.preprocess(x)

        base_feats = self.base_model(x)[1:]
        # multi-scale local distortion aware features
        lda_feat_list = []
        for bf, ldam in zip(base_feats, self.lda_modules):
            lda_feat_list.append(ldam(bf))
        lda_feat = torch.cat(lda_feat_list, dim=1)

        # calculate target net weights & bias
        target_fc_w = []
        target_fc_b = []
        hyper_in_feat = self.conv1(base_feats[-1])
        batch_size = hyper_in_feat.shape[0]

        for i in range(len(self.fc_w_modules)):
            tmp_fc_w = self.fc_w_modules[i](hyper_in_feat).reshape(batch_size, self.hyper_fc_channels[i], -1)
            target_fc_w.append(tmp_fc_w)
            target_fc_b.append(self.fc_b_modules[i](hyper_in_feat))

        # get final IQA score
        x = lda_feat.unsqueeze(1)
        for i in range(len(target_fc_w)):
            if i != 4:
                x = torch.sigmoid(torch.bmm(x, target_fc_w[i].transpose(1, 2)) + target_fc_b[i].unsqueeze(1))
            else:
                x = torch.bmm(x, target_fc_w[i].transpose(1, 2)) + target_fc_b[i].unsqueeze(1)

        return x.squeeze(-1)

    def forward(self, x):
        r"""HYPERNET model.
        Args:
            x: A distortion tensor. Shape :math:`(N, C, H, W)`.
        """
        # imagenet normalization of input is hard coded

        if self.training:
            return self.forward_patch(x)
        else:
            return self.random_crop_test(x)
