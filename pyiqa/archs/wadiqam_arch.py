r"""WaDIQaM model.

Reference:
    Bosse, Sebastian, Dominique Maniry, Klaus-Robert Müller, Thomas Wiegand,
    and Wojciech Samek. "Deep neural networks for no-reference and full-reference
    image quality assessment." IEEE Transactions on image processing 27, no. 1
    (2017): 206-219.

Created by: https://github.com/lidq92/WaDIQaM
Modified by: Chaofeng Chen (https://github.com/chaofengc)
Refer to:
    Official code from https://github.com/dmaniry/deepIQA

"""

import torch
import torch.nn as nn
from pyiqa.utils.registry import ARCH_REGISTRY

from typing import Union, List, cast


def make_layers(cfg: List[Union[str, int]]) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


@ARCH_REGISTRY.register()
class WaDIQaM(nn.Module):
    """WaDIQaM model.
    Args:
        metric_mode (String): Choose metric mode.
        weighted_average (Boolean): Average the weight.
        train_patch_num (int): Number of patch trained. Default: 32.
        pretrained_model_path (String): The pretrained model path.
        load_feature_weight_only (Boolean): Only load featureweight.
        eps (float): Constant value.

    """

    def __init__(
        self,
        metric_mode='FR',
        weighted_average=True,
        train_patch_num=32,
        pretrained_model_path=None,
        load_feature_weight_only=False,
        eps=1e-8,
    ):
        super(WaDIQaM, self).__init__()

        backbone_cfg = [32, 32, 'M', 64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M']
        self.features = make_layers(backbone_cfg)

        self.train_patch_num = train_patch_num
        self.patch_size = 32  # This cannot be changed due to network design
        self.metric_mode = metric_mode
        fc_in_channel = 512 * 3 if metric_mode == 'FR' else 512
        self.eps = eps

        self.fc_q = nn.Sequential(
            nn.Linear(fc_in_channel, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 1),
        )

        self.weighted_average = weighted_average
        if weighted_average:
            self.fc_w = nn.Sequential(
                nn.Linear(fc_in_channel, 512),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(512, 1),
                nn.ReLU(True),
            )

        if pretrained_model_path is not None:
            self.load_pretrained_network(pretrained_model_path, load_feature_weight_only)

    def load_pretrained_network(self, model_path, load_feature_weight_only=False):
        print(f'Loading pretrained model from {model_path}')
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))['state_dict']
        if load_feature_weight_only:
            print('Only load backbone feature net')
            new_state_dict = {}
            for k in state_dict.keys():
                if 'features' in k:
                    new_state_dict[k] = state_dict[k]
            self.net.load_state_dict(new_state_dict, strict=False)
        else:
            self.net.load_state_dict(state_dict, strict=True)

    def _get_random_patches(self, x, y=None):
        """train with random crop patches"""
        self.patch_num = self.train_patch_num

        b, c, h, w = x.shape
        th = tw = self.patch_size
        cropped_x = []
        cropped_y = []
        for s in range(self.train_patch_num):
            i = torch.randint(0, h - th + 1, size=(1, )).item()
            j = torch.randint(0, w - tw + 1, size=(1, )).item()
            cropped_x.append(x[:, :, i:i + th, j:j + tw])
            if y is not None:
                cropped_y.append(y[:, :, i:i + th, j:j + tw])

        if y is not None:
            cropped_x = torch.stack(cropped_x, dim=1).reshape(-1, c, th, tw)
            cropped_y = torch.stack(cropped_y, dim=1).reshape(-1, c, th, tw)
            return cropped_x, cropped_y
        else:
            cropped_x = torch.stack(cropped_x, dim=1).reshape(-1, c, th, tw)
            return cropped_x

    def _get_nonoverlap_patches(self, x, y=None):
        """test with non overlap patches"""
        self.patch_num = 0

        b, c, h, w = x.shape
        th = tw = self.patch_size
        cropped_x = []
        cropped_y = []

        for i in range(0, h - th, th):
            for j in range(0, w - tw, tw):
                cropped_x.append(x[:, :, i:i + th, j:j + tw])
                if y is not None:
                    cropped_y.append(y[:, :, i:i + th, j:j + tw])

                self.patch_num += 1

        if y is not None:
            cropped_x = torch.stack(cropped_x, dim=1).reshape(-1, c, th, tw)
            cropped_y = torch.stack(cropped_y, dim=1).reshape(-1, c, th, tw)
            return cropped_x, cropped_y
        else:
            cropped_x = torch.stack(cropped_x, dim=1).reshape(-1, c, th, tw)
            return cropped_x

    def get_patches(self, x, y=None):
        if self.training:
            return self._get_random_patches(x, y)
        else:
            return self._get_nonoverlap_patches(x, y)

    def extract_features(self, patches):
        h = self.features(patches)
        h = h.reshape(-1, self.patch_num, 512)
        return h

    def forward(self, x, y=None):
        r"""WaDIQaM model.
        Args:
            x: An input tensor. Shape :math:`(N, C, H, W)`.
            y: A reference tensor. Shape :math:`(N, C, H, W)`.
        """
        if self.metric_mode == 'FR':
            assert y is not None, 'Full reference metric requires reference input'
            x_patches, y_patches = self.get_patches(x, y)
            feat_img = self.extract_features(x_patches)
            feat_ref = self.extract_features(y_patches)
            feat_q = torch.cat((feat_ref, feat_img, feat_img - feat_ref), dim=-1)
        else:
            x_patches = self.get_patches(x)
            feat_q = self.extract_features(x_patches)

        q_score = self.fc_q(feat_q)
        weight = self.fc_w(feat_q) + self.eps  # add eps to avoid training collapse

        if self.weighted_average:
            q_final = torch.sum(q_score * weight, dim=1) / torch.sum(weight, dim=1)
        else:
            q_final = q_score.mean(dim=1)

        return q_final.reshape(-1, 1)
