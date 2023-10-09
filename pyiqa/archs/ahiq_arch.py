r"""AHIQ metric introduced by

@article{lao2022attentions,
  title   = {Attentions Help CNNs See Better: Attention-based Hybrid Image Quality Assessment Network},
  author  = {Lao, Shanshan and Gong, Yuan and Shi, Shuwei and Yang, Sidi and Wu, Tianhe and Wang, Jiahao and Xia, Weihao and Yang, Yujiu},
  journal = {arXiv preprint arXiv:2204.10485},
  year    = {2022}
}

Ref url: https://github.com/IIGROUP/AHIQ
Re-implmented by: Chaofeng Chen (https://github.com/chaofengc)

"""

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.resnet import Bottleneck
from timm.models.vision_transformer import Block
from torchvision.ops.deform_conv import DeformConv2d

from pyiqa.archs.arch_util import (
    load_file_from_url,
    load_pretrained_network,
    random_crop,
)
from pyiqa.utils.registry import ARCH_REGISTRY

default_model_urls = {
    "pipal": "https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/AHIQ_vit_p8_epoch33-da3ea303.pth"
}


class SaveOutput:
    def __init__(self):
        self.outputs = {}

    def __call__(self, module, module_in, module_out):
        if module_out.device in self.outputs.keys():
            self.outputs[module_out.device].append(module_out)
        else:
            self.outputs[module_out.device] = [module_out]

    def clear(self, device):
        self.outputs[device] = []


class DeformFusion(nn.Module):
    def __init__(
        self,
        patch_size=8,
        in_channels=768 * 5,
        cnn_channels=256 * 3,
        out_channels=256 * 3,
    ):
        super().__init__()
        # in_channels, out_channels, kernel_size, stride, padding
        self.d_hidn = 512
        if patch_size == 8:
            stride = 1
        else:
            stride = 2
        self.conv_offset = nn.Conv2d(in_channels, 2 * 3 * 3, 3, 1, 1)
        self.deform = DeformConv2d(cnn_channels, out_channels, 3, 1, 1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=self.d_hidn,
                kernel_size=3,
                padding=1,
                stride=2,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.d_hidn,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                stride=stride,
            ),
        )

    def forward(self, cnn_feat, vit_feat):
        vit_feat = F.interpolate(vit_feat, size=cnn_feat.shape[-2:], mode="nearest")
        offset = self.conv_offset(vit_feat)
        deform_feat = self.deform(cnn_feat, offset)
        deform_feat = self.conv1(deform_feat)

        return deform_feat


class Pixel_Prediction(nn.Module):
    def __init__(self, inchannels=768 * 5 + 256 * 3, outchannels=256, d_hidn=1024):
        super().__init__()
        self.d_hidn = d_hidn
        self.down_channel = nn.Conv2d(inchannels, outchannels, kernel_size=1)
        self.feat_smoothing = nn.Sequential(
            nn.Conv2d(
                in_channels=256 * 3, out_channels=self.d_hidn, kernel_size=3, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.d_hidn, out_channels=512, kernel_size=3, padding=1
            ),
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv_attent = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1), nn.Sigmoid()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1),
        )

    def forward(self, f_dis, f_ref, cnn_dis, cnn_ref):
        f_dis = torch.cat((f_dis, cnn_dis), 1)
        f_ref = torch.cat((f_ref, cnn_ref), 1)
        f_dis = self.down_channel(f_dis)
        f_ref = self.down_channel(f_ref)

        f_cat = torch.cat((f_dis - f_ref, f_dis, f_ref), 1)

        feat_fused = self.feat_smoothing(f_cat)
        feat = self.conv1(feat_fused)
        f = self.conv(feat)
        w = self.conv_attent(feat)
        pred = (f * w).sum(dim=-1).sum(dim=-1) / w.sum(dim=-1).sum(dim=-1)

        return pred


@ARCH_REGISTRY.register()
class AHIQ(nn.Module):
    """
    AHIQ model implementation.

    Args:
        - num_crop (int): Number of crops to use for testing. Default is 20.
        - crop_size (int): Size of the crops. Default is 224.
        - default_mean (list): List of mean values for normalization. Default is [0.485, 0.456, 0.406].
        - default_std (list): List of standard deviation values for normalization. Default is [0.229, 0.224, 0.225].
        - pretrained (bool): Whether to use a pretrained model. Default is True.
        - pretrained_model_path (str): Path to a pretrained model. Default is None.

    Attributes:
        - resnet50 (nn.Module): ResNet50 backbone.
        - vit (nn.Module): Vision Transformer backbone.
        - deform_net (nn.Module): Deformable fusion network.
        - regressor (nn.Module): Pixel prediction network.
        - default_mean (torch.Tensor): Mean values for normalization.
        - default_std (torch.Tensor): Standard deviation values for normalization.
        - eps (float): Small value to avoid division by zero.
        - crops (int): Number of crops to use for testing.
        crop_size (int): Size of the crops.

    Methods:
        - init_saveoutput(): Initializes the SaveOutput hook to get intermediate features.
        - fix_network(model): Fixes the network by setting all parameters to not require gradients.
        - preprocess(x): Preprocesses the input tensor by normalizing it.
        - get_vit_feature(x): Gets the intermediate features from the Vision Transformer backbone.
        - get_resnet_feature(x): Gets the intermediate features from the ResNet50 backbone.
        - regress_score(dis, ref): Computes the quality score for a distorted and reference image pair.
        - forward(x, y): Computes the quality score for a batch of distorted and reference image pairs.
    """

    def __init__(
        self,
        num_crop=20,
        crop_size=224,
        default_mean=[0.485, 0.456, 0.406],
        default_std=[0.229, 0.224, 0.225],
        pretrained=True,
        pretrained_model_path=None,
    ):
        super().__init__()

        self.resnet50 = timm.create_model("resnet50", pretrained=True)
        self.vit = timm.create_model("vit_base_patch8_224", pretrained=True)
        self.fix_network(self.resnet50)
        self.fix_network(self.vit)

        self.deform_net = DeformFusion()
        self.regressor = Pixel_Prediction()

        # register hook to get intermediate features
        self.init_saveoutput()

        self.default_mean = torch.Tensor(default_mean).view(1, 3, 1, 1)
        self.default_std = torch.Tensor(default_std).view(1, 3, 1, 1)

        if pretrained_model_path is not None:
            load_pretrained_network(
                self, pretrained_model_path, True, weight_keys="params"
            )
        elif pretrained:
            weight_path = load_file_from_url(default_model_urls["pipal"])
            checkpoint = torch.load(weight_path)
            self.regressor.load_state_dict(checkpoint["regressor_model_state_dict"])
            self.deform_net.load_state_dict(checkpoint["deform_net_model_state_dict"])

        self.eps = 1e-12
        self.crops = num_crop
        self.crop_size = crop_size

    def init_saveoutput(self):
        """
        Initializes the SaveOutput hook to get intermediate features.
        """
        self.save_output = SaveOutput()
        hook_handles = []
        for layer in self.resnet50.modules():
            if isinstance(layer, Bottleneck):
                handle = layer.register_forward_hook(self.save_output)
                hook_handles.append(handle)
        for layer in self.vit.modules():
            if isinstance(layer, Block):
                handle = layer.register_forward_hook(self.save_output)
                hook_handles.append(handle)

    def fix_network(self, model):
        """
        Fixes the network by setting all parameters to not require gradients.

        Args:
            model (nn.Module): The model to fix.
        """
        for p in model.parameters():
            p.requires_grad = False

    def preprocess(self, x):
        """
        Preprocesses the input tensor by normalizing it.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.
        """
        x = (x - self.default_mean.to(x)) / self.default_std.to(x)
        return x

    def get_vit_feature(self, x):
        """
        Gets the intermediate features from the Vision Transformer backbone.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The intermediate features.
        """
        self.vit(x)
        feat = torch.cat(
            (
                self.save_output.outputs[x.device][0][:, 1:, :],
                self.save_output.outputs[x.device][1][:, 1:, :],
                self.save_output.outputs[x.device][2][:, 1:, :],
                self.save_output.outputs[x.device][3][:, 1:, :],
                self.save_output.outputs[x.device][4][:, 1:, :],
            ),
            dim=2,
        )
        self.save_output.clear(x.device)
        return feat

    def get_resnet_feature(self, x):
        """
        Gets the intermediate features from the ResNet50 backbone.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The intermediate features.
        """
        self.resnet50(x)
        feat = torch.cat(
            (
                self.save_output.outputs[x.device][0],
                self.save_output.outputs[x.device][1],
                self.save_output.outputs[x.device][2],
            ),
            dim=1,
        )
        self.save_output.clear(x.device)
        return feat

    def regress_score(self, dis, ref):
        """
        Computes the quality score for a distorted and reference image pair.

        Args:
            - dis (torch.Tensor): The distorted image.
            - ref (torch.Tensor): The reference image.

        Returns:
            torch.Tensor: The quality score.
        """
        self.resnet50.eval()
        self.vit.eval()
        dis = self.preprocess(dis)
        ref = self.preprocess(ref)

        vit_dis = self.get_vit_feature(dis)
        vit_ref = self.get_vit_feature(ref)

        B, N, C = vit_ref.shape
        H, W = 28, 28
        vit_ref = vit_ref.transpose(1, 2).view(B, C, H, W)
        vit_dis = vit_dis.transpose(1, 2).view(B, C, H, W)

        cnn_dis = self.get_resnet_feature(dis)
        cnn_ref = self.get_resnet_feature(ref)
        cnn_dis = self.deform_net(cnn_dis, vit_ref)
        cnn_ref = self.deform_net(cnn_ref, vit_ref)

        score = self.regressor(vit_dis, vit_ref, cnn_dis, cnn_ref)

        return score

    def forward(self, x, y):
        """
        Computes the quality score for a batch of distorted and reference image pairs.

        Args:
            - x (torch.Tensor): The batch of distorted images.
            - y (torch.Tensor): The batch of reference images.

        Returns:
            torch.Tensor: The quality scores.
        """
        bsz = x.shape[0]

        if self.crops > 1 and not self.training:
            x, y = random_crop([x, y], self.crop_size, self.crops)
            score = self.regress_score(x, y)
            score = score.reshape(bsz, self.crops, 1)
            score = score.mean(dim=1)
        else:
            score = self.regress_score(x, y)
        return score
