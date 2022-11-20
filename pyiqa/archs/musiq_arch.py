r"""MUSIQ model.

Implemented by: Chaofeng Chen (https://github.com/chaofengc)

Refer to:
    Official code from: https://github.com/google-research/google-research/tree/master/musiq

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .arch_util import dist_to_mos, load_pretrained_network
from .arch_util import ExactPadding2d, excact_padding_2d
from pyiqa.utils.registry import ARCH_REGISTRY
from pyiqa.data.multiscale_trans_util import get_multiscale_patches

default_model_urls = {
    'ava':
    'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/musiq_ava_ckpt-e8d3f067.pth',
    'koniq10k':
    'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/musiq_koniq_ckpt-e95806b9.pth',
    'spaq':
    'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/musiq_spaq_ckpt-358bb6af.pth',
    'paq2piq':
    'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/musiq_paq2piq_ckpt-364c0c84.pth',
    'imagenet_pretrain':
    'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/musiq_imagenet_pretrain-51d9b0a5.pth',
}


class StdConv(nn.Conv2d):
    """
    Reference: https://github.com/joe-siyuan-qiao/WeightStandardization
    """

    def forward(self, x):
        # implement same padding
        x = excact_padding_2d(x, self.kernel_size, self.stride, mode='same')
        weight = self.weight
        weight = weight - weight.mean((1, 2, 3), keepdim=True)
        weight = weight / (weight.std((1, 2, 3), keepdim=True) + 1e-5)
        return F.conv2d(x, weight, self.bias, self.stride)


class Bottleneck(nn.Module):

    def __init__(self, inplanes, outplanes, stride=1):
        super().__init__()

        width = inplanes

        self.conv1 = StdConv(inplanes, width, 1, 1, bias=False)
        self.gn1 = nn.GroupNorm(32, width, eps=1e-4)
        self.conv2 = StdConv(width, width, 3, 1, bias=False)
        self.gn2 = nn.GroupNorm(32, width, eps=1e-4)
        self.conv3 = StdConv(width, outplanes, 1, 1, bias=False)
        self.gn3 = nn.GroupNorm(32, outplanes, eps=1e-4)

        self.relu = nn.ReLU(True)

        self.needs_projection = inplanes != outplanes or stride != 1
        if self.needs_projection:
            self.conv_proj = StdConv(inplanes, outplanes, 1, stride, bias=False)
            self.gn_proj = nn.GroupNorm(32, outplanes, eps=1e-4)

    def forward(self, x):
        identity = x
        if self.needs_projection:
            identity = self.gn_proj(self.conv_proj(identity))

        x = self.relu(self.gn1(self.conv1(x)))
        x = self.relu(self.gn2(self.conv2(x)))
        x = self.gn3(self.conv3(x))
        out = self.relu(x + identity)

        return out


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MultiHeadAttention(nn.Module):

    def __init__(self, dim, num_heads=6, bias=False, attn_drop=0., out_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.query = nn.Linear(dim, dim, bias=bias)
        self.key = nn.Linear(dim, dim, bias=bias)
        self.value = nn.Linear(dim, dim, bias=bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.out = nn.Linear(dim, dim)
        self.out_drop = nn.Dropout(out_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            mask_h = mask.reshape(B, 1, N, 1)
            mask_w = mask.reshape(B, 1, 1, N)
            mask2d = mask_h * mask_w
            attn = attn.masked_fill(mask2d == 0, -1e3)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.out(x)
        x = self.out_drop(x)
        return x


class TransformerBlock(nn.Module):

    def __init__(self,
                 dim,
                 mlp_dim,
                 num_heads,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim, eps=1e-6)
        self.attention = MultiHeadAttention(dim, num_heads, bias=True, attn_drop=attn_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim, eps=1e-6)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, inputs_masks):
        y = self.norm1(x)
        y = self.attention(y, inputs_masks)
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class AddHashSpatialPositionEmbs(nn.Module):
    """Adds learnable hash-based spatial embeddings to the inputs."""

    def __init__(self, spatial_pos_grid_size, dim):
        super().__init__()
        self.position_emb = nn.parameter.Parameter(torch.randn(1, spatial_pos_grid_size * spatial_pos_grid_size, dim))
        nn.init.normal_(self.position_emb, std=0.02)

    def forward(self, inputs, inputs_positions):
        return inputs + self.position_emb.squeeze(0)[inputs_positions.long()]


class AddScaleEmbs(nn.Module):
    """Adds learnable scale embeddings to the inputs."""

    def __init__(self, num_scales, dim):
        super().__init__()
        self.scale_emb = nn.parameter.Parameter(torch.randn(num_scales, dim))
        nn.init.normal_(self.scale_emb, std=0.02)

    def forward(self, inputs, inputs_scale_positions):
        return inputs + self.scale_emb[inputs_scale_positions.long()]


class TransformerEncoder(nn.Module):

    def __init__(
        self,
        input_dim,
        mlp_dim=1152,
        attention_dropout_rate=0.,
        dropout_rate=0,
        num_heads=6,
        num_layers=14,
        num_scales=3,
        spatial_pos_grid_size=10,
        use_scale_emb=True,
        use_sinusoid_pos_emb=False,
    ):
        super().__init__()
        self.use_scale_emb = use_scale_emb
        self.posembed_input = AddHashSpatialPositionEmbs(spatial_pos_grid_size, input_dim)
        self.scaleembed_input = AddScaleEmbs(num_scales, input_dim)

        self.cls = nn.parameter.Parameter(torch.zeros(1, 1, input_dim))
        self.dropout = nn.Dropout(dropout_rate)
        self.encoder_norm = nn.LayerNorm(input_dim, eps=1e-6)

        self.transformer = nn.ModuleDict()
        for i in range(num_layers):
            self.transformer[f'encoderblock_{i}'] = TransformerBlock(input_dim, mlp_dim, num_heads, dropout_rate,
                                                                     attention_dropout_rate)

    def forward(self, x, inputs_spatial_positions, inputs_scale_positions, inputs_masks):
        n, _, c = x.shape

        x = self.posembed_input(x, inputs_spatial_positions)
        if self.use_scale_emb:
            x = self.scaleembed_input(x, inputs_scale_positions)

        cls_token = self.cls.repeat(n, 1, 1)
        x = torch.cat([cls_token, x], dim=1)

        cls_mask = torch.ones((n, 1)).to(inputs_masks)
        inputs_mask = torch.cat([cls_mask, inputs_masks], dim=1)
        x = self.dropout(x)

        for k, m in self.transformer.items():
            x = m(x, inputs_mask)
        x = self.encoder_norm(x)

        return x


@ARCH_REGISTRY.register()
class MUSIQ(nn.Module):
    r"""

    Evaluation:
        - n_crops: currently only test with 1 crop evaluation

    Reference:
        Ke, Junjie, Qifei Wang, Yilin Wang, Peyman Milanfar, and Feng Yang.
        "Musiq: Multi-scale image quality transformer." In Proceedings of the
        IEEE/CVF International Conference on Computer Vision (ICCV), pp. 5148-5157. 2021.
    """

    def __init__(
        self,
        patch_size=32,
        num_class=1,
        hidden_size=384,
        mlp_dim=1152,
        attention_dropout_rate=0.,
        dropout_rate=0,
        num_heads=6,
        num_layers=14,
        num_scales=3,
        spatial_pos_grid_size=10,
        use_scale_emb=True,
        use_sinusoid_pos_emb=False,
        pretrained=True,
        pretrained_model_path=None,
        # data opts
        longer_side_lengths=[224, 384],
        max_seq_len_from_original_res=-1,
    ):
        super(MUSIQ, self).__init__()

        resnet_token_dim = 64
        self.patch_size = patch_size

        self.data_preprocess_opts = {
            'patch_size': patch_size,
            'patch_stride': patch_size,
            'hse_grid_size': spatial_pos_grid_size,
            'longer_side_lengths': longer_side_lengths,
            'max_seq_len_from_original_res': max_seq_len_from_original_res
        }

        # set num_class to 10 if pretrained model used AVA dataset
        # if not specified pretrained dataset, use AVA for default
        if pretrained_model_path is None and pretrained:
            url_key = 'ava' if isinstance(pretrained, bool) else pretrained
            num_class = 10 if url_key == 'ava' else num_class
            pretrained_model_path = default_model_urls[url_key]

        self.conv_root = StdConv(3, resnet_token_dim, 7, 2, bias=False)
        self.gn_root = nn.GroupNorm(32, resnet_token_dim, eps=1e-6)
        self.root_pool = nn.Sequential(
            nn.ReLU(True),
            ExactPadding2d(3, 2, mode='same'),
            nn.MaxPool2d(3, 2),
        )

        token_patch_size = patch_size // 4
        self.block1 = Bottleneck(resnet_token_dim, resnet_token_dim * 4)

        self.embedding = nn.Linear(resnet_token_dim * 4 * token_patch_size**2, hidden_size)
        self.transformer_encoder = TransformerEncoder(hidden_size, mlp_dim, attention_dropout_rate, dropout_rate,
                                                      num_heads, num_layers, num_scales, spatial_pos_grid_size,
                                                      use_scale_emb, use_sinusoid_pos_emb)

        if num_class > 1:
            self.head = nn.Sequential(
                nn.Linear(hidden_size, num_class),
                nn.Softmax(dim=-1),
            )
        else:
            self.head = nn.Linear(hidden_size, num_class)

        if pretrained_model_path is not None:
            load_pretrained_network(self, pretrained_model_path, True)

    def forward(self, x, return_mos=True, return_dist=False):
        if not self.training:
            # normalize inputs to [-1, 1] as the official code
            x = (x - 0.5) * 2
            x = get_multiscale_patches(x, **self.data_preprocess_opts)

        assert len(x.shape) in [3, 4]
        if len(x.shape) == 4:
            b, num_crops, seq_len, dim = x.shape
            x = x.reshape(b * num_crops, seq_len, dim)
        else:
            b, seq_len, dim = x.shape
            num_crops = 1

        inputs_spatial_positions = x[:, :, -3]
        inputs_scale_positions = x[:, :, -2]
        inputs_masks = x[:, :, -1].bool()
        x = x[:, :, :-3]

        x = x.reshape(-1, 3, self.patch_size, self.patch_size)
        x = self.conv_root(x)
        x = self.gn_root(x)
        x = self.root_pool(x)
        x = self.block1(x)
        # to match tensorflow channel order
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(b, seq_len, -1)
        x = self.embedding(x)
        x = self.transformer_encoder(x, inputs_spatial_positions, inputs_scale_positions, inputs_masks)
        q = self.head(x[:, 0])

        q = q.reshape(b, num_crops, -1)
        q = q.mean(dim=1)  # for multiple crops evaluation
        mos = dist_to_mos(q)

        return_list = []
        if return_mos:
            return_list.append(mos)
        if return_dist:
            return_list.append(q)

        if len(return_list) > 1:
            return return_list
        else:
            return return_list[0]
