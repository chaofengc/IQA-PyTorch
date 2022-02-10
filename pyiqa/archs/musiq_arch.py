import torch
import torch.nn as nn
import torch.nn.functional as F
from pyiqa.utils.registry import ARCH_REGISTRY

from torchvision.models.resnet import Bottleneck

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
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


class TransformerBlock(nn.Module):
    def __init__(self, dim, mlp_dim, num_heads, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, bias=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, inputs_masks):
        y = self.norm1(x)
        y, attn = self.attn(y, y, y, inputs_masks)
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class AddHashSpatialPositionEmbs(nn.Module):
    """Adds learnable hash-based spatial embeddings to the inputs."""
    def __init__(self, spatial_pos_grid_size, dim):
        super().__init__()
        self.position_emb = nn.parameter.Parameter(torch.randn(1, 
                    spatial_pos_grid_size * spatial_pos_grid_size, dim))
        nn.init.normal_(self.position_emb, std=0.02)

    def forward(self, inputs, inputs_positions):
        return inputs + self.position_emb[inputs_positions]


class AddScaleEmbs(nn.Module):
    """Adds learnable scale embeddings to the inputs."""
    def __init__(self, num_scales, dim):
        super().__init__()
        self.scale_emb = nn.parameter.Parameter(torch.randn(1, 
                    num_scales, dim))
        nn.init.normal_(self.scale_emb, std=0.02)

    def forward(self, inputs, inputs_scale_positions):
        return inputs + self.scale_emb[inputs_scale_positions]


class TransformerEncoder(nn.Module):
    def __init__(self,
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
        self.add_hash_pos_emb = AddHashSpatialPositionEmbs(spatial_pos_grid_size, input_dim)
        self.add_scale_emb = AddScaleEmbs(num_scales, input_dim)

        self.cls_token = nn.parameter.Parameter(torch.zeros(1, 1, input_dim))
        self.dropout = nn.Dropout(dropout_rate)

        self.transformer_layers = []
        for i in range(num_layers): 
            self.transformer_layers.append(
                TransformerBlock(input_dim, mlp_dim, num_heads, dropout_rate, attention_dropout_rate)
            )
    
    def forward(self, x):
        return x


@ARCH_REGISTRY.register()
class MUSIQ(nn.Module):
    """MUSIQ model proposed by 

    Ke, Junjie, Qifei Wang, Yilin Wang, Peyman Milanfar, and Feng Yang. 
    "Musiq: Multi-scale image quality transformer." 
    In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 5148-5157. 2021.

    Ref url: https://github.com/google-research/google-research/tree/master/musiq

    """
    def __init__(self, 
                 pretrained_model_path=None, 
                 ):
        super(MUSIQ, self).__init__()

        resnet_token_dim = 64 

        self.conv_first = nn.Sequential(
            nn.Conv2d(3, resnet_token_dim, 7, 2, bias=False),
            nn.GroupNorm(32, resnet_token_dim),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2, padding=1)
        )

        self.resnet_encoder = nn.Sequential(
            *[Bottleneck(resnet_token_dim, resnet_token_dim) for i in range(5)] 
        )

        self.transformer_encoder = TransformerEncoder()
        self.fc_q = nn.Linear(resnet_token_dim, 1)

        if pretrained_model_path is not None:
            self.load_pretrained_network(pretrained_model_path)

    def load_pretrained_network(self, model_path):
        print(f'Loading pretrained model from {model_path}')
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))['state_dict']
        self.net.load_state_dict(state_dict, strict=True) 

    def forward(self, x):
        h  = self.conv1(x)

        h1 = F.max_pool2d(h, (h.size(-2), h.size(-1)))
        h2 = -F.max_pool2d(-h, (h.size(-2), h.size(-1)))
        h  = torch.cat((h1, h2), 1)  # max-min pooling
        h  = h.squeeze(3).squeeze(2)

        h  = F.relu(self.fc1(h))
        h  = self.dropout(h)
        h  = F.relu(self.fc2(h))

        q  = self.fc3(h)
        return q
        