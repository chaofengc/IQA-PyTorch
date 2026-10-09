r"""TReS model.

Reference:
    No-Reference Image Quality Assessment via Transformers, Relative Ranking, and Self-Consistency.
    S. Alireza Golestaneh, Saba Dadsetan, Kris M. Kitani
    WACV2022

Official code: https://github.com/isalirezag/TReS
"""

import math
import copy
from typing import Optional
import numpy as np

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import torchvision.models as models

from .arch_util import load_pretrained_network
from pyiqa.utils.registry import ARCH_REGISTRY
from .arch_util import random_crop, uniform_crop
from pyiqa.archs.arch_util import get_url_from_name


default_model_urls = {
    'koniq': get_url_from_name('tres_koniq-f0502926.pth'),
    'flive': get_url_from_name('tres_flive-09b0de5b.pth'),
}


def _get_activation_fn(activation):
    """Return activation callable by name.

    Args:
        activation (str): Activation name. Supported values are ``'relu'``,
            ``'gelu'``, and ``'glu'``.

    Returns:
        Callable: Activation function from :mod:`torch.nn.functional`.

    Raises:
        RuntimeError: If ``activation`` is unsupported.
    """
    if activation == 'relu':
        return F.relu
    if activation == 'gelu':
        return F.gelu
    if activation == 'glu':
        return F.glu
    raise RuntimeError(f'activation should be relu/gelu, not {activation}.')


def _get_clones(module, N):
    """Create ``N`` deep-copied modules in a :class:`~torch.nn.ModuleList`."""
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Transformer(nn.Module):
    """Transformer encoder used by TReS to aggregate multiscale features.

    Args:
        d_model (int): Feature dimension.
        nhead (int): Number of attention heads.
        num_encoder_layers (int): Number of encoder layers.
        num_decoder_layers (int): Unused legacy argument kept for compatibility.
        dim_feedforward (int): Hidden dimension in feed-forward blocks.
        dropout (float): Dropout ratio.
        activation (str): Activation function name.
        normalize_before (bool): Whether to apply pre-normalization.
        return_intermediate_dec (bool): Legacy compatibility argument.
    """
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation='relu',
        normalize_before=False,
        return_intermediate_dec=False,
    ):
        """Initialize the transformer and configure its layers, parameters, and optional pretrained state.

        Args:
            d_model: Transformer embedding width.
            nhead: Number of attention heads.
            num_encoder_layers: Number of items, stages, tokens, or channels configured for this operation.
            num_decoder_layers: Number of items, stages, tokens, or channels configured for this operation.
            dim_feedforward: Hidden width of the transformer feed-forward network.
            dropout: Dropout probability used by the projection or transformer layers.
            activation: Activation-layer constructor or activation mode.
            normalize_before: Whether to apply layer normalization before the attention/feed-forward operations.
            return_intermediate_dec: Whether to return intermediate decoder-layer outputs.
        """
        super().__init__()

        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )

        self._reset_parameters()

        self.nhead = nhead

    def _reset_parameters(self):
        """Perform the internal reset parameters operation used by transformer.

        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, pos_embed):
        # flatten NxCxHxW to HWxNxC
        """Apply the transformer computation to the provided activations.

        Args:
            src: Source sequence or feature tensor supplied to the transformer encoder.
            pos_embed: Positional embeddings added to token features.

        Returns:
            Transformed tensor or feature representation; shape follows the layer configuration.
        """
        bs, c, h, w = src.shape
        src2 = src
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed2 = pos_embed

        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)

        memory = self.encoder(src, pos=pos_embed)

        return memory.permute(1, 2, 0).view(bs, c, h, w)


class TransformerEncoder(nn.Module):
    """Transformer encoder that processes TReS image-patch features.

    """
    def __init__(self, encoder_layer, num_layers, norm=None):
        """Initialize the transformer encoder and configure its layers, parameters, and optional pretrained state.

        Args:
            encoder_layer: Encoder-layer instance to clone for the transformer stack.
            num_layers: Number of items, stages, tokens, or channels configured for this operation.
            norm: Normalization module or mode applied to the feature tensor.
        """
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        """Apply the stacked transformer encoder layers to patch features.

        Args:
            src: Source sequence or feature tensor supplied to the transformer encoder.
            mask: Optional mask controlling which positions participate in attention or aggregation.
            src_key_padding_mask: Optional mask controlling which positions participate in attention or aggregation.
            pos: Positional embedding or position indices associated with the input tokens.


        Returns:
            Encoded patch feature tensor.
        """
        output = src

        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer used to model interactions among TReS image patches.

    """
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation='relu',
        normalize_before=False,
    ):
        """Initialize the transformer encoder layer and configure its layers, parameters, and optional pretrained state.

        Args:
            d_model: Transformer embedding width.
            nhead: Number of attention heads.
            dim_feedforward: Hidden width of the transformer feed-forward network.
            dropout: Dropout probability used by the projection or transformer layers.
            activation: Activation-layer constructor or activation mode.
            normalize_before: Whether to apply layer normalization before the attention/feed-forward operations.
        """
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        """Perform the with pos embed operation for transformer encoder layer.

        Args:
            tensor: Input tensor to transform or initialize.
            pos: Positional embedding or position indices associated with the input tokens.

        Returns:
            Computed result; type and shape follow the supplied inputs and model configuration.
        """
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        """Perform the forward post operation for transformer encoder layer.

        Args:
            src: Source sequence or feature tensor supplied to the transformer encoder.
            src_mask: Optional mask controlling which positions participate in attention or aggregation.
            src_key_padding_mask: Optional mask controlling which positions participate in attention or aggregation.
            pos: Positional embedding or position indices associated with the input tokens.

        Returns:
            Computed result; type and shape follow the supplied inputs and model configuration.
        """
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(
            q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        """Perform the forward pre operation for transformer encoder layer.

        Args:
            src: Source sequence or feature tensor supplied to the transformer encoder.
            src_mask: Optional mask controlling which positions participate in attention or aggregation.
            src_key_padding_mask: Optional mask controlling which positions participate in attention or aggregation.
            pos: Positional embedding or position indices associated with the input tokens.

        Returns:
            Computed result; type and shape follow the supplied inputs and model configuration.
        """
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(
            q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        """Apply the transformer encoder layer computation to the provided activations.

        Args:
            src: Source sequence or feature tensor supplied to the transformer encoder.
            src_mask: Optional mask controlling which positions participate in attention or aggregation.
            src_key_padding_mask: Optional mask controlling which positions participate in attention or aggregation.
            pos: Positional embedding or position indices associated with the input tokens.

        Returns:
            Transformed tensor or feature representation; shape follows the layer configuration.
        """
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class PositionEmbeddingSine(nn.Module):
    """Sine-cosine positional encoding for 2D feature maps.

    This implementation is adapted from DETR-style positional encoding and
    generates a fixed embedding tensor with shape ``(N, C, H, W)``.
    """

    def __init__(
        self, num_pos_feats=64, temperature=10000, normalize=False, scale=None
    ):
        """Initialize the position embedding sine and configure its layers, parameters, and optional pretrained state.

        Args:
            num_pos_feats: Number of items, stages, tokens, or channels configured for this operation.
            temperature: Positive temperature used to scale similarity logits.
            normalize: Whether to normalize compared feature vectors before computing similarity.
            scale: Scaling factor or embedding scale configured by this model.
        """
        super().__init__()
        self.num_pos_feats = num_pos_feats  # 128 in dert
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError('normalize should be True if scale is passed')
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    # def forward(self, tensor_list: NestedTensor):
    def forward(self, tensor_val):
        """Compute the TReS prediction for the supplied image or image pair.

        Args:
            tensor_val: Tensor or scalar value used to initialize the parameter.

        Returns:
            Quality score tensor, generally one score per input image or image pair.
        """
        x = tensor_val
        # mask = tensor_list.mask # it has 1 for padding, so the important stuff is 0
        mask = torch.gt(torch.zeros(x.shape), 0).to(x.device)[:, 0, :, :]
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (
            2 * (torch.div(dim_t, 2, rounding_mode='trunc')) / self.num_pos_feats
        )

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class L2pooling(nn.Module):
    """L2 pooling with Hann-window smoothing."""
    def __init__(self, filter_size=5, stride=1, channels=None, pad_off=0):
        """Initialize the l2pooling and configure its layers, parameters, and optional pretrained state.

        Args:
            filter_size: Requested spatial or sequence dimension, compatible with the model configuration.
            stride: Spatial stride for the convolutional stage.
            channels: Number of feature channels processed by the filter.
            pad_off: Spatial offset applied to the padding operation.
        """
        super(L2pooling, self).__init__()
        self.padding = (filter_size - 2) // 2
        self.stride = stride
        self.channels = channels
        a = np.hanning(filter_size)[1:-1]
        g = torch.Tensor(a[:, None] * a[None, :])
        g = g / torch.sum(g)
        self.register_buffer(
            'filter', g[None, None, :, :].repeat((self.channels, 1, 1, 1))
        )

    def forward(self, input):
        """Apply the l2pooling computation to the provided activations.

        Args:
            input: Image or feature tensor supplied to the feature extractor.

        Returns:
            Transformed tensor or feature representation; shape follows the layer configuration.
        """
        input = input**2
        out = F.conv2d(
            input,
            self.filter,
            stride=self.stride,
            padding=self.padding,
            groups=input.shape[1],
        )
        return (out + 1e-12).sqrt()


@ARCH_REGISTRY.register()
class TReS(nn.Module):
    """TReS no-reference IQA model.

    Args:
        network (str): ResNet backbone name.
        train_dataset (str): Dataset key used to choose default checkpoint.
        nheadt (int): Number of transformer attention heads.
        num_encoder_layerst (int): Number of transformer encoder blocks.
        dim_feedforwardt (int): Transformer feed-forward hidden size.
        test_sample (int): Number of uniform crops during evaluation.
        default_mean (list[float]): Input normalization mean in RGB order.
        default_std (list[float]): Input normalization std in RGB order.
        pretrained (bool): Whether to load default pretrained checkpoint.
        pretrained_model_path (str | None): Optional local checkpoint path.

    Example:
        >>> metric = TReS(train_dataset='koniq')
        >>> x = torch.rand(1, 3, 512, 512)
        >>> score = metric(x)
    """
    def __init__(
        self,
        network='resnet50',
        train_dataset='koniq',
        nheadt=16,
        num_encoder_layerst=2,
        dim_feedforwardt=64,
        test_sample=50,
        default_mean=[0.485, 0.456, 0.406],
        default_std=[0.229, 0.224, 0.225],
        pretrained=True,
        pretrained_model_path=None,
    ):
        """Initialize the tre s and configure its layers, parameters, and optional pretrained state.

        Args:
            network: Feature-extractor backbone instance or architecture.
            train_dataset: Training dataset/model variant used to select its weights or preprocessing.
            nheadt: Number of attention heads in the transformer.
            num_encoder_layerst: Number of items, stages, tokens, or channels configured for this operation.
            dim_feedforwardt: Hidden width of the transformer feed-forward network.
            test_sample: Number of test-time image samples or crops used to estimate the score.
            default_mean: Per-channel input normalization means.
            default_std: Per-channel input normalization standard deviations.
            pretrained: Whether to initialize or load pretrained weights.
            pretrained_model_path: Optional local checkpoint path; ``None`` selects the implementation default.
        """
        super().__init__()

        self.test_sample = test_sample

        self.L2pooling_l1 = L2pooling(channels=256)
        self.L2pooling_l2 = L2pooling(channels=512)
        self.L2pooling_l3 = L2pooling(channels=1024)
        self.L2pooling_l4 = L2pooling(channels=2048)

        if network == 'resnet50':
            dim_modelt = 3840
            self.model = models.resnet50(weights='IMAGENET1K_V1')
        elif network == 'resnet34':
            self.model = models.resnet34(weights='IMAGENET1K_V1')
            dim_modelt = 960
            self.L2pooling_l1 = L2pooling(channels=64)
            self.L2pooling_l2 = L2pooling(channels=128)
            self.L2pooling_l3 = L2pooling(channels=256)
            self.L2pooling_l4 = L2pooling(channels=512)
        elif network == 'resnet18':
            self.model = models.resnet18(weights='IMAGENET1K_V1')
            dim_modelt = 960
            self.L2pooling_l1 = L2pooling(channels=64)
            self.L2pooling_l2 = L2pooling(channels=128)
            self.L2pooling_l3 = L2pooling(channels=256)
            self.L2pooling_l4 = L2pooling(channels=512)

        self.dim_modelt = dim_modelt

        nheadt = nheadt
        num_encoder_layerst = num_encoder_layerst
        dim_feedforwardt = dim_feedforwardt
        ddropout = 0.5
        normalize = True

        self.transformer = Transformer(
            d_model=dim_modelt,
            nhead=nheadt,
            num_encoder_layers=num_encoder_layerst,
            dim_feedforward=dim_feedforwardt,
            normalize_before=normalize,
            dropout=ddropout,
        )

        self.position_embedding = PositionEmbeddingSine(dim_modelt // 2, normalize=True)

        self.fc2 = nn.Linear(dim_modelt, self.model.fc.in_features)
        self.fc = nn.Linear(self.model.fc.in_features * 2, 1)

        self.ReLU = nn.ReLU()
        self.avg7 = nn.AvgPool2d((7, 7))
        self.avg8 = nn.AvgPool2d((8, 8))
        self.avg4 = nn.AvgPool2d((4, 4))
        self.avg2 = nn.AvgPool2d((2, 2))

        self.drop2d = nn.Dropout(p=0.1)
        self.consistency = nn.L1Loss()

        self.default_mean = torch.Tensor(default_mean).view(1, 3, 1, 1)
        self.default_std = torch.Tensor(default_std).view(1, 3, 1, 1)

        if pretrained_model_path is not None:
            load_pretrained_network(self, pretrained_model_path, True)
        elif pretrained:
            load_pretrained_network(self, default_model_urls[train_dataset], True)

    def forward_backbone(self, model, x):
        # See note [TorchScript super()]
        """Perform the forward backbone operation for tre s.

        Args:
            model: Model instance whose parameters or attributes are used by this helper.
            x: Input tensor or activation; image entry points generally use ``(B, C, H, W)`` layout, while internal layers may use other layouts.

        Returns:
            Computed result; type and shape follow the supplied inputs and model configuration.
        """
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)

        x = model.layer1(x)
        l1 = x
        x = model.layer2(x)
        l2 = x
        x = model.layer3(x)
        l3 = x
        x = model.layer4(x)
        l4 = x
        x = model.avgpool(x)
        x = torch.flatten(x, 1)
        x = model.fc(x)

        return x, l1, l2, l3, l4

    def forward(self, x):
        """Compute TReS quality score.

        Args:
            x (torch.Tensor): Input tensor with shape ``(N, 3, H, W)``.

        Returns:
            torch.Tensor: Predicted quality score with shape ``(N, 1)``.
        """
        x = (x - self.default_mean.to(x)) / self.default_std.to(x)
        bsz = x.shape[0]

        if self.training:
            x = random_crop(x, 224, 1)
            num_patches = 1
        else:
            x = uniform_crop(x, 224, self.test_sample)
            num_patches = self.test_sample

        self.pos_enc_1 = self.position_embedding(
            torch.ones(1, self.dim_modelt, 7, 7).to(x)
        )
        self.pos_enc = self.pos_enc_1.repeat(x.shape[0], 1, 1, 1).contiguous()

        out, layer1, layer2, layer3, layer4 = self.forward_backbone(self.model, x)

        layer1_t = self.avg8(
            self.drop2d(self.L2pooling_l1(F.normalize(layer1, dim=1, p=2)))
        )
        layer2_t = self.avg4(
            self.drop2d(self.L2pooling_l2(F.normalize(layer2, dim=1, p=2)))
        )
        layer3_t = self.avg2(
            self.drop2d(self.L2pooling_l3(F.normalize(layer3, dim=1, p=2)))
        )
        layer4_t = self.drop2d(self.L2pooling_l4(F.normalize(layer4, dim=1, p=2)))
        layers = torch.cat((layer1_t, layer2_t, layer3_t, layer4_t), dim=1)

        out_t_c = self.transformer(layers, self.pos_enc)
        out_t_o = torch.flatten(self.avg7(out_t_c), start_dim=1)
        out_t_o = self.fc2(out_t_o)
        layer4_o = self.avg7(layer4)
        layer4_o = torch.flatten(layer4_o, start_dim=1)
        predictionQA = self.fc(
            torch.flatten(torch.cat((out_t_o, layer4_o), dim=1), start_dim=1)
        )

        fout, flayer1, flayer2, flayer3, flayer4 = self.forward_backbone(
            self.model, torch.flip(x, [3])
        )
        flayer1_t = self.avg8(self.L2pooling_l1(F.normalize(flayer1, dim=1, p=2)))
        flayer2_t = self.avg4(self.L2pooling_l2(F.normalize(flayer2, dim=1, p=2)))
        flayer3_t = self.avg2(self.L2pooling_l3(F.normalize(flayer3, dim=1, p=2)))
        flayer4_t = self.L2pooling_l4(F.normalize(flayer4, dim=1, p=2))
        flayers = torch.cat((flayer1_t, flayer2_t, flayer3_t, flayer4_t), dim=1)
        fout_t_c = self.transformer(flayers, self.pos_enc)
        fout_t_o = torch.flatten(self.avg7(fout_t_c), start_dim=1)
        fout_t_o = self.fc2(fout_t_o)
        flayer4_o = self.avg7(flayer4)
        flayer4_o = torch.flatten(flayer4_o, start_dim=1)
        fpredictionQA = self.fc(
            torch.flatten(torch.cat((fout_t_o, flayer4_o), dim=1), start_dim=1)
        )

        consistloss1 = self.consistency(out_t_c, fout_t_c.detach())
        consistloss2 = self.consistency(layer4, flayer4.detach())
        consistloss = 1 * (consistloss1 + consistloss2)

        predictionQA = predictionQA.reshape(bsz, num_patches, 1)
        predictionQA = predictionQA.mean(dim=1)

        if self.training:
            return predictionQA, consistloss
        else:
            return predictionQA
