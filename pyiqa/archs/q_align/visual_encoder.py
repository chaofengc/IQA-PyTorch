"""Vision transformer and visual abstractor modules used by mPLUG-Owl2."""

import math
from typing import Optional, Tuple, Union

from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import prune_linear_layer

try:
    from transformers.pytorch_utils import find_pruneable_heads_and_indices
except ImportError:
    def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
        """Compatibility fallback for Transformers>=5 where this helper was removed."""
        mask = torch.ones(n_heads, head_size)
        heads = set(heads) - already_pruned_heads
        for head in heads:
            head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(mask.numel())[mask].long()
        return heads, index

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

def get_abs_pos(abs_pos, tgt_size):
    # abs_pos: L, C
    # tgt_size: M
    # return: M, C
    """Resize a square grid of learned absolute positional embeddings to the requested token-grid size.

    Args:
        abs_pos: Flattened square-grid absolute embeddings shaped (source_tokens, channels).
        tgt_size: Target token count; its integer square root defines the output square-grid side.

    Returns:
        Embeddings shaped (target_grid_tokens, channels), or the original tensor when sizes match."""
    src_size = int(math.sqrt(abs_pos.size(0)))
    tgt_size = int(math.sqrt(tgt_size))
    dtype = abs_pos.dtype

    if src_size != tgt_size:
        return F.interpolate(
            abs_pos.float().reshape(1, src_size, src_size, -1).permute(0, 3, 1, 2),
            size=(tgt_size, tgt_size),
            mode="bicubic",
            align_corners=False,
        ).permute(0, 2, 3, 1).flatten(0, 2).to(dtype=dtype)
    else:
        return abs_pos

# https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/pos_embed.py#L20
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """Combine height and width sine/cosine embeddings for a supplied two-axis grid.

    Args:
        embed_dim: Embedding width; sine/cosine helpers require an even width.
        grid: Two-axis coordinate grid with height and width positions.

    Returns:
        A NumPy array shaped (grid_points, embed_dim)."""
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb



class MplugOwlVisionEmbeddings(nn.Module):
    """Creates class-token and patch embeddings with learned positional encodings."""
    def __init__(self, config):
        """Initialize MplugOwlVisionEmbeddings from its configuration and constructor arguments.

        Args:
            config: Configuration object supplying architecture dimensions and options.

        Returns:
            None; initializes module state and parameters."""
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_size))

        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2

        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, self.hidden_size))

        self.pre_layernorm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        """Run the MplugOwlVisionEmbeddings computation for the supplied inputs.

        Args:
            pixel_values: Image pixels shaped (batch, channels, height, width).

        Returns:
            A tensor shaped (batch, patches + 1, hidden_size), including the class token."""
        batch_size = pixel_values.size(0)
        image_embeds = self.patch_embed(pixel_values)
        image_embeds = image_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.cls_token.expand(batch_size, 1, -1).to(image_embeds.dtype)
        embeddings = torch.cat([class_embeds, image_embeds], dim=1)
        embeddings = embeddings + self.position_embedding[:, : embeddings.size(1)].to(image_embeds.dtype)
        embeddings = self.pre_layernorm(embeddings)
        return embeddings



class MplugOwlVisionAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        """Initialize MplugOwlVisionAttention from its configuration and constructor arguments.

        Args:
            config: Configuration object supplying architecture dimensions and options.

        Returns:
            None; initializes module state and parameters."""
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = nn.Dropout(config.attention_dropout)

        self.query_key_value = nn.Linear(self.hidden_size, 3 * self.hidden_size)
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        """Reshape projected states to batch, head, sequence, and head-dimension order.

        Args:
            tensor: Projected states shaped (batch, sequence, hidden_size).
            seq_len: Number of sequence positions represented by tensor.
            bsz: Batch size represented by tensor.

        Returns:
            Projected states shaped (batch, heads, sequence, head_dim)."""
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, seq_len, embed_dim = hidden_states.size()

        mixed_qkv = self.query_key_value(hidden_states)

        mixed_qkv = mixed_qkv.reshape(bsz, seq_len, self.num_heads, 3, embed_dim // self.num_heads).permute(
            3, 0, 2, 1, 4
        )  # [3, b, np, sq, hn]
        query_states, key_states, value_states = (
            mixed_qkv[0],
            mixed_qkv[1],
            mixed_qkv[2],
        )
        # if self.config.use_flash_attn and flash_attn_func is not None:
        if False:
            # [b*sq, np, hn]
            query_states = query_states.permute(0, 2, 1, 3).contiguous()
            query_states = query_states.view(query_states.size(0) * query_states.size(1), query_states.size(2), -1)

            key_states = key_states.permute(0, 2, 1, 3).contiguous()
            key_states = key_states.view(key_states.size(0) * key_states.size(1), key_states.size(2), -1)

            value_states = value_states.permute(0, 2, 1, 3).contiguous()
            value_states = value_states.view(value_states.size(0) * value_states.size(1), value_states.size(2), -1)

            cu_seqlens = torch.arange(
                0, (bsz + 1) * seq_len, step=seq_len, dtype=torch.int32, device=query_states.device
            )

            context_layer = flash_attn_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens,
                cu_seqlens,
                seq_len,
                seq_len,
                self.dropout if self.training else 0.0,
                softmax_scale=self.scale,
                causal=False,
                return_attn_probs=False,
            )
            # [b*sq, np, hn] => [b, sq, np, hn]
            context_layer = context_layer.view(bsz, seq_len, context_layer.size(1), context_layer.size(2))
        else:
            # Take the dot product between "query" and "key" to get the raw attention scores.
            attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2))

            attention_scores = attention_scores * self.scale

            # Normalize the attention scores to probabilities.
            attention_probs = torch.softmax(attention_scores, dim=-1)

            # This is actually dropping out entire tokens to attend to, which might
            # seem a bit unusual, but is taken from the original Transformer paper.
            attention_probs = self.dropout(attention_probs)

            # Mask heads if we want to
            if head_mask is not None:
                attention_probs = attention_probs * head_mask

            context_layer = torch.matmul(attention_probs, value_states).permute(0, 2, 1, 3)

        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.reshape(new_context_layer_shape)

        output = self.dense(context_layer)

        outputs = (output, attention_probs) if output_attentions else (output, None)

        return outputs


class QuickGELU(nn.Module):
    """Module implementing the QuickGELU activation used by the vision MLP."""
    def forward(self, x: torch.Tensor):
        """Run the QuickGELU computation for the supplied inputs.

        Args:
            x: Input tensor to transform, activate, or encode.

        Returns:
            The elementwise QuickGELU activation, with the same shape as x."""
        return x * torch.sigmoid(1.702 * x)


class MplugOwlMLP(nn.Module):
    """Two-layer QuickGELU feed-forward block in the vision transformer."""
    def __init__(self, config):
        """Initialize MplugOwlMLP from its configuration and constructor arguments.

        Args:
            config: Configuration object supplying architecture dimensions and options.

        Returns:
            None; initializes module state and parameters."""
        super().__init__()
        self.config = config
        self.activation_fn = QuickGELU()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run the MplugOwlMLP computation for the supplied inputs.

        Args:
            hidden_states: Input activations, generally shaped (batch, sequence, hidden_size).

        Returns:
            A tensor with the same leading dimensions and hidden size as its input."""
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class MplugOwlVisionEncoderLayer(nn.Module):
    """Vision-transformer block with self-attention and an MLP residual path."""
    def __init__(self, config):
        """Initialize MplugOwlVisionEncoderLayer from its configuration and constructor arguments.

        Args:
            config: Configuration object supplying architecture dimensions and options.

        Returns:
            None; initializes module state and parameters."""
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = MplugOwlVisionAttention(config)
        self.input_layernorm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)
        self.mlp = MplugOwlMLP(config)
        self.post_attention_layernorm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            head_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        hidden_states = hidden_states + residual

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs
    
    
class MplugOwlVisionEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`MplugOwlVisionEncoderLayer`].

    Args:
        config (`MplugOwlVisionConfig`):
            The corresponding vision configuration for the `MplugOwlEncoder`.
    """

    def __init__(self, config):
        """Initialize MplugOwlVisionEncoder from its configuration and constructor arguments.

        Args:
            config: Configuration object supplying architecture dimensions and options.

        Returns:
            None; initializes module state and parameters."""
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([MplugOwlVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = True

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Embedded representation of the inputs. Should be float, not int tokens.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    """Wrap a module in a checkpoint-compatible forward callable.

                    Args:
                        module: Layer module executed by this wrapper.

                    Returns:
                        A callable suitable for checkpointing the given module."""
                    def custom_forward(*inputs):
                        """Call the wrapped layer with the captured cache and attention-output setting.

                        Args:
                            inputs: Positional tensors passed to the checkpointed layer.

                        Returns:
                            The wrapped module’s forward result."""
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(encoder_layer),
                    hidden_states,
                    attention_mask,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class MplugOwlVisionModel(PreTrainedModel):
    """Vision transformer that encodes image pixels into patch-level hidden states."""
    main_input_name = "pixel_values"
    _no_split_modules = ["MplugOwlVisionEncoderLayer"]

    def __init__(self, config):
        """Initialize MplugOwlVisionModel from its configuration and constructor arguments.

        Args:
            config: Configuration object supplying architecture dimensions and options.

        Returns:
            None; initializes module state and parameters."""
        super().__init__(config)
        self.config = config
        self.hidden_size = config.hidden_size

        self.embeddings = MplugOwlVisionEmbeddings(config)
        self.encoder = MplugOwlVisionEncoder(config)
        self.post_layernorm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)

        self.post_init()


    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.post_layernorm(last_hidden_state)

        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def get_input_embeddings(self):
        """Return the image patch-and-class-token embedding module.

        Returns:
            The image patch-and-class-token embedding module."""
        return self.embeddings


class MplugOwlVisualAbstractorMLP(nn.Module):
    """Gated feed-forward block used in the visual abstractor."""
    def __init__(self, config):
        """Initialize MplugOwlVisualAbstractorMLP from its configuration and constructor arguments.

        Args:
            config: Configuration object supplying architecture dimensions and options.

        Returns:
            None; initializes module state and parameters."""
        super().__init__()
        self.config = config
        in_features = config.hidden_size
        self.act = nn.SiLU()

        self.w1 = nn.Linear(in_features, config.intermediate_size)
        self.w2 = nn.Linear(config.intermediate_size, in_features)
        self.w3 = nn.Linear(in_features, config.intermediate_size)
        self.ffn_ln = nn.LayerNorm(config.intermediate_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run the MplugOwlVisualAbstractorMLP computation for the supplied inputs.

        Args:
            hidden_states: Input activations, generally shaped (batch, sequence, hidden_size).

        Returns:
            A tensor with the same leading dimensions and hidden size as its input."""
        hidden_states = self.act(self.w1(hidden_states)) * self.w3(hidden_states)
        hidden_states = self.ffn_ln(hidden_states)
        hidden_states = self.w2(hidden_states)
        return hidden_states


class MplugOwlVisualAbstractorMultiHeadAttention(nn.Module):
    """Cross-attention from learnable queries to vision-encoder states."""
    def __init__(self, config):
        """Initialize MplugOwlVisualAbstractorMultiHeadAttention from its configuration and constructor arguments.

        Args:
            config: Configuration object supplying architecture dimensions and options.

        Returns:
            None; initializes module state and parameters."""
        super().__init__()
        self.config = config
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention heads (%d)"
                % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.encoder_hidden_size, self.all_head_size)
        self.value = nn.Linear(config.encoder_hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.save_attention = False
        
#         self.q_pos_embed = nn.Parameter(
#             torch.from_numpy(get_1d_sincos_pos_embed_from_grid(config.hidden_size, np.arange(config.num_learnable_queries, dtype=np.float32))).float()
#         ).requires_grad_(False)
#         grids = config.grid_size
#         self.k_pos_embed = nn.Parameter(
#             torch.from_numpy(get_2d_sincos_pos_embed(config.hidden_size, grids, cls_token=True)).float()
#         ).requires_grad_(False)
        grids = config.grid_size
        self.register_buffer(
            'q_pos_embed', 
            torch.from_numpy(get_1d_sincos_pos_embed_from_grid(config.hidden_size, np.arange(config.num_learnable_queries, dtype=np.float32))).float()
        )
        self.register_buffer(
            'k_pos_embed', 
            torch.from_numpy(get_2d_sincos_pos_embed(config.hidden_size, grids, cls_token=True)).float()
        )
        

    def save_attn_gradients(self, attn_gradients):
        """Store the gradient captured for the most recently saved attention map.

        Args:
            attn_gradients: Gradient tensor for the saved attention probabilities.

        Returns:
            None; stores the supplied attention gradient on the instance."""
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        """Return the attention-map gradient previously stored by the module.

        Returns:
            The stored attention gradient."""
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        """Store the attention probability tensor for later inspection.

        Args:
            attention_map: Attention probability tensor to retain for inspection or gradient analysis.

        Returns:
            None; stores the supplied attention map on the instance."""
        self.attention_map = attention_map

    def get_attention_map(self):
        """Return the attention probability tensor previously stored by the module.

        Returns:
            The stored attention map."""
        return self.attention_map

    def transpose_for_scores(self, x):
        """Split the final projection dimension into attention heads and move heads before sequence.

        Args:
            x: Input tensor to transform, activate, or encode.

        Returns:
            A tensor shaped (batch, heads, sequence, head_size)."""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        
        """Run the MplugOwlVisualAbstractorMultiHeadAttention computation for the supplied inputs.

        Args:
            hidden_states: Input activations, generally shaped (batch, sequence, hidden_size).
            attention_mask: Optional mask aligned with sequence positions; convention follows the selected attention backend.
            head_mask: Optional per-head mask applied to attention probabilities.
            encoder_hidden_states: Vision features shaped (batch, source_tokens, encoder_hidden_size) for cross-attention.
            encoder_attention_mask: Optional mask for encoder key/value positions.
            past_key_value: Optional per-layer key/value cache used for incremental decoding.
            output_attentions: Whether attention probabilities should be returned when supported.

        Returns:
            A tuple containing query context states and optional attention probabilities."""
        qk_pos_embed = torch.cat([self.q_pos_embed, self.k_pos_embed], dim = 0).unsqueeze(0).to(dtype=hidden_states.dtype)
        
        key_layer = self.transpose_for_scores(self.key(encoder_hidden_states + qk_pos_embed))
        value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
        attention_mask = encoder_attention_mask

        mixed_query_layer = self.query(hidden_states + self.q_pos_embed.unsqueeze(0).to(dtype=hidden_states.dtype))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        if self.save_attention:
            self.save_attention_map(attention_probs)
            attention_probs.register_hook(self.save_attn_gradients)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs_dropped = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs_dropped = attention_probs_dropped * head_mask

        context_layer = torch.matmul(attention_probs_dropped, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        outputs = outputs + (past_key_value,)
        return outputs


class MplugOwlVisualAbstractorCrossOutput(nn.Module):
    """Projection, normalization, and residual output block for abstractor attention."""
    def __init__(self, config):
        """Initialize MplugOwlVisualAbstractorCrossOutput from its configuration and constructor arguments.

        Args:
            config: Configuration object supplying architecture dimensions and options.

        Returns:
            None; initializes module state and parameters."""
        super().__init__()
        dim = config.hidden_size
        self.out_proj = nn.Linear(dim, dim, bias=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MplugOwlVisualAbstractorMLP(config)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        """Run the MplugOwlVisualAbstractorCrossOutput computation for the supplied inputs.

        Args:
            hidden_states: Input activations, generally shaped (batch, sequence, hidden_size).
            input_tensor: Residual query states added to the projected attention context.

        Returns:
            Residual-updated query states with the same shape as the input query tensor."""
        input_tensor = input_tensor + self.out_proj(hidden_states)
        input_tensor = input_tensor + self.mlp(self.norm2(input_tensor))
        return input_tensor


class MplugOwlVisualAbstractorAttention(nn.Module):
    """Visual abstractor attention block with query/key normalization."""
    def __init__(self, config):
        """Initialize MplugOwlVisualAbstractorAttention from its configuration and constructor arguments.

        Args:
            config: Configuration object supplying architecture dimensions and options.

        Returns:
            None; initializes module state and parameters."""
        super().__init__()
        self.attention = MplugOwlVisualAbstractorMultiHeadAttention(config)
        self.output = MplugOwlVisualAbstractorCrossOutput(config)
        self.pruned_heads = set()
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.normk = nn.LayerNorm(config.hidden_size)

    def prune_heads(self, heads):
        """Remove the requested attention heads and update the module’s projection dimensions.

        Args:
            heads: Attention-head indices to prune from this layer.

        Returns:
            None; replaces projection layers and records the pruned heads."""
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.out_proj, index, dim=1)

        # Update hyper params and store pruned heads
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # HACK we apply norm on q and k
        """Run the MplugOwlVisualAbstractorAttention computation for the supplied inputs.

        Args:
            hidden_states: Input activations, generally shaped (batch, sequence, hidden_size).
            attention_mask: Optional mask aligned with sequence positions; convention follows the selected attention backend.
            head_mask: Optional per-head mask applied to attention probabilities.
            encoder_hidden_states: Vision features shaped (batch, source_tokens, encoder_hidden_size) for cross-attention.
            encoder_attention_mask: Optional mask for encoder key/value positions.
            past_key_value: Optional per-layer key/value cache used for incremental decoding.
            output_attentions: Whether attention probabilities should be returned when supported.

        Returns:
            A tuple containing updated query states and optional attention outputs."""
        hidden_states = self.norm1(hidden_states)
        encoder_hidden_states = self.normk(encoder_hidden_states)
        encoder_hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)
        encoder_attention_mask = torch.cat([attention_mask, encoder_attention_mask], dim=-1)
        self_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        # add attentions if we output them
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class MplugOwlVisualAbstractorLayer(nn.Module):
    """One learnable-query cross-attention layer in the visual abstractor."""
    def __init__(self, config, layer_idx):
        """Initialize MplugOwlVisualAbstractorLayer from its configuration and constructor arguments.

        Args:
            config: Configuration object supplying architecture dimensions and options.
            layer_idx: Zero-based index of this layer in the abstractor stack.

        Returns:
            None; initializes module state and parameters."""
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1

        self.layer_idx = layer_idx

        self.crossattention = MplugOwlVisualAbstractorAttention(config)
        self.has_cross_attention = True

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        """Run the MplugOwlVisualAbstractorLayer computation for the supplied inputs.

        Args:
            hidden_states: Input activations, generally shaped (batch, sequence, hidden_size).
            attention_mask: Optional mask aligned with sequence positions; convention follows the selected attention backend.
            head_mask: Optional per-head mask applied to attention probabilities.
            encoder_hidden_states: Vision features shaped (batch, source_tokens, encoder_hidden_size) for cross-attention.
            encoder_attention_mask: Optional mask for encoder key/value positions.
            output_attentions: Whether attention probabilities should be returned when supported.

        Returns:
            A tuple containing the updated learnable-query states."""
        if encoder_hidden_states is None:
            raise ValueError("encoder_hidden_states must be given for cross-attention layers")
        cross_attention_outputs = self.crossattention(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            output_attentions=output_attentions,
        )
        query_attention_output = cross_attention_outputs[0]

        outputs = (query_attention_output,)
        return outputs


class MplugOwlVisualAbstractorEncoder(nn.Module):
    """Stack of visual abstractor cross-attention layers."""
    def __init__(self, config):
        """Initialize MplugOwlVisualAbstractorEncoder from its configuration and constructor arguments.

        Args:
            config: Configuration object supplying architecture dimensions and options.

        Returns:
            None; initializes module state and parameters."""
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [MplugOwlVisualAbstractorLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = True

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        """Run the MplugOwlVisualAbstractorEncoder computation for the supplied inputs.

        Args:
            hidden_states: Input activations, generally shaped (batch, sequence, hidden_size).
            attention_mask: Optional mask aligned with sequence positions; convention follows the selected attention backend.
            head_mask: Optional per-head mask applied to attention probabilities.
            encoder_hidden_states: Vision features shaped (batch, source_tokens, encoder_hidden_size) for cross-attention.
            encoder_attention_mask: Optional mask for encoder key/value positions.
            past_key_values: Optional collection of per-layer key/value caches.
            output_attentions: Whether attention probabilities should be returned when supported.
            output_hidden_states: Whether intermediate layer states should be included in the output.
            return_dict: Whether to return a Transformers ModelOutput instead of a tuple.

        Returns:
            A BaseModelOutput whose last_hidden_state contains final query states."""
        all_hidden_states = () if output_hidden_states else None

        for i in range(self.config.num_hidden_layers):
            layer_module = self.layers[i]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                def create_custom_forward(module):
                    """Wrap a module in a checkpoint-compatible forward callable.

                    Args:
                        module: Layer module executed by this wrapper.

                    Returns:
                        A callable suitable for checkpointing the given module."""
                    def custom_forward(*inputs):
                        """Call the wrapped layer with the captured cache and attention-output setting.

                        Args:
                            inputs: Positional tensors passed to the checkpointed layer.

                        Returns:
                            The wrapped module’s forward result."""
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]

        return BaseModelOutput(
            last_hidden_state=hidden_states,
        )


class MplugOwlVisualAbstractorModel(PreTrainedModel):
    """Maps vision-encoder tokens to a fixed set of language-space visual tokens."""
    _no_split_modules = ["MplugOwlVisualAbstractorLayer"]
    def __init__(self, config, language_hidden_size):
        """Initialize MplugOwlVisualAbstractorModel from its configuration and constructor arguments.

        Args:
            config: Configuration object supplying architecture dimensions and options.
            language_hidden_size: Output feature width matching the language model hidden dimension.

        Returns:
            None; initializes module state and parameters."""
        super().__init__(config)
        self.config = config

        self.encoder = MplugOwlVisualAbstractorEncoder(config)
        self.visual_fc = torch.nn.Linear(config.hidden_size, language_hidden_size)
        self.query_embeds = torch.nn.Parameter(torch.randn(1, config.num_learnable_queries, config.hidden_size))
        self.vit_eos = torch.nn.Parameter(torch.randn(1, 1, language_hidden_size))

        self.post_init()

    def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        """Compatibility helper for Transformers>=5 where PreTrainedModel.get_head_mask was removed."""
        if head_mask is None:
            return [None] * num_hidden_layers

        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        else:
            raise ValueError(f"head_mask must have dim 1 or 2, but got dim={head_mask.dim()}")

        head_mask = head_mask.to(dtype=self.dtype)
        if is_attention_chunked:
            head_mask = head_mask.unsqueeze(-1)
        return head_mask

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def get_extended_attention_mask(
        self,
        attention_mask: torch.Tensor,
        input_shape: Tuple[int],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (`Tuple[int]`):
                The shape of the input to the model.
            device: (`torch.device`):
                The device of the input to the model.

        Returns:
            `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - the model is an encoder, so make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(
        self,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of:
            shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`): Contains precomputed key and
            value hidden states of the attention blocks. Can be used to speed up decoding. If `past_key_values` are
            used, the user can optionally input only the last `decoder_input_ids` (those that don't have their past key
            value states given to this model) of shape `(batch_size, 1)` instead of all `decoder_input_ids` of shape
            `(batch_size, sequence_length)`.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        query_embeds = self.query_embeds.repeat(encoder_hidden_states.shape[0], 1, 1)
        embedding_output = query_embeds
        input_shape = embedding_output.size()[:-1]
        batch_size, seq_length = input_shape
        device = embedding_output.device

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask is None:
            attention_mask = torch.ones(
                (query_embeds.shape[0], query_embeds.shape[1]), dtype=torch.long, device=query_embeds.device
            )
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if encoder_hidden_states is not None:
            if type(encoder_hidden_states) == list:
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states[0].size()
            else:
                (
                    encoder_batch_size,
                    encoder_sequence_length,
                    _,
                ) = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)

            if type(encoder_attention_mask) == list:
                encoder_extended_attention_mask = [self.invert_attention_mask(mask) for mask in encoder_attention_mask]
            elif encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
                encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
            else:
                encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = sequence_output[:, 0, :]

        sequence_output = self.visual_fc(sequence_output)
        sequence_output = torch.cat([sequence_output, self.vit_eos.repeat(sequence_output.shape[0], 1, 1)], dim=1)

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
        )
