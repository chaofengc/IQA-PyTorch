from pyexpat import model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.deform_conv import DeformConv2d
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import timm
from timm.models.vision_transformer import Block
from timm.models.resnet import BasicBlock,Bottleneck

from pyiqa.utils.registry import ARCH_REGISTRY
from pyiqa.archs.arch_util import load_pretrained_network, default_init_weights, to_2tuple, ExactPadding2d


class IQARegression(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.conv_enc = nn.Conv2d(in_channels=320*6, out_channels=config.d_hidn, kernel_size=1)
        self.conv_dec = nn.Conv2d(in_channels=320*6, out_channels=config.d_hidn, kernel_size=1)

        self.transformer = Transformer(self.config)
        self.projection = nn.Sequential(
            nn.Linear(self.config.d_hidn, self.config.d_MLP_head, bias=False),
            nn.ReLU(),
            nn.Linear(self.config.d_MLP_head, self.config.n_output, bias=False)
        )
    
    def forward(self, enc_inputs, enc_inputs_embed, dec_inputs, dec_inputs_embed):
        # batch x (320*6) x 29 x 29 -> batch x 256 x 29 x 29
        enc_inputs_embed = self.conv_enc(enc_inputs_embed)
        dec_inputs_embed = self.conv_dec(dec_inputs_embed)
        # batch x 256 x 29 x 29 -> batch x 256 x (29*29)
        b, c, h, w = enc_inputs_embed.size()
        enc_inputs_embed = torch.reshape(enc_inputs_embed, (b, c, h*w))
        enc_inputs_embed = enc_inputs_embed.permute(0, 2, 1)
        # batch x 256 x (29*29) -> batch x (29*29) x 256
        dec_inputs_embed = torch.reshape(dec_inputs_embed, (b, c, h*w))
        dec_inputs_embed = dec_inputs_embed.permute(0, 2, 1)

        # (bs, n_dec_seq+1, d_hidn), [(bs, n_head, n_enc_seq+1, n_enc_seq+1)], [(bs, n_head, n_dec_seq+1, n_dec_seq+1)], [(bs, n_head, n_dec_seq+1, n_enc_seq+1)]
        dec_outputs, enc_self_attn_probs, dec_self_attn_probs, dec_enc_attn_probs = self.transformer(enc_inputs, enc_inputs_embed, dec_inputs, dec_inputs_embed)
        
        # (bs, n_dec_seq+1, d_hidn) -> (bs, d_hidn)
        # dec_outputs, _ = torch.max(dec_outputs, dim=1)    # original transformer
        dec_outputs = dec_outputs[:, 0, :]                  # in the IQA paper
        # dec_outputs = torch.mean(dec_outputs, dim=1)      # general idea
        
        # (bs, n_output)
        pred = self.projection(dec_outputs)
        
        return pred


""" transformer """
class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.encoder = Encoder(self.config)
        self.decoder = Decoder(self.config)
    
    def forward(self, enc_inputs, enc_inputs_embed, dec_inputs, dec_inputs_embed):
        # (bs, n_enc_seq, d_hidn), [(bs, n_head, n_enc_seq, n_enc_seq)]
        enc_outputs, enc_self_attn_probs = self.encoder(enc_inputs, enc_inputs_embed)
        # (bs, n_seq, d_hidn), [(bs, n_head, n_dec_seq, n_dec_seq)], [(bs, n_head, n_dec_seq, n_enc_seq)]
        dec_outputs, dec_self_attn_probs, dec_enc_attn_probs = self.decoder(dec_inputs, dec_inputs_embed, enc_inputs, enc_outputs)
        
        # (bs, n_dec_seq, n_dec_vocab), [(bs, n_head, n_enc_seq, n_enc_seq)], [(bs, n_head, n_dec_seq, n_dec_seq)], [(bs, n_head, n_dec_seq, n_enc_seq)]
        return dec_outputs, enc_self_attn_probs, dec_self_attn_probs, dec_enc_attn_probs


""" encoder """
class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # fixed position embedding
        # sinusoid_table = torch.FloatTensor(get_sinusoid_encoding_table(self.config.n_enc_seq+1, self.config.d_hidn))
        # self.pos_emb = nn.Embedding.from_pretrained(sinusoid_table, freeze=True)

        # learnable position embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.config.n_enc_seq+1, self.config.d_hidn))
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.config.d_hidn))
        self.dropout = nn.Dropout(self.config.emb_dropout)

        self.layers = nn.ModuleList([EncoderLayer(self.config) for _ in range(self.config.n_layer)])
    
    def forward(self, inputs, inputs_embed):  
        # inputs: batch x (len_seq+1) / inputs_embed: batch x len_seq x n_feat  
        b, n, _ = inputs_embed.shape

        # positions: batch x (len_seq+1)
        positions = torch.arange(inputs.size(1), device=inputs.device, dtype=torch.int64).expand(inputs.size(0), inputs.size(1)).contiguous() + 1
        pos_mask = inputs.eq(self.config.i_pad)
        positions.masked_fill_(pos_mask, 0)
        
        # outputs: batch x (len_seq+1) x n_feat
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, inputs_embed), dim=1)
        x += self.pos_embedding
        # x += self.pos_emb(positions)
        outputs = self.dropout(x)        

        # (bs, n_enc_seq+1, n_enc_seq+1)
        attn_mask = get_attn_pad_mask(inputs, inputs, self.config.i_pad)

        attn_probs = []
        for layer in self.layers:
            # (bs, n_enc_seq+1, d_hidn), (bs, n_head, n_enc_seq+1, n_enc_seq+1)
            outputs, attn_prob = layer(outputs, attn_mask)
            attn_probs.append(attn_prob)
        
        # (bs, n_enc_seq+1, d_hidn), [(bs, n_head, n_enc_seq+1, n_enc_seq+1)]
        return outputs, attn_probs


""" encoder layer """
class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.self_attn = MultiHeadAttention(self.config)
        self.layer_norm1 = nn.LayerNorm(self.config.d_hidn, eps=self.config.layer_norm_epsilon)
        self.pos_ffn = PoswiseFeedForwardNet(self.config)
        self.layer_norm2 = nn.LayerNorm(self.config.d_hidn, eps=self.config.layer_norm_epsilon)
    
    def forward(self, inputs, attn_mask):
        # (bs, n_enc_seq, d_hidn), (bs, n_head, n_enc_seq, n_enc_seq)
        att_outputs, attn_prob = self.self_attn(inputs, inputs, inputs, attn_mask)
        att_outputs = self.layer_norm1(inputs + att_outputs)
        
        # (bs, n_enc_seq, d_hidn)
        ffn_outputs = self.pos_ffn(att_outputs)
        ffn_outputs = self.layer_norm2(ffn_outputs + att_outputs)
        
        # (bs, n_enc_seq, d_hidn), (bs, n_head, n_enc_seq, n_enc_seq)
        return ffn_outputs, attn_prob


def get_sinusoid_encoding_table(n_seq, d_hidn):
    def cal_angle(position, i_hidn):
        return position / np.power(10000, 2 * (i_hidn // 2) / d_hidn)
    def get_posi_angle_vec(position):
        return [cal_angle(position, i_hidn) for i_hidn in range(d_hidn)]

    sinusoid_table = np.array([get_posi_angle_vec(i_seq) for i_seq in range(n_seq)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # even index sin 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # odd index cos

    return sinusoid_table


""" attention pad mask """
def get_attn_pad_mask(seq_q, seq_k, i_pad):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(i_pad)
    pad_attn_mask= pad_attn_mask.unsqueeze(1).expand(batch_size, len_q, len_k)
    return pad_attn_mask


""" multi head attention """
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.W_Q = nn.Linear(self.config.d_hidn, self.config.n_head * self.config.d_head)
        self.W_K = nn.Linear(self.config.d_hidn, self.config.n_head * self.config.d_head)
        self.W_V = nn.Linear(self.config.d_hidn, self.config.n_head * self.config.d_head)
        self.scaled_dot_attn = ScaledDotProductAttention(self.config)
        self.linear = nn.Linear(self.config.n_head * self.config.d_head, self.config.d_hidn)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, Q, K, V, attn_mask):
        batch_size = Q.size(0)
        
        # (bs, n_head, n_q_seq, d_head)
        q_s = self.W_Q(Q).view(batch_size, -1, self.config.n_head, self.config.d_head).transpose(1,2)
        # (bs, n_head, n_k_seq, d_head)
        k_s = self.W_K(K).view(batch_size, -1, self.config.n_head, self.config.d_head).transpose(1,2)
        # (bs, n_head, n_v_seq, d_head)
        v_s = self.W_V(V).view(batch_size, -1, self.config.n_head, self.config.d_head).transpose(1,2)

        # (bs, n_head, n_q_seq, n_k_seq)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.config.n_head, 1, 1)

        # (bs, n_head, n_q_seq, d_head), (bs, n_head, n_q_seq, n_k_seq)
        context, attn_prob = self.scaled_dot_attn(q_s, k_s, v_s, attn_mask)
        # (bs, n_head, n_q_seq, h_head * d_head)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.config.n_head * self.config.d_head)
        # (bs, n_head, n_q_seq, e_embd)
        output = self.linear(context)
        output = self.dropout(output)
        
        # (bs, n_q_seq, d_hidn), (bs, n_head, n_q_seq, n_k_seq)
        return output, attn_prob


""" scale dot product attention """
class ScaledDotProductAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.dropout)
        self.scale = 1 / (self.config.d_head ** 0.5)
    
    def forward(self, Q, K, V, attn_mask):
        # (bs, n_head, n_q_seq, n_k_seq)
        scores = torch.matmul(Q, K.transpose(-1, -2))
        scores = scores.mul_(self.scale)
        scores.masked_fill_(attn_mask, -1e9)
        # (bs, n_head, n_q_seq, n_k_seq)
        attn_prob = nn.Softmax(dim=-1)(scores)
        attn_prob = self.dropout(attn_prob)
        # (bs, n_head, n_q_seq, d_v)
        context = torch.matmul(attn_prob, V)
        
        # (bs, n_head, n_q_seq, d_v), (bs, n_head, n_q_seq, n_v_seq)
        return context, attn_prob


""" feed forward """
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.conv1 = nn.Conv1d(in_channels=self.config.d_hidn, out_channels=self.config.d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=self.config.d_ff, out_channels=self.config.d_hidn, kernel_size=1)
        self.active = F.gelu
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, inputs):
        # (bs, d_ff, n_seq)
        output = self.conv1(inputs.transpose(1, 2))
        output = self.active(output)
        # (bs, n_seq, d_hidn)
        output = self.conv2(output).transpose(1, 2)
        output = self.dropout(output)
        
        # (bs, n_seq, d_hidn)
        return output


""" decoder """
class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.pos_embedding = nn.Parameter(torch.randn(1, self.config.n_enc_seq+1, self.config.d_hidn))
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.config.d_hidn))
        self.dropout = nn.Dropout(self.config.emb_dropout)

        self.layers = nn.ModuleList([DecoderLayer(self.config) for _ in range(self.config.n_layer)])
    
    def forward(self, dec_inputs, dec_inputs_embed, enc_inputs, enc_outputs):
        # enc_inputs: batch x (len_seq+1) / enc_outputs: batch x (len_seq+1) x n_feat
        # dec_inputs: batch x (len_seq+1) / dec_inputs_embed: batch x len_seq x n_feat
        b, n, _ = dec_inputs_embed.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, dec_inputs_embed), dim=1)
        x += self.pos_embedding[:, :(n+1)]
        # (bs, n_dec_seq+1, d_hidn)
        dec_outputs = self.dropout(x)

        # (bs, n_dec_seq+1, n_dec_seq+1)
        dec_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs, self.config.i_pad)
        # (bs, n_dec_seq+1, n_dec_seq+1)
        dec_attn_decoder_mask = get_attn_decoder_mask(dec_inputs)
        # (bs, n_dec_seq+1, n_dec_seq+1)
        dec_self_attn_mask = torch.gt((dec_attn_pad_mask + dec_attn_decoder_mask), 0)
        # (bs, n_dec_seq+1, n_enc_seq+1)
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs, self.config.i_pad)

        self_attn_probs, dec_enc_attn_probs = [], []
        for layer in self.layers:
            # (bs, n_dec_seq+1, d_hidn), (bs, n_dec_seq+1, n_dec_seq+1), (bs, n_dec_seq+1, n_enc_seq+1)
            dec_outputs, self_attn_prob, dec_enc_attn_prob = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            self_attn_probs.append(self_attn_prob)
            dec_enc_attn_probs.append(dec_enc_attn_prob)
        
        # (bs, n_dec_seq+1, d_hidn), [(bs, n_dec_seq+1, n_dec_seq+1)], [(bs, n_dec_seq+1, n_enc_seq+1)]
        return dec_outputs, self_attn_probs, dec_enc_attn_probs


""" decoder layer """
class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.self_attn = MultiHeadAttention(self.config)
        self.layer_norm1 = nn.LayerNorm(self.config.d_hidn, eps=self.config.layer_norm_epsilon)
        self.dec_enc_attn = MultiHeadAttention(self.config)
        self.layer_norm2 = nn.LayerNorm(self.config.d_hidn, eps=self.config.layer_norm_epsilon)
        self.pos_ffn = PoswiseFeedForwardNet(self.config)
        self.layer_norm3 = nn.LayerNorm(self.config.d_hidn, eps=self.config.layer_norm_epsilon)
    
    def forward(self, dec_inputs, enc_outputs, self_attn_mask, dec_enc_attn_mask):
        # (bs, n_dec_seq, d_hidn), (bs, n_head, n_dec_seq, n_dec_seq)
        self_att_outputs, self_attn_prob = self.self_attn(dec_inputs, dec_inputs, dec_inputs, self_attn_mask)
        self_att_outputs = self.layer_norm1(dec_inputs + self_att_outputs)
        # (bs, n_dec_seq, d_hidn), (bs, n_head, n_dec_seq, n_enc_seq)
        dec_enc_att_outputs, dec_enc_attn_prob = self.dec_enc_attn(self_att_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_enc_att_outputs = self.layer_norm2(self_att_outputs + dec_enc_att_outputs)
        # (bs, n_dec_seq, d_hidn)
        ffn_outputs = self.pos_ffn(dec_enc_att_outputs)
        ffn_outputs = self.layer_norm3(dec_enc_att_outputs + ffn_outputs)
        
        # (bs, n_dec_seq, d_hidn), (bs, n_head, n_dec_seq, n_dec_seq), (bs, n_head, n_dec_seq, n_enc_seq)
        return ffn_outputs, self_attn_prob, dec_enc_attn_prob


""" attention decoder mask """
def get_attn_decoder_mask(seq):
    subsequent_mask = torch.ones_like(seq).unsqueeze(-1).expand(seq.size(0), seq.size(1), seq.size(1))
    subsequent_mask = subsequent_mask.triu(diagonal=1) # upper triangular part of a matrix(2-D)
    return subsequent_mask



def random_crop(x, y, crop_size, crop_num):
    b, c, h, w = x.shape
    ch, cw = to_2tuple(crop_size)

    crops_x = []
    crops_y = []
    for i in range(crop_num):
        sh = np.random.randint(0, h - ch) 
        sw = np.random.randint(0, w - cw)
        crops_x.append(x[..., sh: sh + ch, sw: sw + cw])
        crops_y.append(y[..., sh: sh + ch, sw: sw + cw])
    crops_x = torch.stack(crops_x, dim=1)
    crops_y = torch.stack(crops_y, dim=1)
    return crops_x.reshape(b * crop_num, c, ch, cw), crops_y.reshape(b * crop_num, c, ch, cw)


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
    def __init__(self, patch_size=8, in_channels=768*5, cnn_channels=256*3, out_channels=256*3):
        super().__init__()
        #in_channels, out_channels, kernel_size, stride, padding
        self.d_hidn = 512
        if patch_size == 8:
            stride = 1
        else:
            stride = 2
        self.conv_offset = nn.Conv2d(in_channels, 2*3*3, 3, 1, 1)
        self.deform = DeformConv2d(cnn_channels, out_channels, 3, 1, 1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=self.d_hidn, kernel_size=3,padding=1,stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.d_hidn, out_channels=out_channels, kernel_size=3, padding=1,stride=stride)
        )

    def forward(self, cnn_feat, vit_feat):
        vit_feat = F.interpolate(vit_feat, size=cnn_feat.shape[-2:], mode="nearest")
        offset = self.conv_offset(vit_feat)
        deform_feat = self.deform(cnn_feat, offset)
        deform_feat = self.conv1(deform_feat)
        
        return deform_feat

class Pixel_Prediction(nn.Module):
    def __init__(self, inchannels=768*5+256*3, outchannels=256, d_hidn=1024):
        super().__init__()
        self.d_hidn = d_hidn
        self.down_channel = nn.Conv2d(inchannels, outchannels, kernel_size=1)
        self.feat_smoothing = nn.Sequential(
            nn.Conv2d(in_channels=256*3, out_channels=self.d_hidn, kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.d_hidn, out_channels=512, kernel_size=3, padding=1)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3,padding=1), 
            nn.ReLU()
        )
        self.conv_attent =  nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1),
            nn.Sigmoid()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1),
        )
    
    def forward(self,f_dis, f_ref, cnn_dis, cnn_ref):
        f_dis = torch.cat((f_dis,cnn_dis),1)
        f_ref = torch.cat((f_ref,cnn_ref),1)
        f_dis = self.down_channel(f_dis)
        f_ref = self.down_channel(f_ref)

        f_cat = torch.cat((f_dis - f_ref, f_dis, f_ref), 1)

        feat_fused = self.feat_smoothing(f_cat)
        feat = self.conv1(feat_fused)
        f = self.conv(feat)
        w = self.conv_attent(feat)
        pred = (f*w).sum(dim=-1).sum(dim=-1)/w.sum(dim=-1).sum(dim=-1)

        return pred


@ARCH_REGISTRY.register()
class IQT(nn.Module):
    def __init__(self,
                 num_crop=20,
                 config_dataset='live',
                 default_mean = timm.data.IMAGENET_INCEPTION_MEAN,
                 default_std = timm.data.IMAGENET_INCEPTION_STD,
                 pretrained=False,
                 pretrained_model_path=None,
                 ):
        super().__init__()

        self.backbone = timm.create_model('inception_resnet_v2', pretrained=True)
        self.fix_network(self.backbone)

        class Config:
            def __init__(self, dataset=config_dataset) -> None:
                if dataset in ['live', 'csiq', 'tid']:
                    # model for PIPAL (NTIRE2021 Challenge)
                    self.n_enc_seq = 29*29 # feature map dimension (H x W) from backbone, this size is related to crop_size
                    self.n_dec_seq = 29*29 # feature map dimension (H x W) from backbone
                    self.n_layer = 2 # number of encoder/decoder layers
                    self.d_hidn = 256 # input channel (C) of encoder / decoder (input: C x N)
                    self.i_pad = 0
                    self.d_ff = 1024 # feed forward hidden layer dimension
                    self.d_MLP_head = 512 # hidden layer of final MLP 
                    self.n_head=4 # number of head (in multi-head attention)
                    self.d_head = 256 # input channel (C) of each head (input: C x N) -> same as d_hidn
                    self.dropout = 0.1 # dropout ratio of transformer
                    self.emb_dropout = 0.1 # dropout ratio of input embedding
                    self.layer_norm_epsilon = 1e-12
                    self.n_output = 1 # dimension of final prediction
                    self.crop_size = 256 # input image crop size

                elif dataset == 'pipal':
                    # model for PIPAL (NTIRE2021 Challenge)
                    self.n_enc_seq = 21*21 # feature map dimension (H x W) from backbone, this size is related to crop_size
                    self.n_dec_seq = 21*21 # feature map dimension (H x W) from backbone
                    self.n_layer = 1 # number of encoder/decoder layers
                    self.d_hidn = 128 # input channel (C) of encoder / decoder (input: C x N)
                    self.i_pad = 0
                    self.d_ff = 1024 # feed forward hidden layer dimension
                    self.d_MLP_head = 128 # hidden layer of final MLP 
                    self.n_head=4 # number of head (in multi-head attention)
                    self.d_head = 128 # input channel (C) of each head (input: C x N) -> same as d_hidn
                    self.dropout = 0.1 # dropout ratio of transformer
                    self.emb_dropout = 0.1 # dropout ratio of input embedding
                    self.layer_norm_epsilon = 1e-12
                    self.n_output = 1 # dimension of final prediction
                    self.crop_size = 192 # input image crop size

        config = Config()
        self.config = config

        self.register_buffer('enc_inputs', torch.ones(1, config.n_enc_seq+1))
        self.register_buffer('dec_inputs', torch.ones(1, config.n_dec_seq+1))
        
        self.regressor = IQARegression(config)

        # register hook to get intermediate features
        self.init_saveoutput()

        self.default_mean = torch.Tensor(default_mean).view(1, 3, 1, 1)
        self.default_std = torch.Tensor(default_std).view(1, 3, 1, 1)

        if pretrained_model_path is not None:
            load_pretrained_network(self, pretrained_model_path, False, weight_keys='params')

        self.eps = 1e-12
        self.crops = num_crop 
        self.crop_size = config.crop_size 
    
    def init_saveoutput(self):
        self.save_output = SaveOutput()
        hook_handles = []
        for layer in self.backbone.modules():
            if type(layer).__name__ == 'Mixed_5b':
                handle = layer.register_forward_hook(self.save_output)
                hook_handles.append(handle)
            elif type(layer).__name__ == 'Block35':
                handle = layer.register_forward_hook(self.save_output)
        
    def fix_network(self, model):
        for p in model.parameters():
            p.requires_grad = False    

    def preprocess(self, x):
        x = (x - self.default_mean.to(x)) / self.default_std.to(x)
        return x

    @torch.no_grad() 
    def get_backbone_feature(self, x):
        self.backbone(x)
        feat = torch.cat(
            (
                self.save_output.outputs[x.device][0],
                self.save_output.outputs[x.device][2],
                self.save_output.outputs[x.device][4],
                self.save_output.outputs[x.device][6],
                self.save_output.outputs[x.device][8],
                self.save_output.outputs[x.device][10],
            ),
            dim=1
        )
        self.save_output.clear(x.device)
        return feat
    
    def regress_score(self, dis, ref):
        assert dis.shape[-1] == dis.shape[-2]== self.config.crop_size, f'Input shape should be {self.config.crop_size, self.config.crop_size} but got {dis.shape[2:]}'
        
        self.backbone.eval()

        dis = self.preprocess(dis)
        ref = self.preprocess(ref)

        feat_dis = self.get_backbone_feature(dis)
        feat_ref = self.get_backbone_feature(ref)
        feat_diff = feat_ref - feat_dis

        score = self.regressor(self.enc_inputs, feat_diff, self.dec_inputs, feat_ref)

        return score
           
    def forward(self, x, y):
        bsz = x.shape[0]

        if self.crops > 1 and not self.training:
            x, y = random_crop(x, y, self.crop_size, self.crops)
            score = self.regress_score(x, y)
            score = score.reshape(bsz, self.crops, 1)
            score = score.mean(dim=1)
        else:
            score = self.regress_score(x, y)
        return score
