import math

import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm

from pyiqa.utils.download_util import load_file_from_url

# --------------------------------------------
# IQA utils 
# --------------------------------------------

def dist_to_mos(dist_score: torch.Tensor) -> torch.Tensor:
    """Convert distribution prediction to mos score.
    For datasets with detailed score labels, such as AVA  

    Args:
        dist_score (tensor): (*, C), C is the class number

    Output:
        mos_score (tensor): (*, 1)
    """
    num_classes = dist_score.shape[-1]
    mos_score = dist_score * torch.arange(1, num_classes + 1).to(dist_score)
    mos_score = mos_score.sum(dim=-1, keepdim=True)
    return mos_score 

# --------------------------------------------
# Common utils 
# --------------------------------------------

def load_pretrained_network(net, model_path, strict=True):
    if model_path.startswith('https://') or model_path.startswith('http://'):
        model_path = load_file_from_url(model_path)
    print(f'Loading pretrained model {net.__class__.__name__} from {model_path}')
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    net.load_state_dict(state_dict, strict=strict) 

@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    r"""Initialize network weights.
    
    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0.
        kwargs (dict): Other arguments for initialization function.

    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


class SymmetricPad2d(nn.Module):
    r"""Symmetric pad 2d for pytorch.

    Args:
        pad (int or tuple): (pad_left, pad_right, pad_top, pad_bottom)

    """
    def __init__(self, pad):
        super(SymmetricPad2d, self).__init__()
        if isinstance(pad, int):
            self.pad_l = self.pad_r = self.pad_t = self.pad_b = pad
        elif isinstance(pad, tuple):
            assert len(pad) == 4, f"tuple pad should have format (pad_left, pad_right, pad_top, pad_bottom), but got {pad}"
            self.pad_l, self.pad_r, self.pad_t, self.pad_b = pad

    def forward(self, x):

        _, _, h, w = x.shape
        sym_h = torch.flip(x, [2]) 
        sym_w = torch.flip(x, [3]) 
        sym_hw = torch.flip(x, [2, 3])

        row1 = torch.cat((sym_hw, sym_h, sym_hw), dim=3) 
        row2 = torch.cat((sym_w, x, sym_w), dim=3) 
        row3 = torch.cat((sym_hw, sym_h, sym_hw), dim=3) 
        
        whole_map = torch.cat((row1, row2, row3), dim=2)

        pad_x = whole_map[:, :, 
                h - self.pad_t: 2*h + self.pad_b:,
                w - self.pad_l: 2*w + self.pad_r:,
                ]

        return pad_x


def simple_sample_padding2d(x, kernel, stride, dilation=1):
    r"""Simple same padding for 4D tensor. Only support int kernel, stride and dilation.

    Args:
        x (tensor): The input. Shape :math:`(N, C, H, W)`.
        kernel (int): Kernel size.
        stride (int): Stride size.
        dilation (int): Dilation size, default with 1.

    """
    assert len(x.shape) == 4, f'Only support 4D tensor input, but got {x.shape}'
    b,c,h,w = x.shape
    h2 = math.ceil(h / stride)
    w2 = math.ceil(w / stride)
    pad_row = (h2 - 1) * stride + (kernel - 1) * dilation + 1 - h
    pad_col = (w2 - 1) * stride + (kernel - 1) * dilation + 1 - w
    x = F.pad(x, (pad_col//2, pad_col - pad_col//2, pad_row//2, pad_row - pad_row//2))
    return x


class SimpleSamePadding2d(nn.Module):
    r"""Simple same padding for 4D tensor. Only support int kernel, stride and dilation.

    Args:
        kernel (int): kernel size.
        stride (int): stride size.
        dilation (int): dilation size, default with 1.

    """
    def __init__(self, kernel, stride, dilation=1):
        super().__init__()
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
    
    def forward(self, x):
        return simple_sample_padding2d(x, self.kernel, self.stride, self.dilation)
