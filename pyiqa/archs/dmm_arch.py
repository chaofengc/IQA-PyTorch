r"""Debiased Mapping for Full-Reference Image Quality Assessment

@article{chen2025debiased,
  title={Debiased mapping for full-reference image quality assessment},
  author={Chen, Baoliang and Zhu, Hanwei and Zhu, Lingyu and Wang, Shanshe and Pan, Jingshan and Wang, Shiqi},
  journal={IEEE Transactions on Multimedia},
  year={2025},
  publisher={IEEE}
}

Reference:
    - Arxiv link: https://ieeexplore.ieee.org/abstract/document/10886996
    - Official Github: https://github.com/Baoliang93/DMM
"""

import torch
import torchvision
import torch.nn as nn
from collections import OrderedDict
from torchvision import  transforms
import torch.nn.functional as F
from pyiqa.utils.registry import ARCH_REGISTRY

#----------------------- VGGNet-----------------------------------
names = {'vgg16': ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
                   'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
                   'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2',
                   'conv3_3', 'relu3_3', 'pool3',
                   'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2',
                   'conv4_3', 'relu4_3', 'pool4',
                   'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2',
                   'conv5_3', 'relu5_3', 'pool5'],
         }

class FeaturesExtractor(nn.Module):
    def __init__(self, target_features=('relu3_3','relu4_3'),  use_input_norm=False, requires_grad=False, replace_pooling=True):
        super(FeaturesExtractor, self).__init__()
        self.use_input_norm = use_input_norm
        self.target_features = target_features

        model = torchvision.models.vgg16(pretrained=True)
        names_key = 'vgg16'

        if replace_pooling:
            self.model = self.replace_pooling(model)

        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)

        self.target_indexes = [names[names_key].index(k) for k in self.target_features]
        self.features = nn.Sequential(*list(model.features.children())[:(max(self.target_indexes) + 1)])

        if not requires_grad:
            for k, v in self.features.named_parameters():
                v.requires_grad = False
            self.features.eval()

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std

        y = OrderedDict()
        for key, layer in self.features._modules.items():
            x = layer(x)
            if int(key) in self.target_indexes:
                y.update({self.target_features[self.target_indexes.index(int(key))]: x})
        return y

    def _normalize_tensor(sefl, in_feat, eps=1e-10):
        norm_factor = torch.sqrt(torch.sum(in_feat**2, dim=1, keepdim=True))
        return in_feat / (norm_factor + eps)
    
    def replace_pooling(self, module: torch.nn.Module) -> torch.nn.Module:
        module_output = module
        if isinstance(module, torch.nn.MaxPool2d):
            module_output = L2Pool2d(kernel_size=3, stride=2, padding=1)

        for name, child in module.named_children():
            module_output.add_module(name, self.replace_pooling(child))

        return module_output


class L2Pool2d(torch.nn.Module):
    r"""Applies L2 pooling with Hann window of size 3x3
    Args:
        x: Tensor with shape (N, C, H, W)"""
    EPS = 1e-12
    def __init__(self, kernel_size: int = 3, stride: int = 2, padding=1) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.kernel = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kernel is None:
            C = x.size(1)
            self.kernel = self._hann_filter(self.kernel_size).repeat((C, 1, 1, 1)).to(x)

        out = torch.nn.functional.conv2d(
            x ** 2, self.kernel,
            stride=self.stride,
            padding=self.padding,
            groups=x.shape[1]
        )
        return (out + self.EPS).sqrt()

    def _hann_filter(self, kernel_size: int) -> torch.Tensor:
        r"""Creates  Hann kernel
        Returns:
            kernel: Tensor with shape (1, kernel_size, kernel_size)
        """
        window = torch.hann_window(kernel_size + 2, periodic=False)[1:-1]
        kernel = window[:, None] * window[None, :]
        return kernel.view(1, kernel_size, kernel_size) / kernel.sum()


#----------------------- Main Class-----------------------------------
@ARCH_REGISTRY.register()
class DMM(nn.Module):
    def __init__(self, reduce_dim=256, kernel_size=5, features_to_compute=('relu3_3','relu4_3'), criterion=torch.nn.CosineSimilarity(), use_dropout=True, **kwargs):
        super().__init__()
        self.criterion = criterion
        self.features_extractor = FeaturesExtractor(target_features=features_to_compute, replace_pooling=True)
        self.patchsize = 16
        self.stride = 4
        self.unfold = nn.Unfold(kernel_size=self.patchsize, stride=self.stride )

    def forward(self, Dist, Ref, as_loss=False):
        # preprocess image
        Ref = self.prepare_image_adt(Ref)
        Dist = self.prepare_image_adt(Dist)
        # main forward
        Ref_fea = self.features_extractor(Ref)
        with torch.no_grad():
            Dist_fea = self.features_extractor(Dist)
       
        dist = 0. 
        c1 = 1e-6
        c2 = 1e-6

        for key in Ref_fea.keys():
            
            tdistparam = Dist_fea[key]
            tprisparam = Ref_fea[key]
            a, b, c, d = tdistparam.size()
            k = self.patchsize 

            distparam = self.unfold(tdistparam).view(a, b, k, k, -1).transpose(2, 4).contiguous()
            prisparam = self.unfold(tprisparam).view(a, b, k, k, -1).transpose(2, 4).contiguous()
            pt_num = distparam.shape[2]

            distparam = distparam.view(a*b*pt_num, k, k)
            prisparam = prisparam.view(a*b*pt_num, k, k)
            
            u_r, s_r, v_r = torch.svd(prisparam)
            u_d, s_d, v_d = torch.svd(distparam)

            s_r = s_r.contiguous().view(a*b,pt_num,k)
            s_d = s_d.contiguous().view(a*b,pt_num,k)
            diff_s = ((s_r - s_d)**2)

            u_rd = (torch.abs((u_r.contiguous().view(a,b,pt_num,k,k))*(u_d.contiguous().view(a,b,pt_num,k,k)))).sum(-1)
            wt = (u_rd.std(-1))/(u_rd.mean(-1)+1e-9)

            diff_s= (((diff_s).view(a,b,pt_num,k).sum(-1))*wt).mean(-1).mean(-1)
            x_mean = tdistparam.mean([2,3], keepdim=True)
            y_mean = tprisparam.mean([2,3], keepdim=True)
            S1 = (2*x_mean*y_mean+c1)/(x_mean**2+y_mean**2+c2)
            glb = S1.squeeze(-1).squeeze(-1).mean(1)
            glb = torch.exp(-2.0*glb)
         
            dist  = dist+diff_s*glb
            
        if as_loss:
            return dist.mean()
        else:
            return dist

    def prepare_image_adt(self, tensor_image):
        b, c, h, w = tensor_image.shape
        msize = min(w, h)
        if msize > 128:
            tar_size = max(int(msize/(1.0*48))*32, 128)
            tensor_image = F.interpolate(tensor_image, size=tar_size, mode='bilinear',align_corners=False )
        return tensor_image