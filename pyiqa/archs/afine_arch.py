"""A-FINE Model

github repo link: https://github.com/ChrisDud0257/AFINE

Cite as:
@inproceedings{chen2025toward,
  title={Toward Generalized Image Quality Assessment: Relaxing the Perfect Reference Quality Assumption},
  author={Chen, Du and Wu, Tianhe and Ma, Kede and Zhang, Lei},
  booktitle={Proceedings of the 2025 IEEE/CVF Computer Vision and Pattern Recognition Conference},
  pages={12742--12752},
  year={2025}
}

This file only support inferring A-FINE value. If you want to further train A-FINE, plase refer to https://github.com/ChrisDud0257/AFINE

"""

import torch
import torch.nn as nn
import os
import math
from torchvision.transforms.functional import normalize
import numpy as np

from .constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

from pyiqa.utils.registry import ARCH_REGISTRY
from pyiqa.utils.download_util import DEFAULT_CACHE_DIR
from pyiqa.archs.arch_util import load_pretrained_network
from pyiqa.utils.download_util import load_file_from_url

from copy import deepcopy


from .afineclip_model import load
import torch.nn.functional as F
from itertools import product
from pyiqa.archs.arch_util import get_url_from_name

default_model_urls = {
    'afine': get_url_from_name('afine.pth'),
}


def scale_finalscore(score, yita1 = 100, yita2 = 0, yita3 = -1.9710, yita4 = -2.3734):

    exp_pow = -1 * (score - yita3) / (math.fabs(yita4) + 1e-10)
    if exp_pow >=10:
        scale_score = (yita1 - yita2) * torch.exp(-1 * exp_pow) / (1 + torch.exp(-1 * exp_pow)) + yita2
    else:
        scale_score = (yita1 - yita2) / (1 + torch.exp(exp_pow)) + yita2

    # scale_score = (yita1 - yita2) / (1 + math.exp(-1 * (score - yita3) / (np.abs(yita4)))) + yita2
    return scale_score


class AFINEQhead(nn.Module):
    def __init__(self, chns = (3, 768, 768, 768, 768, 768, 768, 768, 768, 768, 768, 768, 768), feature_out_channel = 1,
                       input_dim = 768, hidden_dim = 128,
                       mean = (0.48145466, 0.4578275, 0.40821073), std = (0.26862954, 0.26130258, 0.27577711)):
        super(AFINEQhead, self).__init__()

        self.chns = chns
        self.feature_out_channel = feature_out_channel
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.register_buffer("mean", torch.tensor(mean).view(1,-1,1,1))
        self.register_buffer("std", torch.tensor(std).view(1,-1,1,1))

        self.proj_feat = nn.Linear(input_dim * 2, hidden_dim)
        self.proj_head = nn.Sequential(
            nn.Linear(self.chns[0] * 2 + hidden_dim * (len(self.chns) - 1), hidden_dim * 6),
            nn.GELU(),
            nn.Linear(hidden_dim * 6, self.feature_out_channel)
        )


    def forward(self, x, h_list_x):
        x = x * self.std + self.mean

        img_feature_x = x.flatten(2).permute(0, 2, 1)

        feature_list_x = []

        feature_list_x.append(img_feature_x)
        for h_x in h_list_x:
            feature_list_x.append(F.relu(h_x))

        final_feature_list_x = []

        for k in range(len(self.chns)):
            x_mean = feature_list_x[k].mean(1, keepdim=True)

            x_var = ((feature_list_x[k]-x_mean)**2).mean(1, keepdim=True)

            concat_x_feature = torch.cat((x_mean.flatten(1), x_var.flatten(1)), dim=1)

            if k != 0:
                concat_x_feature = self.proj_feat(concat_x_feature)

            final_feature_list_x.append(concat_x_feature)

        concat_final_feature_lixt_x = torch.cat(final_feature_list_x, dim = 1)

        n_x = self.proj_head(concat_final_feature_lixt_x)

        return n_x



class AFINEDhead(nn.Module):
    def __init__(self, chns = (3, 768, 768, 768, 768, 768, 768, 768, 768, 768, 768, 768, 768),
                 mean = (0.48145466, 0.4578275, 0.40821073), std = (0.26862954, 0.26130258, 0.27577711)):
        super(AFINEDhead, self).__init__()

        self.chns = chns

        self.register_parameter("alpha", nn.Parameter(torch.randn(1, 1, sum(self.chns)), requires_grad=True))
        self.register_parameter("beta", nn.Parameter(torch.randn(1, 1, sum(self.chns)), requires_grad=True))
        self.alpha.data.normal_(0.1,0.01)
        self.beta.data.normal_(0.1,0.01)

        self.softplus = nn.Softplus()

        self.register_buffer("mean", torch.tensor(mean).view(1,-1,1,1))
        self.register_buffer("std", torch.tensor(std).view(1,-1,1,1))

    def forward(self, x, y, h_list_x, h_list_y):
        ### the input image should be generalized back to its original values
        x = x * self.std + self.mean
        y = y * self.std + self.mean

        # print(f"mean is {self.mean}, std is {self.std}")

        img_feature_x = x.flatten(2).permute(0, 2, 1)
        img_feature_y = y.flatten(2).permute(0, 2, 1)

        feature_list_x = []
        feature_list_y = []

        feature_list_x.append(img_feature_x)
        for h_x in h_list_x:
            feature_list_x.append(F.relu(h_x))

        feature_list_y.append(img_feature_y)
        for h_y in h_list_y:
            feature_list_y.append(F.relu(h_y))

        dist1 = 0
        dist2 = 0
        c1 = 1e-10
        c2 = 1e-10

        alpha_ = self.softplus(self.alpha)
        beta_ = self.softplus(self.beta)

        w_sum = alpha_.sum() + beta_.sum() + 1e-10
        alpha = torch.split(alpha_/w_sum, self.chns, dim=2)
        beta = torch.split(beta_/w_sum, self.chns, dim=2)

        for k in range(len(self.chns)):
            x_mean = feature_list_x[k].mean(1, keepdim=True)
            y_mean = feature_list_y[k].mean(1, keepdim=True)

            S1 = (2*x_mean*y_mean+c1)/(x_mean**2+y_mean**2+c1)
            # print(f"feature_list_x{[k]} shape is {feature_list_x[k].shape}, feature_list_y{[k]} shape is {feature_list_y[k].shape}, alpha[{k}] shape is {alpha[k].shape}, S1 shape is {S1.shape}")
            dist1 = dist1+(alpha[k]*S1).sum(2,keepdim=True)

            x_var = ((feature_list_x[k]-x_mean)**2).mean(1, keepdim=True)
            y_var = ((feature_list_y[k]-y_mean)**2).mean(1, keepdim=True)
            xy_cov = (feature_list_x[k]*feature_list_y[k]).mean(1,keepdim=True) - x_mean*y_mean
            S2 = (2*xy_cov+c2)/(x_var+y_var+c2)
            dist2 = dist2+(beta[k]*S2).sum(2,keepdim=True)

        score = 1 - (dist1+dist2).squeeze(2)
        # print(f"score shape is {score.shape}")

        return score


### Non-linear mapping to generalize NR and FR scores to a fixed limitation
class AFINENLM_NR_Fit(nn.Module):
    def __init__(self, yita1 = 2, yita2 = -2, yita3 = 3.7833, yita4 = 7.5676):
        super(AFINENLM_NR_Fit, self).__init__()
        self.yita3 = nn.Parameter(torch.tensor(yita3, dtype=torch.float32), requires_grad = True)
        self.yita4 = nn.Parameter(torch.tensor(yita4, dtype=torch.float32), requires_grad = True)
        self.yita1 = yita1
        self.yita2 = yita2

    def forward(self, x):
        # print(f"For NR, self.yita3 is {self.yita3}, self.yita4 is {self.yita4}")
        # d_hat = (self.yita1 - self.yita2) / (1 + torch.exp(-1 * (x - self.yita3) / (torch.abs(self.yita4) + 1e-10))) + self.yita2

        exp_pow = -1 * (x - self.yita3) / (torch.abs(self.yita4) + 1e-10)

        if exp_pow >=10:
            d_hat = (self.yita1 - self.yita2) * torch.exp(-1 * exp_pow) / (1 + torch.exp(-1 * exp_pow)) + self.yita2
        else:
            d_hat = (self.yita1 - self.yita2) / (1 + torch.exp(exp_pow)) + self.yita2

        return d_hat


### Non-linear mapping to generalize NR and FR scores to a fixed limitation
class AFINENLM_FR_Fit_with_limit(nn.Module):
    def __init__(self, yita1 = 2, yita2 = -2, yita3 = -24.1335, yita4 = 8.1093, yita3_upper = -21, yita3_lower = -27, yita4_upper = 9, yita4_lower = 7):
        super(AFINENLM_FR_Fit_with_limit, self).__init__()
        self.yita3 = nn.Parameter(torch.tensor(yita3, dtype=torch.float32), requires_grad = True)
        self.yita4 = nn.Parameter(torch.tensor(yita4, dtype=torch.float32), requires_grad = True)
        self.yita1 = yita1
        self.yita2 = yita2
        self.yita3_upper = yita3_upper
        self.yita3_lower = yita3_lower
        self.yita4_upper = yita4_upper
        self.yita4_lower = yita4_lower

    def forward(self, x):
        yita3_ = torch.clamp(self.yita3, self.yita3_lower, self.yita3_upper)
        yita4_ = torch.clamp(self.yita4, self.yita4_lower, self.yita4_upper)
        # print(f"For FR, self.yita3 is {self.yita3}, yita3 is {yita3_}, self.yita4 is {self.yita4}, yita4 is {yita4_}")
        # d_hat = (self.yita1 - self.yita2) / (1 + torch.exp(-1 * (x - yita3_) / (torch.abs(yita4_) + 1e-10))) + self.yita2

        exp_pow = -1 * (x - yita3_) / (torch.abs(yita4_) + 1e-10)

        if exp_pow >=10:
            d_hat = (self.yita1 - self.yita2) * torch.exp(-1 * exp_pow) / (1 + torch.exp(-1 * exp_pow)) + self.yita2
        else:
            d_hat = (self.yita1 - self.yita2) / (1 + torch.exp(exp_pow)) + self.yita2

        return d_hat


### adapter
class AFINELearnLambda(nn.Module):
    def __init__(self, k = 5):
        super(AFINELearnLambda, self).__init__()

        self.k = nn.Parameter(torch.tensor(k, dtype=torch.float32), requires_grad = True)


    def forward(self, x_nr, ref_nr, xref_fr):
        k_ = F.softplus(self.k)
        # print(f"self.k is {self.k}, k_ is {k_}")
        u = torch.exp(k_*(ref_nr - x_nr)) * x_nr + xref_fr

        return u

@ARCH_REGISTRY.register()
class AFINE(nn.Module):
    def __init__(
        self,
        model_type='afine_all_scale',
        clip_backbone='ViT-B/32',
        # clip_backbone='/mnt/bn/chenduchris/pretrained_models/CLIP/ViT-B-32.pt',
        step=32,
        num_patch=15,
        pretrained=True,
        # pretrained_model_path='/mnt/bn/chenduchris/pretrained_models/AFINE/afine.pth',
        pretrained_model_path=None,
        url_key = 'afine'
) -> None:
        super().__init__()
        self.clip_backbone = clip_backbone
        ### If you cannot download the pretrained CLIP model in on-line manner when you infer A-FINE, then please manually download it from "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt"
        ### After you download the CLIP model, please substitute the following code as self.clip_model = load("/your save path/ViT-B-32.pt", device="cpu", jit = False)
        clip_model, _ = load(self.clip_backbone, device="cpu", jit = False)

        ###afine_all_scale means the final scaled A-FINE score which adaptively combines fidelity term and naturalness term, afine_fr means the fidelity term, afine_nr means the naturalness term
        self.model_type = model_type

        self.mean = (0.48145466, 0.4578275, 0.40821073)
        self.std = (0.26862954, 0.26130258, 0.27577711)

        # load our finetuned CLIP
        if pretrained_model_path is None:
            model_path = default_model_urls[url_key]
            if model_path.startswith('https://') or model_path.startswith('http://'):
                pretrained_model_path = load_file_from_url(model_path)
        
        finetuned_clip_checkpoint = torch.load(pretrained_model_path, map_location = 'cpu')['finetuned_clip']
        clip_model.load_state_dict(finetuned_clip_checkpoint)
        self.clip_model = clip_model
        # load naturalness term
        net_qhead = AFINEQhead()
        net_qhead.load_state_dict(torch.load(pretrained_model_path, map_location = 'cpu')['natural'], strict=True)
        self.net_qhead = net_qhead
        # load fidelity term
        net_dhead = AFINEDhead()
        net_dhead.load_state_dict(torch.load(pretrained_model_path, map_location = 'cpu')['fidelity'], strict=True)
        self.net_dhead = net_dhead
        # load non-linear mapping for fidelity term
        net_scale_fr = AFINENLM_FR_Fit_with_limit(yita1=2,yita2=-2,yita3=0.5,yita4=0.15,
                                                yita3_upper=0.95,yita3_lower=0.05,yita4_upper=0.70,yita4_lower=0.01)
        net_scale_fr.load_state_dict(torch.load(pretrained_model_path, map_location = 'cpu')['fidelity_scale'], strict=True)
        self.net_scale_fr = net_scale_fr
        # load non-linear mapping for naturalness term
        net_scale_nr = AFINENLM_NR_Fit(yita1 = 2, yita2 = -2, yita3 = 4.9592, yita4 = 21.5968)
        net_scale_nr.load_state_dict(torch.load(pretrained_model_path, map_location = 'cpu')['natural_scale'], strict=True)
        self.net_scale_nr = net_scale_nr
        # load adptive term
        adapter = AFINELearnLambda(k=5)
        adapter.load_state_dict(torch.load(pretrained_model_path, map_location = 'cpu')['adapter'], strict=True)
        self.adapter = adapter

        self.clip_model.eval()
        self.net_qhead.eval()
        self.net_dhead.eval()
        self.net_scale_fr.eval()
        self.net_scale_nr.eval()
        self.adapter.eval()


    def forward(self, dis, ref=None):
        ### note that, dis must path to the distortion image path, while ref must path to the reference image path, you cannot switch them.
        
        # preprocess for distortion image and reference image
        dis = dis.squeeze(0)
        normalize(dis, self.mean, self.std, inplace=True)
        dis = dis.unsqueeze(0)

        if ref is None:
            ref = deepcopy(dis)
        ref = ref.squeeze(0)
        normalize(ref, self.mean, self.std, inplace=True)
        ref = ref.unsqueeze(0)


        # The height and width of all the images must be divisible by 32, since we utilize the pretrained CLIP ViT-B-32 model
        _,c,h,w = dis.shape
        if h % 32 != 0:
            pad_h = 32 - h % 32
        else:
            pad_h = 0

        if w % 32 != 0:
            pad_w = 32 - w % 32
        else:
            pad_w = 0

        if pad_h > 0 or pad_w > 0:
            dis = F.interpolate(dis, size = (h + pad_h, w + pad_w), mode = 'bicubic', align_corners = False)
            ref = F.interpolate(ref, size = (h + pad_h, w + pad_w), mode = 'bicubic', align_corners = False)

        # Compute A-FINE scores
        # Please note that, for all terms, including the final A-FINE score, the A-FINE fidelity/naturalness term, lower values indicate better quality
        # To prevent from numerical overflow, we use 'afine_all_scale' value to indicate the final scaled Full-reference score for (dis, ref)
        with torch.no_grad():
            cls_dis, feat_dis = self.clip_model.encode_image(dis)
            cls_ref, feat_ref = self.clip_model.encode_image(ref)
            natural_dis = self.net_qhead(dis, feat_dis)
            natural_ref = self.net_qhead(ref, feat_ref)
            natural_dis_scale = self.net_scale_nr(natural_dis)
            natural_ref_scale = self.net_scale_nr(natural_ref)

            fidelity_disref = self.net_dhead(dis, ref, feat_dis, feat_ref)
            fidelity_disref_scale = self.net_scale_fr(fidelity_disref)

            afine_all = self.adapter(natural_dis_scale, natural_ref_scale, fidelity_disref_scale)

            afine_all_scale = scale_finalscore(score = afine_all)

            if self.model_type == 'afine_nr':
                return natural_dis_scale.squeeze()
            elif self.model_type == 'afine_fr':
                return fidelity_disref_scale.squeeze()
            elif self.model_type == 'afine_all_scale':
                return afine_all_scale.squeeze()
            elif self.model_type == 'afine_all':
                return afine_all.squeeze()
            else:
                raise ValueError(f"self.model_type must be afine_nr, afine_fr, afine_all or afine_all_scale.")

