r"""NRQM Metric, proposed in

Chao Ma, Chih-Yuan Yang, Xiaokang Yang, Ming-Hsuan Yang
"Learning a No-Reference Quality Metric for Single-Image Super-Resolution"
Computer Vision and Image Understanding (CVIU), 2017

Matlab reference: https://github.com/chaoma99/sr-metric
This PyTorch implementation by: Chaofeng Chen (https://github.com/chaofengc)

"""
import math
import scipy.io
import torch
from torch import Tensor
import torch.nn.functional as F

from pyiqa.utils.registry import ARCH_REGISTRY
from pyiqa.utils.color_util import to_y_channel
from pyiqa.utils.download_util import load_file_from_url
from pyiqa.matlab_utils import imresize, fspecial, SCFpyr_PyTorch, dct2d, im2col
from pyiqa.archs.func_util import extract_2d_patches
from pyiqa.archs.ssim_arch import SSIM
from pyiqa.archs.arch_util import ExactPadding2d
from pyiqa.archs.niqe_arch import NIQE
from warnings import warn

default_model_urls = {'url': 'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/NRQM_model.mat'}


def get_guass_pyramid(x: Tensor, scale: int = 2):
    r"""Get gaussian pyramid images with gaussian kernel.
    """
    pyr = [x]
    kernel = fspecial(3, 0.5, x.shape[1]).to(x)
    pad_func = ExactPadding2d(3, stride=1, mode='same')
    for i in range(scale):
        x = F.conv2d(pad_func(x), kernel, groups=x.shape[1])
        x = x[:, :, 1::2, 1::2]
        pyr.append(x)

    return pyr


def get_var_gen_gauss(x, eps=1e-7):
    r"""Get mean and variance of input local patch.
    """
    std = x.abs().std(dim=-1, unbiased=True)
    mean = x.abs().mean(dim=-1)
    rho = std / (mean + eps)
    return rho


def gamma_gen_gauss(x: Tensor, block_seg=1e4):
    r"""General gaussian distribution estimation.

    Args:
        block_seg: maximum number of blocks in parallel to avoid OOM
    """
    pshape = x.shape[:-1]
    x = x.reshape(-1, x.shape[-1])
    eps = 1e-7
    gamma = torch.arange(0.03, 10 + 0.001, 0.001).to(x)
    r_table = (torch.lgamma(1. / gamma) + torch.lgamma(3. / gamma) - 2 * torch.lgamma(2. / gamma)).exp()
    r_table = r_table.unsqueeze(0)

    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=True)
    mean_abs = (x - mean).abs().mean(dim=-1, keepdim=True)**2

    rho = var / (mean_abs + eps)

    if rho.shape[0] > block_seg:
        rho_seg = rho.chunk(int(rho.shape[0] // block_seg))
        indexes = []
        for r in rho_seg:
            tmp_idx = (r - r_table).abs().argmin(dim=-1)
            indexes.append(tmp_idx)
        indexes = torch.cat(indexes)
    else:
        indexes = (rho - r_table).abs().argmin(dim=-1)

    solution = gamma[indexes].reshape(*pshape)
    return solution


def gamma_dct(dct_img_block: torch.Tensor):
    r"""Generalized gaussian distribution features
    """
    b, _, _, h, w = dct_img_block.shape
    dct_flatten = dct_img_block.reshape(b, -1, h * w)[:, :, 1:]
    g = gamma_gen_gauss(dct_flatten)
    g = torch.sort(g, dim=-1)[0]
    return g


def coeff_var_dct(dct_img_block: torch.Tensor):
    r"""Gaussian var, mean features
    """
    b, _, _, h, w = dct_img_block.shape
    dct_flatten = dct_img_block.reshape(b, -1, h * w)[:, :, 1:]
    rho = get_var_gen_gauss(dct_flatten)
    rho = torch.sort(rho, dim=-1)[0]
    return rho


def oriented_dct_rho(dct_img_block: torch.Tensor):
    r"""Oriented frequency features
    """
    eps = 1e-8

    # oriented 1
    feat1 = torch.cat([
        dct_img_block[..., 0, 1:],
        dct_img_block[..., 1, 2:],
        dct_img_block[..., 2, 4:],
        dct_img_block[..., 3, 5:],
    ],
        dim=-1).squeeze(-2)
    g1 = get_var_gen_gauss(feat1, eps)

    # oriented 2
    feat2 = torch.cat([
        dct_img_block[..., 1, [1]],
        dct_img_block[..., 2, 2:4],
        dct_img_block[..., 3, 2:5],
        dct_img_block[..., 4, 3:],
        dct_img_block[..., 5, 4:],
        dct_img_block[..., 6, 4:],
    ],
        dim=-1).squeeze(-2)
    g2 = get_var_gen_gauss(feat2, eps)

    # oriented 3
    feat3 = torch.cat([
        dct_img_block[..., 1:, 0],
        dct_img_block[..., 2:, 1],
        dct_img_block[..., 4:, 2],
        dct_img_block[..., 5:, 3],
    ],
        dim=-1).squeeze(-2)
    g3 = get_var_gen_gauss(feat3, eps)

    rho = torch.stack([g1, g2, g3], dim=-1).var(dim=-1)
    rho = torch.sort(rho, dim=-1)[0]
    return rho


def block_dct(img: Tensor):
    r"""Get local frequency features
    """
    img_blocks = extract_2d_patches(img, 3 + 2 * 2, 3)
    dct_img_blocks = dct2d(img_blocks)

    features = []
    # general gaussian distribution features
    gamma_L1 = gamma_dct(dct_img_blocks)
    p10_gamma_L1 = gamma_L1[:, :math.ceil(0.1 * gamma_L1.shape[-1]) + 1].mean(dim=-1)
    p100_gamma_L1 = gamma_L1.mean(dim=-1)
    features += [p10_gamma_L1, p100_gamma_L1]

    # coefficient variation estimation
    coeff_var_L1 = coeff_var_dct(dct_img_blocks)
    p10_last_cv_L1 = coeff_var_L1[:, math.floor(0.9 * coeff_var_L1.shape[-1]):].mean(dim=-1)
    p100_cv_L1 = coeff_var_L1.mean(dim=-1)
    features += [p10_last_cv_L1, p100_cv_L1]

    # oriented dct features
    ori_dct_feat = oriented_dct_rho(dct_img_blocks)
    p10_last_orientation_L1 = ori_dct_feat[:, math.floor(0.9 * ori_dct_feat.shape[-1]):].mean(dim=-1)
    p100_orientation_L1 = ori_dct_feat.mean(dim=-1)
    features += [p10_last_orientation_L1, p100_orientation_L1]

    dct_feat = torch.stack(features, dim=1)
    return dct_feat


def norm_sender_normalized(pyr, num_scale=2, num_bands=6, blksz=3, eps=1e-12):
    r"""Normalize pyramid with local spatial neighbor and band neighbor
    """
    border = blksz // 2
    guardband = 16
    subbands = []
    for si in range(num_scale):
        for bi in range(num_bands):
            idx = si * num_bands + bi
            current_band = pyr[idx]

            N = blksz**2

            # 3x3 window pixels
            tmp = F.unfold(current_band.unsqueeze(1), 3, stride=1)
            tmp = tmp.transpose(1, 2)
            b, hw = tmp.shape[:2]
            # parent pixels
            parent_idx = idx + num_bands
            if parent_idx < len(pyr):
                tmp_parent = pyr[parent_idx]
                tmp_parent = imresize(tmp_parent, sizes=current_band.shape[-2:])
                tmp_parent = tmp_parent[:, border:-border, border:-border].reshape(b, hw, 1)
                tmp = torch.cat((tmp, tmp_parent), dim=-1)
                N += 1
            # neighbor band pixels
            for ni in range(num_bands):
                if ni != bi:
                    ni_idx = si * num_bands + ni
                    tmp_nei = pyr[ni_idx]
                    tmp_nei = tmp_nei[:, border:-border, border:-border].reshape(b, hw, 1)
                    tmp = torch.cat((tmp, tmp_nei), dim=-1)
            C_x = tmp.transpose(1, 2) @ tmp / tmp.shape[1]
            # correct possible negative eigenvalue
            L, Q = torch.linalg.eigh(C_x)
            L_pos = L * (L > 0)
            L_pos_sum = L_pos.sum(dim=1, keepdim=True)
            L = L_pos * L.sum(dim=1, keepdim=True) / (L_pos_sum + (L_pos_sum == 0).float())
            C_x = Q @ torch.diag_embed(L) @ Q.transpose(1, 2)

            o_c = current_band[:, border:-border, border:-border]
            b, h, w = o_c.shape
            o_c = o_c.reshape(b, hw)
            o_c = o_c - o_c.mean(dim=1, keepdim=True)

            if hasattr(torch.linalg, 'lstsq'):
                tmp_y = torch.linalg.lstsq(C_x.transpose(1, 2), tmp.transpose(1, 2)).solution.transpose(1, 2) * tmp / N
            else:
                warn(
                    "For numerical stability, we use torch.linal.lstsq to calculate matrix inverse for PyTorch > 1.9.0. The results might be slightly different if you use older version of PyTorch.")
                tmp_y = (tmp @ torch.linalg.pinv(C_x)) * tmp / N

            z = tmp_y.sum(dim=2).sqrt()
            mask = z != 0
            g_c = o_c * mask / (z * mask + eps)
            g_c = g_c.reshape(b, h, w)

            gb = int(guardband / (2**(si)))
            g_c = g_c[:, gb:-gb, gb:-gb]
            g_c = g_c - g_c.mean(dim=(1, 2), keepdim=True)
            subbands.append(g_c)

    return subbands


def global_gsm(img: Tensor):
    """Global feature from gassian scale mixture model
    """
    batch_size = img.shape[0]
    num_bands = 6
    pyr = SCFpyr_PyTorch(height=2, nbands=num_bands, device=img.device).build(img)
    lp_bands = [x[..., 0] for x in pyr[1]] \
        + [x[..., 0] for x in pyr[2]]
    subbands = norm_sender_normalized(lp_bands)

    feat = []
    # gamma
    for sb in subbands:
        feat.append(gamma_gen_gauss(sb.reshape(batch_size, -1)))

    # gamma cross scale
    for i in range(num_bands):
        sb1 = subbands[i].reshape(batch_size, -1)
        sb2 = subbands[i + num_bands].reshape(batch_size, -1)
        gs = gamma_gen_gauss(torch.cat((sb1, sb2), dim=1))
        feat.append(gs)

    # structure correlation between scales
    hp_band = pyr[0]
    ssim_func = SSIM(channels=1, test_y_channel=False)
    for sb in subbands:
        sb_tmp = imresize(sb, sizes=hp_band.shape[1:]).unsqueeze(1)
        tmp_ssim = ssim_func(sb_tmp, hp_band.unsqueeze(1))
        feat.append(tmp_ssim)

    # structure correlation between orientations
    for i in range(num_bands):
        for j in range(i + 1, num_bands):
            feat.append(ssim_func(subbands[i].unsqueeze(1), subbands[j].unsqueeze(1)))

    feat = torch.stack(feat, dim=1)
    return feat


def tree_regression(feat, ldau, rdau, threshold_value, pred_value, best_attri):
    r"""Simple decision tree regression.
    """
    prev_k = k = 0
    for i in range(ldau.shape[0]):
        best_col = best_attri[k] - 1
        threshold = threshold_value[k]
        key_value = feat[best_col]
        prev_k = k
        k = ldau[k] - 1 if key_value <= threshold else rdau[k] - 1
        if k == -1:
            break
    y_pred = pred_value[prev_k]
    return y_pred


def random_forest_regression(feat, ldau, rdau, threshold_value, pred_value, best_attri):
    r"""Simple random forest regression.

    Note: currently, this is non-differentiable and only support CPU.
    """
    feat = feat.cpu().data.numpy()
    b, dim = feat.shape
    node_num, tree_num = ldau.shape

    pred = []
    for i in range(b):
        tmp_feat = feat[i]
        tmp_pred = []
        for i in range(tree_num):
            tmp_result = tree_regression(tmp_feat, ldau[:, i], rdau[:, i], threshold_value[:, i], pred_value[:, i],
                                         best_attri[:, i])
            tmp_pred.append(tmp_result)
        pred.append(tmp_pred)
    pred = torch.Tensor(pred)
    return pred.mean(dim=1, keepdim=True)


def nrqm(
    img: Tensor,
    linear_param,
    rf_param,
) -> Tensor:
    """Calculate NRQM
    Args:
        img (Tensor): Input image.
        linear_param (np.array): (4, 1) linear regression params
        rf_param: params of 3 random forest for 3 kinds of features
    """
    assert img.ndim == 4, ('Input image must be a gray or Y (of YCbCr) image with shape (b, c, h, w).')

    # crop image
    b, c, h, w = img.shape
    img_pyr = get_guass_pyramid(img.float() / 255.)

    # DCT features
    f1 = []
    for im in img_pyr:
        f1.append(block_dct(im))
    f1 = torch.cat(f1, dim=1)

    # gsm features
    f2 = global_gsm(img)

    # svd features
    f3 = []
    for im in img_pyr:
        col = im2col(im, 5, 'distinct')
        _, s, _ = torch.linalg.svd(col, full_matrices=False)
        f3.append(s)
    f3 = torch.cat(f3, dim=1)

    # Random forest regression. Currently not differentiable and only support CPU
    preds = torch.ones(b, 1)
    for feat, rf in zip([f1, f2, f3], rf_param):
        tmp_pred = random_forest_regression(feat, *rf)
        preds = torch.cat((preds, tmp_pred), dim=1)
    quality = preds @ torch.Tensor(linear_param)

    return quality.squeeze()


def calculate_nrqm(img: torch.Tensor,
                   crop_border: int = 0,
                   test_y_channel: bool = True,
                   pretrained_model_path: str = None,
                   color_space: str = 'yiq',
                   **kwargs) -> torch.Tensor:
    """Calculate NRQM
    Args:
        img (Tensor): Input image whose quality needs to be computed.
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the metric calculation.
        test_y_channel (Bool): Whether converted to 'y' (of MATLAB YCbCr) or 'gray'.
        pretrained_model_path (String): The pretrained model path.
    Returns:
        Tensor: NIQE result.
    """

    params = scipy.io.loadmat(pretrained_model_path)['model']
    linear_param = params['linear'][0, 0]
    rf_params_list = []
    for i in range(3):
        tmp_list = []
        tmp_param = params['rf'][0, 0][0, i][0, 0]
        tmp_list.append(tmp_param[0])  # ldau
        tmp_list.append(tmp_param[1])  # rdau
        tmp_list.append(tmp_param[4])  # threshold value
        tmp_list.append(tmp_param[5])  # pred value
        tmp_list.append(tmp_param[6])  # best attribute index
        rf_params_list.append(tmp_list)

    if test_y_channel and img.shape[1] == 3:
        img = to_y_channel(img, 255, color_space)

    if crop_border != 0:
        img = img[..., crop_border:-crop_border, crop_border:-crop_border]

    nrqm_result = nrqm(img, linear_param, rf_params_list)

    return nrqm_result.to(img)


@ARCH_REGISTRY.register()
class NRQM(torch.nn.Module):
    r""" NRQM metric

    Ma, Chao, Chih-Yuan Yang, Xiaokang Yang, and Ming-Hsuan Yang.
    "Learning a no-reference quality metric for single-image super-resolution."
    Computer Vision and Image Understanding 158 (2017): 1-16.

    Args:
        channels (int): Number of processed channel.
        test_y_channel (Boolean): whether to use y channel on ycbcr.
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the metric calculation.
        pretrained_model_path (String): The pretrained model path.
    """

    def __init__(self,
                 test_y_channel: bool = True,
                 color_space: str = 'yiq',
                 crop_border: int = 0,
                 pretrained_model_path: str = None) -> None:

        super(NRQM, self).__init__()
        self.test_y_channel = test_y_channel
        self.crop_border = crop_border
        self.color_space = color_space

        if pretrained_model_path is not None:
            self.pretrained_model_path = pretrained_model_path
        else:
            self.pretrained_model_path = load_file_from_url(default_model_urls['url'])

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        r"""Computation of NRQM metric.
        Args:
            X: An input tensor. Shape :math:`(N, C, H, W)`.
        Returns:
            Value of nrqm metric.
        """
        score = calculate_nrqm(X, self.crop_border, self.test_y_channel, self.pretrained_model_path, self.color_space)
        return score


@ARCH_REGISTRY.register()
class PI(torch.nn.Module):
    r""" Perceptual Index (PI), introduced by

    Blau, Yochai, Roey Mechrez, Radu Timofte, Tomer Michaeli, and Lihi Zelnik-Manor.
    "The 2018 pirm challenge on perceptual image super-resolution."
    In Proceedings of the European Conference on Computer Vision (ECCV) Workshops, pp. 0-0. 2018.
    Ref url: https://github.com/roimehrez/PIRM2018

    It is a combination of NIQE and NRQM: 1/2 * ((10 - NRQM) + NIQE)

    Args:
        color_space (str): color space of y channel, default ycbcr.
        crop_border (int): Cropped pixels in each edge of an image, default 4.
    """

    def __init__(self, crop_border=4, color_space='ycbcr'):
        super(PI, self).__init__()
        self.nrqm = NRQM(crop_border=crop_border, color_space=color_space)
        self.niqe = NIQE(crop_border=crop_border, color_space=color_space)

    def forward(self, X: Tensor) -> Tensor:
        r"""Computation of PI metric.
        Args:
            X: An input tensor. Shape :math:`(N, C, H, W)`.
        Returns:
            Value of PI metric.
        """
        nrqm_score = self.nrqm(X)
        niqe_score = self.niqe(X)
        score = 1 / 2 * (10 - nrqm_score + niqe_score)
        return score
