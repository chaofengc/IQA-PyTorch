from cv2 import reduce
import torch
import numpy as np
from torch import nn as nn
from torch.nn import functional as F

from pyiqa.utils.registry import LOSS_REGISTRY
from .loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def emd_loss(pred, target, r=2):
    """
    Args:
        pred (Tensor): of shape (N, C). Predicted tensor.
        target (Tensor): of shape (N, C). Ground truth tensor.
        r (float): norm level, default l2 norm.
    """
    loss = torch.abs(torch.cumsum(pred, dim=-1) - torch.cumsum(target, dim=-1))**r
    loss = loss.mean(dim=-1)**(1. / r)
    return loss


@LOSS_REGISTRY.register()
class EMDLoss(nn.Module):
    """EMD (earth mover distance) loss.

    """

    def __init__(self, loss_weight=1.0, r=2, reduction='mean'):
        super(EMDLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')
        self.loss_weight = loss_weight
        self.r = r
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        return self.loss_weight * emd_loss(pred, target, r=self.r, weight=weight, reduction=self.reduction)


def plcc_loss(pred, target):
    """
    Args:
        pred (Tensor): of shape (N, 1). Predicted tensor.
        target (Tensor): of shape (N, 1). Ground truth tensor.
    """
    batch_size = pred.shape[0]
    if batch_size > 1:
        vx = pred - pred.mean()
        vy = target - target.mean()
        loss = F.normalize(vx, p=2, dim=0) * F.normalize(vy, p=2, dim=0)
        loss = (1 - loss.sum()) / 2  # normalize to [0, 1]
    else:
        loss = F.l1_loss(pred, target)
    return loss.mean()


@LOSS_REGISTRY.register()
class PLCCLoss(nn.Module):
    """PLCC loss, induced from Pearson’s Linear Correlation Coefficient.

    """

    def __init__(self, loss_weight=1.0):
        super(PLCCLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        return self.loss_weight * plcc_loss(pred, target)


@LOSS_REGISTRY.register()
class RankLoss(nn.Module):
    """Monotonicity regularization loss, will be zero when rankings of pred and target are the same.

    Reference:
        - https://github.com/lidq92/LinearityIQA/blob/master/IQAloss.py

    """

    def __init__(self, detach=False, loss_weight=1.0):
        super(RankLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        if pred.size(0) > 1:  #
            ranking_loss = F.relu((pred - pred.t()) * torch.sign((target.t() - target)))
            scale = 1 + torch.max(ranking_loss.detach())
            loss = ranking_loss.mean() / scale
        else:
            loss = F.l1_loss(pred, target.detach())  # 0 for batch with single sample.
        return self.loss_weight * loss


def norm_loss_with_normalization(pred, target, p, q):
    """
    Args:
        pred (Tensor): of shape (N, 1). Predicted tensor.
        target (Tensor): of shape (N, 1). Ground truth tensor.
    """
    batch_size = pred.shape[0]
    if batch_size > 1:
        vx = pred - pred.mean()
        vy = target - target.mean()
        scale = np.power(2, p) * np.power(batch_size, max(0, 1 - p / q))  # p, q>0
        norm_pred = F.normalize(vx, p=q, dim=0)
        norm_target = F.normalize(vy, p=q, dim=0)
        loss = torch.norm(norm_pred - norm_target, p=p) / scale
    else:
        loss = F.l1_loss(pred, target)
    return loss.mean()


@LOSS_REGISTRY.register()
class NiNLoss(nn.Module):
    """NiN (Norm in Norm) loss

    Reference:

        - Dingquan Li, Tingting Jiang, and Ming Jiang. Norm-in-Norm Loss with Faster Convergence and Better
          Performance for Image Quality Assessment. ACMM2020.
        - https://arxiv.org/abs/2008.03889
        - https://github.com/lidq92/LinearityIQA

    This loss can be simply described as: l1_norm(normalize(pred - pred_mean), normalize(target - target_mean))

    """

    def __init__(self, loss_weight=1.0, p=1, q=2):
        super(NiNLoss, self).__init__()
        self.loss_weight = loss_weight
        self.p = p
        self.q = q

    def forward(self, pred, target):
        return self.loss_weight * norm_loss_with_normalization(pred, target, self.p, self.q)
