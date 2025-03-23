"""Inception score metric, proposed by

Salimans, Tim, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. "Improved techniques for training gans." Advances in neural information processing systems 29 (2016).

Reference codes:
    - https://github.com/openai/improved-gan/tree/master/inception_score
    - https://github.com/sbarratt/inception-score-pytorch

"""

import os
from scipy.stats import entropy
from tqdm import tqdm
from glob import glob
import numpy as np

import torch
from torch import nn

from .inception import InceptionV3
from pyiqa.utils.registry import ARCH_REGISTRY


from .fid_arch import get_folder_features


@ARCH_REGISTRY.register()
class InceptionScore(nn.Module):
    """Implements the Inception Score (IS) metric.

    Args:
        dims (int): The number of dimensions of the Inception-v3 feature representation to use.
            Must be one of 64, 192, 768, or 2048. Default: 2048.

    Attributes:
        model (nn.Module): The Inception-v3 network used to extract features.
    """

    def __init__(
        self,
    ) -> None:
        super().__init__()

        self.model = InceptionV3(output_blocks='logits_unbiased')
        self.model.eval()

    def forward(
        self,
        img_dir,
        mode='legacy_tensorflow',
        splits=10,
        num_workers=12,
        batch_size=32,
        device=torch.device('cuda'),
        verbose=True,
        **kwargs,
    ):
        if verbose:
            print(f'Compute inception score for {img_dir}')

        np_feats = get_folder_features(
            img_dir,
            self.model,
            num_workers=num_workers,
            batch_size=batch_size,
            device=device,
            mode=mode,
            description=f'Inception Score {img_dir}: ',
            verbose=verbose,
        )

        features = torch.from_numpy(np_feats)
        features = features[torch.randperm(features.shape[0])]

        # calculate probs and logits
        prob = features.softmax(dim=1)
        log_prob = features.log_softmax(dim=1)

        # split into groups
        prob = prob.chunk(splits, dim=0)
        log_prob = log_prob.chunk(splits, dim=0)

        # calculate score per split
        mean_prob = [p.mean(dim=0, keepdim=True) for p in prob]
        kl_ = [
            p * (log_p - m_p.log()) for p, log_p, m_p in zip(prob, log_prob, mean_prob)
        ]
        scores = [k.sum(dim=1).mean().exp().item() for k in kl_]

        # return mean and std
        return {
            'inception_score_mean': np.mean(scores),
            'inception_score_std': np.std(scores),
        }
