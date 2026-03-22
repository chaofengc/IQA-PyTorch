r"""Entropy Metric for grayscale images.
Entropy is a statistical measure of randomness that can be used to characterize the texture of the input image.

Created by: Chaofeng Chen (https://github.com/chaofengc)

Refer to:
    Matlab: https://www.mathworks.com/help/images/ref/entropy.html

"""

import torch
import torch.nn as nn

from pyiqa.utils.registry import ARCH_REGISTRY
from pyiqa.utils.color_util import to_y_channel


def entropy(x, data_range=255.0, eps=1e-8, color_space='yiq'):
    r"""Compute grayscale entropy from an image tensor.

    Args:
        x (torch.Tensor): Input tensor with shape ``(N, C, H, W)``.
        data_range (float): Maximum intensity value used for histogram bins.
        eps (float): Numerical stability constant in ``log2``.
        color_space (str): Color space used when converting RGB to luminance.

    Returns:
        torch.Tensor: Entropy values with shape ``(N,)``.
    """

    if x.shape[1] == 3:
        # Convert RGB image to gray scale and use Y-channel
        x = to_y_channel(x, data_range, color_space)

    # Compute histogram
    hist = nn.functional.one_hot(x.long(), num_classes=int(data_range + 1)).sum(
        dim=[1, 2, 3]
    )
    hist = hist / hist.sum(dim=1, keepdim=True)

    # Compute entropy
    score = -torch.sum(hist * torch.log2(hist + eps), dim=1)

    return score


@ARCH_REGISTRY.register()
class Entropy(nn.Module):
    r"""Entropy-based no-reference image quality metric wrapper.

    Args:
        **kwargs: Keyword arguments forwarded to :func:`entropy`.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def forward(self, x):
        """Compute entropy score for input batch.

        Args:
            x (torch.Tensor): Image tensor with shape ``(N, C, H, W)``.

        Returns:
            torch.Tensor: Entropy score tensor with shape ``(N,)``.
        """
        score = entropy(x, **self.kwargs)
        return score
