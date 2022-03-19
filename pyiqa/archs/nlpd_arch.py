r"""NLPD Metric

Created by: https://github.com/dingkeyan93/IQA-optimization/blob/master/IQA_pytorch/NLPD.py

Modified by: Jiadi Mo (https://github.com/JiadiMo)

Refer to:
    Matlab code from https://www.cns.nyu.edu/~lcv/NLPyr/NLP_dist.m;

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms.functional as tf

from pyiqa.archs.ssim_arch import to_y_channel
from pyiqa.utils.registry import ARCH_REGISTRY
from pyiqa.archs.arch_util import ExactPadding2d

LAPLACIAN_FILTER = np.array([[0.0025, 0.0125, 0.0200, 0.0125, 0.0025], [0.0125, 0.0625, 0.1000, 0.0625, 0.0125],
                             [0.0200, 0.1000, 0.1600, 0.1000, 0.0200], [0.0125, 0.0625, 0.1000, 0.0625, 0.0125],
                             [0.0025, 0.0125, 0.0200, 0.0125, 0.0025]],
                            dtype=np.float32)


@ARCH_REGISTRY.register()
class NLPD(nn.Module):
    r"""Normalised lapalcian pyramid distance
    Args:
        channels: Number of channel expected to calculate.
        test_y_channel: Boolean, whether to use y channel on ycbcr which mimics official matlab code.

    References:
        Laparra, Valero, Johannes Ballé, Alexander Berardino, and Eero P. Simoncelli.
        "Perceptual image quality assessment using a normalized Laplacian pyramid."
        Electronic Imaging 2016, no. 16 (2016): 1-6.

    """

    def __init__(self, channels=1, test_y_channel=True, k=6, filt=None):
        super(NLPD, self).__init__()
        if filt is None:
            filt = np.reshape(np.tile(LAPLACIAN_FILTER, (channels, 1, 1)), (channels, 1, 5, 5))
        self.k = k
        self.channels = channels
        self.test_y_channel = test_y_channel
        self.filt = nn.Parameter(torch.Tensor(filt), requires_grad=False)
        self.dn_filts, self.sigmas = self.DN_filters()
        self.pad_zero_one = nn.ZeroPad2d(1)
        self.pad_zero_two = nn.ZeroPad2d(2)
        self.pad_sym = ExactPadding2d(5, mode='symmetric')
        self.rep_one = nn.ReplicationPad2d(1)
        self.ps = nn.PixelShuffle(2)

    def DN_filters(self):
        r'''Define parameters for the divisive normalization
        '''
        sigmas = [0.0248, 0.0185, 0.0179, 0.0191, 0.0220, 0.2782]
        dn_filts = []
        dn_filts.append(
            torch.Tensor(
                np.reshape([[0, 0.1011, 0], [0.1493, 0, 0.1460], [0, 0.1015, 0.]] * self.channels,
                           (self.channels, 1, 3, 3)).astype(np.float32)))

        dn_filts.append(
            torch.Tensor(
                np.reshape([[0, 0.0757, 0], [0.1986, 0, 0.1846], [0, 0.0837, 0]] * self.channels,
                           (self.channels, 1, 3, 3)).astype(np.float32)))

        dn_filts.append(
            torch.Tensor(
                np.reshape([[0, 0.0477, 0], [0.2138, 0, 0.2243], [0, 0.0467, 0]] * self.channels,
                           (self.channels, 1, 3, 3)).astype(np.float32)))

        dn_filts.append(
            torch.Tensor(
                np.reshape([[0, 0, 0], [0.2503, 0, 0.2616], [0, 0, 0]] * self.channels,
                           (self.channels, 1, 3, 3)).astype(np.float32)))

        dn_filts.append(
            torch.Tensor(
                np.reshape([[0, 0, 0], [0.2598, 0, 0.2552], [0, 0, 0]] * self.channels,
                           (self.channels, 1, 3, 3)).astype(np.float32)))

        dn_filts.append(
            torch.Tensor(
                np.reshape([[0, 0, 0], [0.2215, 0, 0.0717], [0, 0, 0]] * self.channels,
                           (self.channels, 1, 3, 3)).astype(np.float32)))
        dn_filts = nn.ParameterList([nn.Parameter(x, requires_grad=False) for x in dn_filts])
        sigmas = nn.ParameterList([nn.Parameter(torch.Tensor(np.array(x)), requires_grad=False) for x in sigmas])
        return dn_filts, sigmas

    def pyramid(self, im):
        r'''Compute Laplacian Pyramid
        Args:
            im: An input tensor. Shape :math:`(N, C, H, W)`.
        '''
        out = []
        J = im
        pyr = []
        for i in range(0, self.k - 1):
            # Downsample. Official matlab code use 'symmetric' for padding.
            I = F.conv2d(self.pad_sym(J), self.filt, stride=2, padding=0, groups=self.channels)

            # for each dimension, check if the upsampled version has to be odd.
            odd_h, odd_w = 2 * I.size(2) - J.size(2), 2 * I.size(3) - J.size(3)

            # Upsample. Official matlab code interpolate '0' to upsample.
            I_pad = self.rep_one(I)
            I_rep1, I_rep2, I_rep3 = torch.zeros_like(I_pad), torch.zeros_like(I_pad), torch.zeros_like(I_pad)
            R = torch.cat([I_pad * 4, I_rep1, I_rep2, I_rep3], dim=1)
            I_up = self.ps(R)

            I_up_conv = F.conv2d(self.pad_zero_two(I_up), self.filt, stride=1, padding=0, groups=self.channels)
            I_up_conv = I_up_conv[:, :, 2:(I_up.shape[2] - 2 - odd_h), 2:(I_up.shape[3] - 2 - odd_w)]

            out = J - I_up_conv

            # NLP Transformation, conv2 in matlab rotate filters by 180 degrees.
            out_conv = F.conv2d(
                self.pad_zero_one(torch.abs(out)), tf.rotate(self.dn_filts[i], 180), stride=1, groups=self.channels)
            out_norm = out / (self.sigmas[i] + out_conv)
            pyr.append(out_norm)
            J = I

        # NLP Transformation for top layer, the coarest level contains the residual low pass image
        out_conv = F.conv2d(
            self.pad_zero_one(torch.abs(J)), tf.rotate(self.dn_filts[-1], 180), stride=1, groups=self.channels)
        out_norm = J / (self.sigmas[-1] + out_conv)
        pyr.append(out_norm)
        return pyr

    def nlpd(self, x1, x2):
        r"""Compute Normalised lapalcian pyramid distance for a batch of images.
        Args:
            x1: An input tensor. Shape :math:`(N, C, H, W)`.
            x2: A target tensor. Shape :math:`(N, C, H, W)`.
        Returns:
            Index of similarity betwen two images. Usually in [0, 1] interval.
        """
        assert (self.test_y_channel and self.channels == 1) or (
            not self.test_y_channel and self.channels == 3), 'Number of channel and convert to YCBCR should be match'

        if self.test_y_channel and self.channels == 1:
            x1 = to_y_channel(x1)
            x2 = to_y_channel(x2)

        y1 = self.pyramid(x1)
        y2 = self.pyramid(x2)
        total = []
        for z1, z2 in zip(y1, y2):
            diff = (z1 - z2)**2
            sqrt = torch.sqrt(torch.mean(diff, (1, 2, 3)))
            total.append(sqrt)
        score = torch.stack(total, dim=1).mean(1)
        return score

    def forward(self, X, Y):
        """Computation of NLPD metric.
        Args:
            X: An input tensor. Shape :math:`(N, C, H, W)`.
            Y: A target tensor. Shape :math:`(N, C, H, W)`.

        Returns:
            Value of NLPD metric in [0, 1] range.

        """
        assert X.shape == Y.shape, f'Input {X.shape} and reference images should have the same shape'

        score = self.nlpd(X, Y)
        return score
