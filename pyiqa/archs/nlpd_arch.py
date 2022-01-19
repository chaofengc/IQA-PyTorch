import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms.functional as tf

from pyiqa.archs.ssim_arch import to_y_channel
from pyiqa.utils.registry import ARCH_REGISTRY
from pyiqa.archs.arch_util import SymmetricPad2d


def padding_matlab(x):
    return torch.from_numpy(
        np.lib.pad(x.numpy(), ((0, 0), (0, 0), (2, 2), (2, 2)), 'symmetric'))


def upsample_matlab(x):
    pad_rep_1 = nn.ReplicationPad2d(1)
    padded = pad_rep_1(x)
    upsample_x = torch.zeros(padded.shape[0], padded.shape[1],
                             padded.shape[2] * 2, padded.shape[3] * 2)
    upsample_x[:, :, 0:2:padded.shape[2] * 2 - 1,
               1:2:padded.shape[3] * 2 - 1] = 4 * padded


LAPLACIAN_FILTER = np.array([[0.0025, 0.0125, 0.0200, 0.0125, 0.0025],
                             [0.0125, 0.0625, 0.1000, 0.0625, 0.0125],
                             [0.0200, 0.1000, 0.1600, 0.1000, 0.0200],
                             [0.0125, 0.0625, 0.1000, 0.0625, 0.0125],
                             [0.0025, 0.0125, 0.0200, 0.0125, 0.0025]],
                            dtype=np.float32)


@ARCH_REGISTRY.register()
class NLPD(nn.Module):
    """
    Normalised lapalcian pyramid distance.
    Refer to https://www.cns.nyu.edu/pub/eero/laparra16a-preprint.pdf
    https://github.com/alexhepburn/nlpd-tensorflow
    """

    def __init__(self, channels=1, k=6, filt=None):
        super(NLPD, self).__init__()
        if filt is None:
            filt = np.reshape(np.tile(LAPLACIAN_FILTER, (channels, 1, 1)),
                              (channels, 1, 5, 5))
        self.k = k
        self.channels = channels
        self.filt = nn.Parameter(torch.Tensor(filt), requires_grad=False)
        self.dn_filts, self.sigmas = self.DN_filters()
        self.pad_zero_one = nn.ZeroPad2d(1)
        self.pad_zero_two = nn.ZeroPad2d(2)
        self.pad_sym = SymmetricPad2d(2)
        self.rep_one = nn.ReplicationPad2d(1)
        self.ps = nn.PixelShuffle(2)
        self.upsample = nn.Upsample(scale_factor=2,
                                    mode='bilinear',
                                    align_corners=True)

    def DN_filters(self):
        sigmas = [0.0248, 0.0185, 0.0179, 0.0191, 0.0220, 0.2782]
        dn_filts = []
        dn_filts.append(
            torch.Tensor(
                np.reshape(
                    [[0, 0.1011, 0], [0.1493, 0, 0.1460], [0, 0.1015, 0.]] *
                    self.channels,
                    (self.channels, 1, 3, 3)).astype(np.float32)))

        dn_filts.append(
            torch.Tensor(
                np.reshape(
                    [[0, 0.0757, 0], [0.1986, 0, 0.1846], [0, 0.0837, 0]] *
                    self.channels,
                    (self.channels, 1, 3, 3)).astype(np.float32)))

        dn_filts.append(
            torch.Tensor(
                np.reshape(
                    [[0, 0.0477, 0], [0.2138, 0, 0.2243], [0, 0.0467, 0]] *
                    self.channels,
                    (self.channels, 1, 3, 3)).astype(np.float32)))

        dn_filts.append(
            torch.Tensor(
                np.reshape([[0, 0, 0], [0.2503, 0, 0.2616], [0, 0, 0]] *
                           self.channels,
                           (self.channels, 1, 3, 3)).astype(np.float32)))

        dn_filts.append(
            torch.Tensor(
                np.reshape([[0, 0, 0], [0.2598, 0, 0.2552], [0, 0, 0]] *
                           self.channels,
                           (self.channels, 1, 3, 3)).astype(np.float32)))

        dn_filts.append(
            torch.Tensor(
                np.reshape([[0, 0, 0], [0.2215, 0, 0.0717], [0, 0, 0]] *
                           self.channels,
                           (self.channels, 1, 3, 3)).astype(np.float32)))
        dn_filts = nn.ParameterList(
            [nn.Parameter(x, requires_grad=False) for x in dn_filts])
        sigmas = nn.ParameterList([
            nn.Parameter(torch.Tensor(np.array(x)), requires_grad=False)
            for x in sigmas
        ])
        return dn_filts, sigmas

    def pyramid(self, im):
        out = []
        J = im
        pyr = []
        for i in range(0, self.k - 1):
            # Downsample. Official matlab code use 'symmetric' for padding but it is not
            # impleted the same way in pytorch
            I = F.conv2d(self.pad_sym(J),
                         self.filt,
                         stride=2,
                         padding=0,
                         groups=self.channels)

            # for each dimension, check if the upsampled version has to be odd
            odd_h, odd_w = 2 * I.size(2) - J.size(2), 2 * I.size(3) - J.size(3)

            # Upsample. Official matlab code interpolate '0' to upsample
            I_pad = self.rep_one(I)
            I_rep1, I_rep2, I_rep3 = torch.zeros(I_pad.size()), torch.zeros(
                I_pad.size()), torch.zeros(I_pad.size())
            R = torch.cat([I_pad * 4, I_rep1, I_rep2, I_rep3], dim=1)
            I_up = self.ps(R)

            I_up_conv = F.conv2d(self.pad_zero_two(I_up),
                                 self.filt,
                                 stride=1,
                                 padding=0,
                                 groups=self.channels)
            I_up_conv = I_up_conv[:, :, 2:(I_up.shape[2] - 2 - odd_h),
                                  2:(I_up.shape[3] - 2 - odd_w)]

            out = J - I_up_conv

            # NLP Transformation 
            out_conv = F.conv2d(self.pad_zero_one(torch.abs(out)),
                                tf.rotate(self.dn_filts[i],180),
                                stride=1,
                                groups=self.channels)
            out_norm = out / (self.sigmas[i] + out_conv)
            pyr.append(out_norm)
            J = I

        # NLP Transformation for top layer, the coarest level contains the residual low pass image
        out_conv = F.conv2d(self.pad_zero_one(torch.abs(J)),
                            tf.rotate(self.dn_filts[-1],180),
                            stride=1,
                            groups=self.channels)
        out_norm = J / (self.sigmas[-1] + out_conv)
        pyr.append(out_norm)
        return pyr

    def nlpd(self, x1, x2):
        x1 = to_y_channel(x1) / 255
        x2 = to_y_channel(x2) / 255

        y1 = self.pyramid(x1)
        y2 = self.pyramid(x2)
        total = []
        for z1, z2 in zip(y1, y2):
            diff = (z1 - z2)**2
            sqrt = torch.sqrt(torch.mean(diff, (1, 2, 3)))
            total.append(sqrt)
        score = torch.stack(total, dim=1).mean(1)
        return score

    def forward(
        self,
        y,
        x,
    ):
        assert x.shape == y.shape

        score = self.nlpd(x, y)
        return score
