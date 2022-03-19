r"""Complex-valued steerable pyramid

Created by: https://github.com/tomrunia/PyTorchSteerablePyramid

Modified by: Jiadi Mo (https://github.com/JiadiMo)

Refer to:
    - Offical Matlab code from https://github.com/LabForComputationalVision/matlabPyrTools/blob/master/buildSCFpyr.m;
    - Original Python code from https://github.com/LabForComputationalVision/pyPyrTools/blob/master/pyPyrTools/SCFpyr.py;
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from scipy.special import factorial

from . import math_util

pointOp = math_util.pointOp

################################################################################
################################################################################


class SCFpyr_PyTorch(object):
    '''
    This is a modified version of buildSFpyr, that constructs a
    complex-valued steerable pyramid  using Hilbert-transform pairs
    of filters. Note that the imaginary parts will *not* be steerable.
    Pytorch version >= 1.8.0

    '''

    def __init__(self, height=5, nbands=4, scale_factor=2, device=None):
        self.height = height  # including low-pass and high-pass
        self.nbands = nbands  # number of orientation bands
        self.scale_factor = scale_factor
        self.device = torch.device('cpu') if device is None else device

        # Cache constants
        self.lutsize = 1024
        self.Xcosn = np.pi * np.array(range(-(2 * self.lutsize + 1), (self.lutsize + 2))) / self.lutsize
        self.alpha = (self.Xcosn + np.pi) % (2 * np.pi) - np.pi
        self.complex_fact_construct = np.power(np.complex(0, -1), self.nbands - 1)
        self.complex_fact_reconstruct = np.power(np.complex(0, 1), self.nbands - 1)

    ################################################################################
    # Construction of Steerable Pyramid

    def build(self, im_batch):
        ''' Decomposes a batch of images into a complex steerable pyramid.
        The pyramid typically has ~4 levels and 4-8 orientations.

        Args:
            im_batch (torch.Tensor): Batch of images of shape [N,C,H,W]

        Returns:
            pyramid: list containing torch.Tensor objects storing the pyramid
        '''

        assert im_batch.device == self.device, 'Devices invalid (pyr = {}, batch = {})'.format(
            self.device, im_batch.device)
        assert im_batch.dtype == torch.float32, 'Image batch must be torch.float32'
        assert im_batch.dim() == 4, 'Image batch must be of shape [N,C,H,W]'
        assert im_batch.shape[1] == 1, 'Second dimension must be 1 encoding grayscale image'

        im_batch = im_batch.squeeze(1)  # flatten channels dim
        height, width = im_batch.shape[1], im_batch.shape[2]

        # Check whether image size is sufficient for number of levels
        if self.height > int(np.floor(np.log2(min(width, height))) - 2):
            raise RuntimeError('Cannot build {} levels, image too small.'.format(self.height))

        # Prepare a grid
        log_rad, angle = math_util.prepare_grid(height, width)

        # Radial transition function (a raised cosine in log-frequency):
        Xrcos, Yrcos = math_util.rcosFn(1, -0.5)
        Yrcos = np.sqrt(Yrcos)

        YIrcos = np.sqrt(1 - Yrcos**2)

        lo0mask = pointOp(log_rad, YIrcos, Xrcos)
        hi0mask = pointOp(log_rad, Yrcos, Xrcos)

        # Note that we expand dims to support broadcasting later
        lo0mask = torch.from_numpy(lo0mask).float()[None, :, :, None].to(self.device)
        hi0mask = torch.from_numpy(hi0mask).float()[None, :, :, None].to(self.device)

        # Fourier transform (2D) and shifting
        batch_dft = torch.fft.fft2(im_batch)
        batch_dft = math_util.batch_fftshift2d(batch_dft)

        # Low-pass
        lo0dft = batch_dft * lo0mask

        # Start recursively building the pyramids
        coeff = self._build_levels(lo0dft, log_rad, angle, Xrcos, Yrcos, self.height)

        # High-pass
        hi0dft = batch_dft * hi0mask
        hi0 = math_util.batch_ifftshift2d(hi0dft)
        hi0 = torch.fft.ifft2(hi0)
        hi0_real = hi0.real
        coeff.insert(0, hi0_real)
        return coeff

    def _build_levels(self, lodft, log_rad, angle, Xrcos, Yrcos, height):

        if height <= 0:

            # Low-pass
            lo0 = math_util.batch_ifftshift2d(lodft)
            lo0 = torch.fft.ifft2(lo0)
            lo0_real = lo0.real
            coeff = [lo0_real]

        else:

            Xrcos = Xrcos - np.log2(self.scale_factor)

            ####################################################################
            ####################### Orientation bandpass #######################
            ####################################################################

            himask = pointOp(log_rad, Yrcos, Xrcos)
            himask = torch.from_numpy(himask[None, :, :, None]).float().to(self.device)

            order = self.nbands - 1
            const = np.power(2, 2 * order) * np.square(factorial(order)) / (self.nbands * factorial(2 * order))
            Ycosn = 2 * np.sqrt(const) * np.power(np.cos(self.Xcosn), order) * (np.abs(self.alpha) < np.pi / 2)  # [n,]

            # Loop through all orientation bands
            orientations = []
            for b in range(self.nbands):

                anglemask = pointOp(angle, Ycosn, self.Xcosn + np.pi * b / self.nbands)
                anglemask = anglemask[None, :, :, None]  # for broadcasting
                anglemask = torch.from_numpy(anglemask).float().to(self.device)

                # Bandpass filtering
                banddft = lodft * anglemask * himask

                # Now multiply with complex number
                # (x+yi)(u+vi) = (xu-yv) + (xv+yu)i
                banddft = torch.unbind(banddft, -1)
                banddft_real = self.complex_fact_construct.real * banddft[
                    0] - self.complex_fact_construct.imag * banddft[1]
                banddft_imag = self.complex_fact_construct.real * banddft[
                    1] + self.complex_fact_construct.imag * banddft[0]
                banddft = torch.stack((banddft_real, banddft_imag), -1)

                band = math_util.batch_ifftshift2d(banddft)
                band = torch.fft.ifft2(band)
                orientations.append(torch.stack((band.real, band.imag), -1))

            ####################################################################
            ######################## Subsample lowpass #########################
            ####################################################################

            # Don't consider batch_size and imag/real dim
            dims = np.array(lodft.shape[1:3])

            # Both are tuples of size 2
            low_ind_start = (np.ceil((dims + 0.5) / 2) - np.ceil((np.ceil((dims - 0.5) / 2) + 0.5) / 2)).astype(int)
            low_ind_end = (low_ind_start + np.ceil((dims - 0.5) / 2)).astype(int)

            # Subsampling indices
            log_rad = log_rad[low_ind_start[0]:low_ind_end[0], low_ind_start[1]:low_ind_end[1]]
            angle = angle[low_ind_start[0]:low_ind_end[0], low_ind_start[1]:low_ind_end[1]]

            # Actual subsampling
            lodft = lodft[:, low_ind_start[0]:low_ind_end[0], low_ind_start[1]:low_ind_end[1], :]

            # Filtering
            YIrcos = np.abs(np.sqrt(1 - Yrcos**2))
            lomask = pointOp(log_rad, YIrcos, Xrcos)
            lomask = torch.from_numpy(lomask[None, :, :, None]).float()
            lomask = lomask.to(self.device)

            # Convolution in spatial domain
            lodft = lomask * lodft

            ####################################################################
            ####################### Recursion next level #######################
            ####################################################################

            coeff = self._build_levels(lodft, log_rad, angle, Xrcos, Yrcos, height - 1)
            coeff.insert(0, orientations)

        return coeff
