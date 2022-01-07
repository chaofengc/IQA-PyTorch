import numpy as np
import os
import sys
import torch
from torchvision import models, transforms
import torch.nn as nn
import torch.nn.functional as F
import math

from pyiqa.utils.registry import ARCH_REGISTRY


class FSIM_base(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)
        self.cuda_computation = False
        self.nscale = 4  # Number of wavelet scales
        self.norient = 4  # Number of filter orientations
        self.k = 2.0  # No of standard deviations of the noise
        # energy beyond the mean at which we set the
        # noise threshold point.
        # below which phase congruency values get
        # penalized.

        self.epsilon = .0001  # Used to prevent division by zero
        self.pi = math.pi

        minWaveLength = 6  # Wavelength of smallest scale filter
        mult = 2  # Scaling factor between successive filters
        sigmaOnf = 0.55  # Ratio of the standard deviation of the
        # Gaussian describing the log Gabor filter's
        # transfer function in the frequency domain
        # to the filter center frequency.
        dThetaOnSigma = 1.2  # Ratio of angular interval between filter orientations
        # and the standard deviation of the angular Gaussian
        # function used to construct filters in the
        # freq. plane.

        self.thetaSigma = self.pi / self.norient / dThetaOnSigma  # Calculate the standard deviation of the
        # angular Gaussian function used to
        # construct filters in the freq. plane.

        self.fo = (1.0 / (minWaveLength * torch.pow(
            mult, (torch.arange(0, self.nscale, dtype=torch.float64))))).unsqueeze(
                0)  # Centre frequency of filter
        self.den = 2 * (math.log(sigmaOnf))**2
        self.dx = -torch.tensor([[[[3, 0, -3], [10, 0, -10], [3, 0, -3]]]]) / 16.0
        self.dy = -torch.tensor([[[[3, 10, 3], [0, 0, 0], [-3, -10, -3]]]]) / 16.0
        self.T1 = 0.85
        self.T2 = 160
        self.T3 = 200
        self.T4 = 200
        self.lambdac = 0.03

    def set_arrays_to_cuda(self):
        self.cuda_computation = True
        self.fo = self.fo.cuda()
        self.dx = self.dx.cuda()
        self.dy = self.dy.cuda()

    def forward_gradloss(self, imgr, imgd):
        I1, Q1, Y1 = self.process_image_channels(imgr)
        I2, Q2, Y2 = self.process_image_channels(imgd)

        #PCSimMatrix,PCm = self.calculate_phase_score(PC1,PC2)
        gradientMap1 = self.calculate_gradient_map(Y1)
        gradientMap2 = self.calculate_gradient_map(Y2)

        gradientSimMatrix = self.calculate_gradient_sim(
            gradientMap1, gradientMap2)
        #gradientSimMatrix= gradientSimMatrix.view(PCSimMatrix.size())
        gradloss = torch.sum(torch.sum(torch.sum(gradientSimMatrix, 1), 1))
        return gradloss

    def calculate_fsim(self, gradientSimMatrix, PCSimMatrix, PCm):
        SimMatrix = gradientSimMatrix * PCSimMatrix * PCm
        FSIM = torch.sum(torch.sum(SimMatrix, 1), 1) / torch.sum(torch.sum(PCm, 1), 1)
        return FSIM

    def calculate_fsimc(self, I1, Q1, I2, Q2, gradientSimMatrix, PCSimMatrix,
                        PCm):

        ISimMatrix = (2 * I1 * I2 + self.T3) / (torch.pow(I1, 2) + torch.pow(I2, 2) +
                                                self.T3)
        QSimMatrix = (2 * Q1 * Q2 + self.T4) / (torch.pow(Q1, 2) + torch.pow(Q2, 2) +
                                                self.T4)
        SimMatrixC = gradientSimMatrix * PCSimMatrix * (torch.pow(
            torch.abs(ISimMatrix * QSimMatrix), self.lambdac)) * PCm
        FSIMc = torch.sum(torch.sum(SimMatrixC, 1), 1) / torch.sum(torch.sum(PCm, 1), 1)

        return FSIMc

    def lowpassfilter(self, rows, cols):
        cutoff = .45
        n = 15
        x, y = self.create_meshgrid(cols, rows)
        radius = torch.sqrt(torch.pow(x, 2) + torch.pow(y, 2)).unsqueeze(0)
        f = self.ifftshift2d(1 / (1.0 + torch.pow(torch.div(radius, cutoff), 2 * n)))
        return f

    def calculate_gradient_sim(self, gradientMap1, gradientMap2):

        gradientSimMatrix = (2 * gradientMap1 * gradientMap2 + self.T2) / (
            torch.pow(gradientMap1, 2) + torch.pow(gradientMap2, 2) + self.T2)
        return gradientSimMatrix

    def calculate_gradient_map(self, Y):
        IxY = F.conv2d(Y, self.dx, padding=1)
        IyY = F.conv2d(Y, self.dy, padding=1)
        gradientMap1 = torch.sqrt(torch.pow(IxY, 2) + torch.pow(IyY, 2))
        return gradientMap1

    def calculate_phase_score(self, PC1, PC2):
        PCSimMatrix = (2 * PC1 * PC2 + self.T1) / (torch.pow(PC1, 2) +
                                                   torch.pow(PC2, 2) + self.T1)
        PCm = torch.where(PC1 > PC2, PC1, PC2)
        return PCSimMatrix, PCm

    def roll_1(self, x, n):
        return torch.cat((x[:, -n:, :, :, :], x[:, :-n, :, :, :]), dim=1)

    def ifftshift(self, tens, var_axis):
        len11 = int(tens.size()[var_axis] / 2)
        len12 = tens.size()[var_axis] - len11
        return torch.cat((tens.narrow(var_axis, len11,
                                   len12), tens.narrow(var_axis, 0, len11)),
                      axis=var_axis)

    def ifftshift2d(self, tens):
        return self.ifftshift(self.ifftshift(tens, 1), 2)

    def create_meshgrid(self, cols, rows):
        '''
        Set up X and Y matrices with ranges normalised to +/- 0.5
        The following code adjusts things appropriately for odd and even values
        of rows and columns.
        '''

        if cols % 2:
            xrange = torch.arange(start=-(cols - 1) / 2,
                               end=(cols - 1) / 2 + 1,
                               step=1,
                               requires_grad=False) / (cols - 1)
        else:
            xrange = torch.arange(
                -(cols) / 2, (cols) / 2, step=1, requires_grad=False) / (cols)

        if rows % 2:
            yrange = torch.arange(-(rows - 1) / 2, (rows - 1) / 2 + 1,
                               step=1,
                               requires_grad=False) / (rows - 1)
        else:
            yrange = torch.arange(
                -(rows) / 2, (rows) / 2, step=1, requires_grad=False) / (rows)

        x, y = torch.meshgrid([xrange, yrange])

        if self.cuda_computation:
            x, y = x.cuda(), y.cuda()

        return x.T, y.T

    def process_image_channels(self, img):

        batch, rows, cols = img.shape[0], img.shape[2], img.shape[3]

        minDimension = min(rows, cols)

        Ycoef = torch.tensor([[0.299, 0.587, 0.114]])
        Icoef = torch.tensor([[0.596, -0.274, -0.322]])
        Qcoef = torch.tensor([[0.211, -0.523, 0.312]])

        if self.cuda_computation:
            Ycoef, Icoef, Qcoef = Ycoef.cuda(), Icoef.cuda(), Qcoef.cuda()

        Yfilt = torch.cat(
            batch * [
                torch.cat(rows * cols * [Ycoef.unsqueeze(2)], dim=2).view(
                    1, 3, rows, cols)
            ], 0)
        Ifilt = torch.cat(
            batch * [
                torch.cat(rows * cols * [Icoef.unsqueeze(2)], dim=2).view(
                    1, 3, rows, cols)
            ], 0)
        Qfilt = torch.cat(
            batch * [
                torch.cat(rows * cols * [Qcoef.unsqueeze(2)], dim=2).view(
                    1, 3, rows, cols)
            ], 0)

        # If images have three chanels
        if img.size()[1] == 3:
            Y = torch.sum(Yfilt * img, 1).unsqueeze(1)
            I = torch.sum(Ifilt * img, 1).unsqueeze(1)
            Q = torch.sum(Qfilt * img, 1).unsqueeze(1)
        else:
            Y = torch.mean(img, 1).unsqueeze(1)
            I = torch.ones(Y.size(), dtype=torch.float64)
            Q = torch.ones(Y.size(), dtype=torch.float64)

        F = max(1, round(minDimension / 256))

        aveKernel = nn.AvgPool2d(kernel_size=F, stride=F,
                                 padding=0)  # max(0, math.floor(F/2)))
        if self.cuda_computation:
            aveKernel = aveKernel.cuda()

        # Make sure that the dimension of the returned image is the same as the input
        I = aveKernel(I)
        Q = aveKernel(Q)
        Y = aveKernel(Y)
        return I, Q, Y

    def phasecong2(self, img):
        '''
        % Filters are constructed in terms of two components.
        % 1) The radial component, which controls the frequency band that the filter
        %    responds to
        % 2) The angular component, which controls the orientation that the filter
        %    responds to.
        % The two components are multiplied together to construct the overall filter.
        % Construct the radial filter components...
        % First construct a low-pass filter that is as large as possible, yet falls
        % away to zero at the boundaries.  All log Gabor filters are multiplied by
        % this to ensure no extra frequencies at the 'corners' of the FFT are
        % incorporated as this seems to upset the normalisation process when
        % calculating phase congrunecy.
        '''

        batch, rows, cols = img.shape[0], img.shape[2], img.shape[3]

        imagefft = torch.fft.rfft2(img, dim=(-2, -1))
        imagefft = torch.stack((imagefft.real, imagefft.imag), -1)

        x, y = self.create_meshgrid(cols, rows)

        radius = torch.cat(
            batch * [torch.sqrt(torch.pow(x, 2) + torch.pow(y, 2)).unsqueeze(0)], 0)
        theta = torch.cat(batch * [torch.atan2(-y, x).unsqueeze(0)], 0)

        radius = self.ifftshift2d(
            radius)  # Matrix values contain *normalised* radius from centre
        theta = self.ifftshift2d(theta)  # Matrix values contain polar angle.
        # (note -ve y is used to give +ve
        # anti-clockwise angles)

        radius[:, 0, 0] = 1

        sintheta = torch.sin(theta)
        costheta = torch.cos(theta)

        lp = self.lowpassfilter(rows, cols)  # Radius .45, 'sharpness' 15
        lp = torch.cat(batch * [lp.unsqueeze(0)], 0)

        term1 = torch.cat(rows * cols * [self.fo.unsqueeze(2)],
                       dim=2).view(-1, self.nscale, rows, cols)
        term1 = torch.cat(batch * [term1.unsqueeze(0)],
                       0).view(-1, self.nscale, rows, cols)

        term2 = torch.log(torch.cat(self.nscale * [radius.unsqueeze(1)], 1) / term1)
        #  Apply low-pass filter
        logGabor = torch.exp(-torch.pow(term2, 2) / self.den)
        logGabor = logGabor * lp
        logGabor[:, :, 0,
                 0] = 0  # Set the value at the 0 frequency point of the filter
        # back to zero (undo the radius fudge).

        # Then construct the angular filter components...

        # For each point in the filter matrix calculate the angular distance from
        # the specified filter orientation.  To overcome the angular wrap-around
        # problem sine difference and cosine difference values are first computed
        # and then the atan2 function is used to determine angular distance.
        angl = torch.arange(0, self.norient,
                         dtype=torch.float64) / self.norient * self.pi

        if self.cuda_computation:
            angl = angl.cuda()
        ds_t1 = torch.cat(self.norient * [sintheta.unsqueeze(1)],
                       1) * torch.cos(angl).view(-1, self.norient, 1, 1)
        ds_t2 = torch.cat(self.norient * [costheta.unsqueeze(1)],
                       1) * torch.sin(angl).view(-1, self.norient, 1, 1)
        dc_t1 = torch.cat(self.norient * [costheta.unsqueeze(1)],
                       1) * torch.cos(angl).view(-1, self.norient, 1, 1)
        dc_t2 = torch.cat(self.norient * [sintheta.unsqueeze(1)],
                       1) * torch.sin(angl).view(-1, self.norient, 1, 1)
        ds = ds_t1 - ds_t2  # Difference in sine.
        dc = dc_t1 + dc_t2  # Difference in cosine.
        dtheta = torch.abs(torch.atan2(ds, dc))  # Absolute angular distance.
        spread = torch.exp(-torch.pow(dtheta, 2) /
                        (2 * self.thetaSigma**2))  # Calculate the
        # angular filter component.

        logGabor_rep = torch.repeat_interleave(logGabor, self.norient,
                                            1).view(-1, self.nscale,
                                                    self.norient, rows, cols)

        # Batch size, scale, orientation, pixels, pixels
        spread_rep = torch.cat(self.nscale * [spread]).view(
            -1, self.nscale, self.norient, rows, cols)
        filter_log_spread = logGabor_rep * spread_rep
        array_of_zeros = torch.zeros(filter_log_spread.unsqueeze(5).size(),
                                  dtype=torch.float64)
        if self.cuda_computation:
            array_of_zeros = array_of_zeros.cuda()
        filter_log_spread_zero = torch.cat(
            (filter_log_spread.unsqueeze(5), array_of_zeros), dim=5)
        ifftFilterArray = torch.fft.ifft2(filter_log_spread_zero, dim=(-2, -1)
                                  )
        ifftFilterArray = torch.stack((ifftFilterArray.real, ifftFilterArray.imag), -1).select(5, 0) * math.sqrt(
                                      rows * cols)

        imagefft_repeat = torch.cat(self.nscale * self.norient * [imagefft],
                                 dim=1).view(-1, self.nscale, self.norient,
                                             rows, cols, 2)
        filter_log_spread_repeat = torch.cat(2 * [filter_log_spread.unsqueeze(5)],
                                          dim=5)
        # Convolve image with even and odd filters returning the result in EO
        EO = torch.fft.ifft2(filter_log_spread_repeat * imagefft_repeat, dim=(-2, -1))

        E = EO.select(5, 0)
        O = EO.select(5, 1)
        An = torch.sqrt(torch.pow(E, 2) + torch.pow(O, 2))
        sumAn_ThisOrient = torch.sum(An, 1)
        sumE_ThisOrient = torch.sum(E,
                                 1)  # Sum of even filter convolution results
        sumO_ThisOrient = torch.sum(O,
                                 1)  # Sum of odd filter convolution results.

        # Get weighted mean filter response vector, this gives the weighted mean
        # phase angle.
        XEnergy = torch.sqrt(
            torch.pow(sumE_ThisOrient, 2) +
            torch.pow(sumO_ThisOrient, 2)) + self.epsilon
        MeanE = sumE_ThisOrient / XEnergy
        MeanO = sumO_ThisOrient / XEnergy

        MeanO = torch.cat(self.nscale * [MeanO.unsqueeze(1)], 1)
        MeanE = torch.cat(self.nscale * [MeanE.unsqueeze(1)], 1)

        # Now calculate An(cos(phase_deviation) - | sin(phase_deviation)) | by
        # using dot and cross products between the weighted mean filter response
        # vector and the individual filter response vectors at each scale.  This
        # quantity is phase congruency multiplied by An, which we call energy.
        Energy = torch.sum(E * MeanE + O * MeanO - torch.abs(E * MeanO - O * MeanE),
                        1)
        abs_EO = torch.sqrt(torch.pow(E, 2) + torch.pow(O, 2))

        #   % Compensate for noise
        # We estimate the noise power from the energy squared response at the
        # smallest scale.  If the noise is Gaussian the energy squared will have a
        # Chi-squared 2DOF pdf.  We calculate the median energy squared response
        # as this is a robust statistic.  From this we estimate the mean.
        # The estimate of noise power is obtained by dividing the mean squared
        # energy value by the mean squared filter value
        medianE2n = torch.pow(abs_EO.select(1, 0),
                           2).view(-1, self.norient,
                                   rows * cols).median(2).values

        EM_n = torch.sum(torch.sum(torch.pow(filter_log_spread.select(1, 0), 2), 3), 2)
        noisePower = -(medianE2n / math.log(0.5)) / EM_n

        # Now estimate the total energy^2 due to noise
        # Estimate for sum(An^2) + sum(Ai.*Aj.*(cphi.*cphj + sphi.*sphj))
        EstSumAn2 = torch.sum(torch.pow(ifftFilterArray, 2), 1)

        sumEstSumAn2 = torch.sum(torch.sum(EstSumAn2, 2), 2)
        roll_t1 = ifftFilterArray * self.roll_1(ifftFilterArray, 1)
        roll_t2 = ifftFilterArray * self.roll_1(ifftFilterArray, 2)
        roll_t3 = ifftFilterArray * self.roll_1(ifftFilterArray, 3)
        rolling_mult = roll_t1 + roll_t2 + roll_t3
        EstSumAiAj = torch.sum(rolling_mult, 1) / 2
        sumEstSumAiAj = torch.sum(torch.sum(EstSumAiAj, 2), 2)

        EstNoiseEnergy2 = 2 * noisePower * sumEstSumAn2 + 4 * noisePower * sumEstSumAiAj
        tau = torch.sqrt(EstNoiseEnergy2 / 2)
        EstNoiseEnergy = tau * math.sqrt(self.pi / 2)
        EstNoiseEnergySigma = torch.sqrt((2 - self.pi / 2) * torch.pow(tau, 2))

        # The estimated noise effect calculated above is only valid for the PC_1 measure.
        # The PC_2 measure does not lend itself readily to the same analysis.  However
        # empirically it seems that the noise effect is overestimated roughly by a factor
        # of 1.7 for the filter parameters used here.
        T = (EstNoiseEnergy +
             self.k * EstNoiseEnergySigma) / 1.7  # Noise threshold

        T_exp = torch.cat(rows * cols * [T.unsqueeze(2)],
                       dim=2).view(-1, self.norient, rows, cols)
        AnAll = torch.sum(sumAn_ThisOrient, 1)
        array_of_zeros_energy = torch.zeros(Energy.size(), dtype=torch.float64)
        if self.cuda_computation:
            array_of_zeros_energy = array_of_zeros_energy.cuda()

        EnergyAll = torch.sum(
            torch.where((Energy - T_exp) < 0.0, array_of_zeros_energy,
                     Energy - T_exp), 1)
        ResultPC = EnergyAll / AnAll

        return ResultPC


class FSIM(FSIM_base):
    '''
    Note, the input is expected to be from 0 to 255
    '''

    def __init__(self):
        super().__init__()

    def forward(self, imgr, imgd):
        if imgr.is_cuda:
            self.set_arrays_to_cuda()

        I1, Q1, Y1 = self.process_image_channels(imgr)
        I2, Q2, Y2 = self.process_image_channels(imgd)
        PC1 = self.phasecong2(Y1)
        PC2 = self.phasecong2(Y2)

        PCSimMatrix, PCm = self.calculate_phase_score(PC1, PC2)
        gradientMap1 = self.calculate_gradient_map(Y1)
        gradientMap2 = self.calculate_gradient_map(Y2)

        gradientSimMatrix = self.calculate_gradient_sim(
            gradientMap1, gradientMap2)
        gradientSimMatrix = gradientSimMatrix.view(PCSimMatrix.size())
        FSIM = self.calculate_fsim(gradientSimMatrix, PCSimMatrix, PCm)

        return FSIM.mean()


@ARCH_REGISTRY.register()
class FSIMc(FSIM_base, nn.Module):
    '''
    Note, the input is expected to be from 0 to 255
    '''

    def __init__(self):
        super().__init__()

    def forward(self, imgr, imgd):
        if imgr.is_cuda:
            self.set_arrays_to_cuda()

        I1, Q1, Y1 = self.process_image_channels(imgr)
        I2, Q2, Y2 = self.process_image_channels(imgd)
        PC1 = self.phasecong2(Y1)
        PC2 = self.phasecong2(Y2)

        PCSimMatrix, PCm = self.calculate_phase_score(PC1, PC2)
        gradientMap1 = self.calculate_gradient_map(Y1)
        gradientMap2 = self.calculate_gradient_map(Y2)

        gradientSimMatrix = self.calculate_gradient_sim(
            gradientMap1, gradientMap2)
        gradientSimMatrix = gradientSimMatrix.view(PCSimMatrix.size())
        FSIMc = self.calculate_fsimc(I1.squeeze(), Q1.squeeze(), I2.squeeze(),
                                     Q2.squeeze(), gradientSimMatrix,
                                     PCSimMatrix, PCm)

        return FSIMc.mean()