r"""VIF Metric

Created by: https://github.com/dingkeyan93/IQA-optimization/blob/master/IQA_pytorch/VIF.py

Modified by: Jiadi Mo (https://github.com/JiadiMo)

Refer to:
    Matlab code from http://live.ece.utexas.edu/research/Quality/vifvec_release.zip;

"""

import torch
from torch.nn import functional as F
import numpy as np
from pyiqa.utils.color_util import to_y_channel

from pyiqa.utils.registry import ARCH_REGISTRY


def sp5_filters():
    r'''Define spatial filters.
    '''
    filters = {}
    filters['harmonics'] = np.array([1, 3, 5])
    filters['mtx'] = (
        np.array([[0.3333, 0.2887, 0.1667, 0.0000, -0.1667, -0.2887], [0.0000, 0.1667, 0.2887, 0.3333, 0.2887, 0.1667],
                  [0.3333, -0.0000, -0.3333, -0.0000, 0.3333,
                   -0.0000], [0.0000, 0.3333, 0.0000, -0.3333, 0.0000, 0.3333],
                  [0.3333, -0.2887, 0.1667, -0.0000, -0.1667, 0.2887],
                  [-0.0000, 0.1667, -0.2887, 0.3333, -0.2887, 0.1667]]))
    filters['hi0filt'] = (
        np.array([[
            -0.00033429, -0.00113093, -0.00171484, -0.00133542, -0.00080639, -0.00133542, -0.00171484, -0.00113093,
            -0.00033429
        ],
                  [
                      -0.00113093, -0.00350017, -0.00243812, 0.00631653, 0.01261227, 0.00631653, -0.00243812,
                      -0.00350017, -0.00113093
                  ],
                  [
                      -0.00171484, -0.00243812, -0.00290081, -0.00673482, -0.00981051, -0.00673482, -0.00290081,
                      -0.00243812, -0.00171484
                  ],
                  [
                      -0.00133542, 0.00631653, -0.00673482, -0.07027679, -0.11435863, -0.07027679, -0.00673482,
                      0.00631653, -0.00133542
                  ],
                  [
                      -0.00080639, 0.01261227, -0.00981051, -0.11435863, 0.81380200, -0.11435863, -0.00981051,
                      0.01261227, -0.00080639
                  ],
                  [
                      -0.00133542, 0.00631653, -0.00673482, -0.07027679, -0.11435863, -0.07027679, -0.00673482,
                      0.00631653, -0.00133542
                  ],
                  [
                      -0.00171484, -0.00243812, -0.00290081, -0.00673482, -0.00981051, -0.00673482, -0.00290081,
                      -0.00243812, -0.00171484
                  ],
                  [
                      -0.00113093, -0.00350017, -0.00243812, 0.00631653, 0.01261227, 0.00631653, -0.00243812,
                      -0.00350017, -0.00113093
                  ],
                  [
                      -0.00033429, -0.00113093, -0.00171484, -0.00133542, -0.00080639, -0.00133542, -0.00171484,
                      -0.00113093, -0.00033429
                  ]]))
    filters['lo0filt'] = (
        np.array([[0.00341614, -0.01551246, -0.03848215, -0.01551246, 0.00341614],
                  [-0.01551246, 0.05586982, 0.15925570, 0.05586982, -0.01551246],
                  [-0.03848215, 0.15925570, 0.40304148, 0.15925570, -0.03848215],
                  [-0.01551246, 0.05586982, 0.15925570, 0.05586982, -0.01551246],
                  [0.00341614, -0.01551246, -0.03848215, -0.01551246, 0.00341614]]))
    filters['lofilt'] = (2 * np.array([[
        0.00085404, -0.00244917, -0.00387812, -0.00944432, -0.00962054, -0.00944432, -0.00387812, -0.00244917,
        0.00085404
    ], [
        -0.00244917, -0.00523281, -0.00661117, 0.00410600, 0.01002988, 0.00410600, -0.00661117, -0.00523281, -0.00244917
    ], [
        -0.00387812, -0.00661117, 0.01396746, 0.03277038, 0.03981393, 0.03277038, 0.01396746, -0.00661117, -0.00387812
    ], [
        -0.00944432, 0.00410600, 0.03277038, 0.06426333, 0.08169618, 0.06426333, 0.03277038, 0.00410600, -0.00944432
    ], [
        -0.00962054, 0.01002988, 0.03981393, 0.08169618, 0.10096540, 0.08169618, 0.03981393, 0.01002988, -0.00962054
    ], [
        -0.00944432, 0.00410600, 0.03277038, 0.06426333, 0.08169618, 0.06426333, 0.03277038, 0.00410600, -0.00944432
    ], [-0.00387812, -0.00661117, 0.01396746, 0.03277038, 0.03981393, 0.03277038, 0.01396746, -0.00661117, -0.00387812],
                                       [
                                           -0.00244917, -0.00523281, -0.00661117, 0.00410600, 0.01002988, 0.00410600,
                                           -0.00661117, -0.00523281, -0.00244917
                                       ],
                                       [
                                           0.00085404, -0.00244917, -0.00387812, -0.00944432, -0.00962054, -0.00944432,
                                           -0.00387812, -0.00244917, 0.00085404
                                       ]]))
    filters['bfilts'] = (
        np.array([[
            0.00277643, 0.00496194, 0.01026699, 0.01455399, 0.01026699, 0.00496194, 0.00277643, -0.00986904,
            -0.00893064, 0.01189859, 0.02755155, 0.01189859, -0.00893064, -0.00986904, -0.01021852, -0.03075356,
            -0.08226445, -0.11732297, -0.08226445, -0.03075356, -0.01021852, 0.00000000, 0.00000000, 0.00000000,
            0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.01021852, 0.03075356, 0.08226445, 0.11732297, 0.08226445,
            0.03075356, 0.01021852, 0.00986904, 0.00893064, -0.01189859, -0.02755155, -0.01189859, 0.00893064,
            0.00986904, -0.00277643, -0.00496194, -0.01026699, -0.01455399, -0.01026699, -0.00496194, -0.00277643
        ],
                  [
                      -0.00343249, -0.00640815, -0.00073141, 0.01124321, 0.00182078, 0.00285723, 0.01166982,
                      -0.00358461, -0.01977507, -0.04084211, -0.00228219, 0.03930573, 0.01161195, 0.00128000,
                      0.01047717, 0.01486305, -0.04819057, -0.12227230, -0.05394139, 0.00853965, -0.00459034,
                      0.00790407, 0.04435647, 0.09454202, -0.00000000, -0.09454202, -0.04435647, -0.00790407,
                      0.00459034, -0.00853965, 0.05394139, 0.12227230, 0.04819057, -0.01486305, -0.01047717,
                      -0.00128000, -0.01161195, -0.03930573, 0.00228219, 0.04084211, 0.01977507, 0.00358461,
                      -0.01166982, -0.00285723, -0.00182078, -0.01124321, 0.00073141, 0.00640815, 0.00343249
                  ],
                  [
                      0.00343249, 0.00358461, -0.01047717, -0.00790407, -0.00459034, 0.00128000, 0.01166982, 0.00640815,
                      0.01977507, -0.01486305, -0.04435647, 0.00853965, 0.01161195, 0.00285723, 0.00073141, 0.04084211,
                      0.04819057, -0.09454202, -0.05394139, 0.03930573, 0.00182078, -0.01124321, 0.00228219, 0.12227230,
                      -0.00000000, -0.12227230, -0.00228219, 0.01124321, -0.00182078, -0.03930573, 0.05394139,
                      0.09454202, -0.04819057, -0.04084211, -0.00073141, -0.00285723, -0.01161195, -0.00853965,
                      0.04435647, 0.01486305, -0.01977507, -0.00640815, -0.01166982, -0.00128000, 0.00459034,
                      0.00790407, 0.01047717, -0.00358461, -0.00343249
                  ],
                  [
                      -0.00277643, 0.00986904, 0.01021852, -0.00000000, -0.01021852, -0.00986904, 0.00277643,
                      -0.00496194, 0.00893064, 0.03075356, -0.00000000, -0.03075356, -0.00893064, 0.00496194,
                      -0.01026699, -0.01189859, 0.08226445, -0.00000000, -0.08226445, 0.01189859, 0.01026699,
                      -0.01455399, -0.02755155, 0.11732297, -0.00000000, -0.11732297, 0.02755155, 0.01455399,
                      -0.01026699, -0.01189859, 0.08226445, -0.00000000, -0.08226445, 0.01189859, 0.01026699,
                      -0.00496194, 0.00893064, 0.03075356, -0.00000000, -0.03075356, -0.00893064, 0.00496194,
                      -0.00277643, 0.00986904, 0.01021852, -0.00000000, -0.01021852, -0.00986904, 0.00277643
                  ],
                  [
                      -0.01166982, -0.00128000, 0.00459034, 0.00790407, 0.01047717, -0.00358461, -0.00343249,
                      -0.00285723, -0.01161195, -0.00853965, 0.04435647, 0.01486305, -0.01977507, -0.00640815,
                      -0.00182078, -0.03930573, 0.05394139, 0.09454202, -0.04819057, -0.04084211, -0.00073141,
                      -0.01124321, 0.00228219, 0.12227230, -0.00000000, -0.12227230, -0.00228219, 0.01124321,
                      0.00073141, 0.04084211, 0.04819057, -0.09454202, -0.05394139, 0.03930573, 0.00182078, 0.00640815,
                      0.01977507, -0.01486305, -0.04435647, 0.00853965, 0.01161195, 0.00285723, 0.00343249, 0.00358461,
                      -0.01047717, -0.00790407, -0.00459034, 0.00128000, 0.01166982
                  ],
                  [
                      -0.01166982, -0.00285723, -0.00182078, -0.01124321, 0.00073141, 0.00640815, 0.00343249,
                      -0.00128000, -0.01161195, -0.03930573, 0.00228219, 0.04084211, 0.01977507, 0.00358461, 0.00459034,
                      -0.00853965, 0.05394139, 0.12227230, 0.04819057, -0.01486305, -0.01047717, 0.00790407, 0.04435647,
                      0.09454202, -0.00000000, -0.09454202, -0.04435647, -0.00790407, 0.01047717, 0.01486305,
                      -0.04819057, -0.12227230, -0.05394139, 0.00853965, -0.00459034, -0.00358461, -0.01977507,
                      -0.04084211, -0.00228219, 0.03930573, 0.01161195, 0.00128000, -0.00343249, -0.00640815,
                      -0.00073141, 0.01124321, 0.00182078, 0.00285723, 0.01166982
                  ]]).T)
    return filters


def corrDn(image, filt, step=1, channels=1):
    r'''Compute correlation of image with FILT, followed by downsampling.
    Args:
        image: A tensor. Shape :math:`(N, C, H, W)`.
        filt: A filter.
        step: Downsampling factors.
        channels: Number of channels.
    '''
    filt_ = torch.from_numpy(filt).float().unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1).to(image.device)
    p = (filt_.shape[2] - 1) // 2
    image = F.pad(image, (p, p, p, p), 'reflect')
    img = F.conv2d(image, filt_, stride=step, padding=0, groups=channels)
    return img


def SteerablePyramidSpace(image, height=4, order=5, channels=1):
    r'''Construct a steerable pyramid on image.
    Args:
        image: A tensor. Shape :math:`(N, C, H, W)`.
        height (int): Number of pyramid levels to build.
        order (int): Number of orientations.
        channels (int): Number of channels.
    '''
    num_orientations = order + 1
    filters = sp5_filters()

    hi0 = corrDn(image, filters['hi0filt'], step=1, channels=channels)
    pyr_coeffs = []
    pyr_coeffs.append(hi0)
    lo = corrDn(image, filters['lo0filt'], step=1, channels=channels)
    for _ in range(height):
        bfiltsz = int(np.floor(np.sqrt(filters['bfilts'].shape[0])))
        for b in range(num_orientations):
            filt = filters['bfilts'][:, b].reshape(bfiltsz, bfiltsz).T
            band = corrDn(lo, filt, step=1, channels=channels)
            pyr_coeffs.append(band)
        lo = corrDn(lo, filters['lofilt'], step=2, channels=channels)

    pyr_coeffs.append(lo)
    return pyr_coeffs


@ARCH_REGISTRY.register()
class VIF(torch.nn.Module):
    r'''Image Information and Visual Quality metric
    Args:
        channels (int): Number of channels.
        level (int): Number of levels to build.
        ori (int): Number of orientations.
    Reference:
        Sheikh, Hamid R., and Alan C. Bovik. "Image information and visual quality."
        IEEE Transactions on image processing 15, no. 2 (2006): 430-444.
    '''

    def __init__(self, channels=1, level=4, ori=6):

        super(VIF, self).__init__()
        self.ori = ori - 1
        self.level = level
        self.channels = channels
        self.M = 3
        self.subbands = [4, 7, 10, 13, 16, 19, 22, 25]
        self.sigma_nsq = 0.4
        self.tol = 1e-12

    def corrDn_win(self, image, filt, step=1, channels=1, start=[0, 0], end=[0, 0]):
        r'''Compute correlation of image with FILT using window, followed by downsampling.
        Args:
            image: A tensor. Shape :math:`(N, C, H, W)`.
            filt: A filter.
            step (int): Downsampling factors.
            channels (int): Number of channels.
            start (list): The window over which the convolution occurs.
            end (list): The window over which the convolution occurs.
        '''

        filt_ = torch.from_numpy(filt).float().unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1).to(image.device)
        p = (filt_.shape[2] - 1) // 2
        image = F.pad(image, (p, p, p, p), 'reflect')
        img = F.conv2d(image, filt_, stride=1, padding=0, groups=channels)
        img = img[:, :, start[0]:end[0]:step, start[1]:end[1]:step]
        return img

    def vifsub_est_M(self, org, dist):
        r'''Calculate the parameters of the distortion channel.
        Args:
            org: A reference tensor. Shape :math:`(N, C, H, W)`.
            dist: A distortion tensor. Shape :math:`(N, C, H, W)`.
        '''
        g_all = []
        vv_all = []
        for i in range(len(self.subbands)):
            sub = self.subbands[i] - 1
            y = org[sub]
            yn = dist[sub]

            lev = np.ceil((sub - 1) / 6)
            winsize = int(2**lev + 1)
            win = np.ones((winsize, winsize))

            newsizeX = int(np.floor(y.shape[2] / self.M) * self.M)
            newsizeY = int(np.floor(y.shape[3] / self.M) * self.M)
            y = y[:, :, :newsizeX, :newsizeY]
            yn = yn[:, :, :newsizeX, :newsizeY]

            winstart = [int(1 * np.floor(self.M / 2)), int(1 * np.floor(self.M / 2))]
            winend = [int(y.shape[2] - np.ceil(self.M / 2)) + 1, int(y.shape[3] - np.ceil(self.M / 2)) + 1]

            mean_x = self.corrDn_win(
                y, win / (winsize**2), step=self.M, channels=self.channels, start=winstart, end=winend)
            mean_y = self.corrDn_win(
                yn, win / (winsize**2), step=self.M, channels=self.channels, start=winstart, end=winend)
            cov_xy = self.corrDn_win(
                y * yn, win, step=self.M, channels=self.channels, start=winstart,
                end=winend) - (winsize**2) * mean_x * mean_y
            ss_x = self.corrDn_win(
                y**2, win, step=self.M, channels=self.channels, start=winstart, end=winend) - (winsize**2) * mean_x**2
            ss_y = self.corrDn_win(
                yn**2, win, step=self.M, channels=self.channels, start=winstart, end=winend) - (winsize**2) * mean_y**2

            ss_x = F.relu(ss_x)
            ss_y = F.relu(ss_y)

            g = cov_xy / (ss_x + self.tol)
            vv = (ss_y - g * cov_xy) / (winsize**2)

            g = g.masked_fill(ss_x < self.tol, 0)
            vv[ss_x < self.tol] = ss_y[ss_x < self.tol]
            ss_x = ss_x.masked_fill(ss_x < self.tol, 0)

            g = g.masked_fill(ss_y < self.tol, 0)
            vv = vv.masked_fill(ss_y < self.tol, 0)

            vv[g < 0] = ss_y[g < 0]
            g = F.relu(g)

            vv = vv.masked_fill(vv < self.tol, self.tol)

            g_all.append(g)
            vv_all.append(vv)
        return g_all, vv_all

    def refparams_vecgsm(self, org):
        r'''Calculate the parameters of the reference image.
        Args:
            org: A reference tensor. Shape :math:`(N, C, H, W)`.
        '''
        ssarr, l_arr, cu_arr = [], [], []
        for i in range(len(self.subbands)):
            sub = self.subbands[i] - 1
            y = org[sub]
            M = self.M
            newsizeX = int(np.floor(y.shape[2] / M) * M)
            newsizeY = int(np.floor(y.shape[3] / M) * M)
            y = y[:, :, :newsizeX, :newsizeY]
            B, C, H, W = y.shape

            temp = []
            for j in range(M):
                for k in range(M):
                    temp.append(y[:, :, k:H - (M - k) + 1, j:W - (M - j) + 1].reshape(B, C, -1))
            temp = torch.stack(temp, dim=3)
            mcu = torch.mean(temp, dim=2).unsqueeze(2).repeat(1, 1, temp.shape[2], 1)
            cu = torch.matmul((temp - mcu).permute(0, 1, 3, 2), temp - mcu) / temp.shape[2]

            temp = []
            for j in range(M):
                for k in range(M):
                    temp.append(y[:, :, k:H + 1:M, j:W + 1:M].reshape(B, C, -1))
            temp = torch.stack(temp, dim=2)
            ss = torch.matmul(torch.pinverse(cu), temp)
            ss = torch.sum(ss * temp, dim=2) / (M * M)
            ss = ss.reshape(B, C, H // M, W // M)
            v, _ = torch.linalg.eigh(cu, UPLO='U')
            l_arr.append(v)
            ssarr.append(ss)
            cu_arr.append(cu)

        return ssarr, l_arr, cu_arr

    def vif(self, x, y):
        r"""VIF metric. Order of input is important.
        Args:
            x: A distortion tensor. Shape :math:`(N, C, H, W)`.
            y: A reference tensor. Shape :math:`(N, C, H, W)`.
        """
        # Convert RGB image to YCBCR and use the Y-channel.
        x = to_y_channel(x, 255)
        y = to_y_channel(y, 255)

        sp_x = SteerablePyramidSpace(x, height=self.level, order=self.ori, channels=self.channels)[::-1]
        sp_y = SteerablePyramidSpace(y, height=self.level, order=self.ori, channels=self.channels)[::-1]
        g_all, vv_all = self.vifsub_est_M(sp_y, sp_x)
        ss_arr, l_arr, cu_arr = self.refparams_vecgsm(sp_y)
        num, den = [], []

        for i in range(len(self.subbands)):
            sub = self.subbands[i]
            g = g_all[i]
            vv = vv_all[i]
            ss = ss_arr[i]
            lamda = l_arr[i]
            neigvals = lamda.shape[2]
            lev = np.ceil((sub - 1) / 6)
            winsize = 2**lev + 1
            offset = (winsize - 1) / 2
            offset = int(np.ceil(offset / self.M))

            _, _, H, W = g.shape
            g = g[:, :, offset:H - offset, offset:W - offset]
            vv = vv[:, :, offset:H - offset, offset:W - offset]
            ss = ss[:, :, offset:H - offset, offset:W - offset]

            temp1 = 0
            temp2 = 0
            for j in range(neigvals):
                cc = lamda[:, :, j].unsqueeze(2).unsqueeze(3)
                temp1 = temp1 + torch.sum(torch.log2(1 + g * g * ss * cc / (vv + self.sigma_nsq)), dim=[2, 3])
                temp2 = temp2 + torch.sum(torch.log2(1 + ss * cc / (self.sigma_nsq)), dim=[2, 3])
            num.append(temp1.mean(1))
            den.append(temp2.mean(1))

        return torch.stack(num, dim=1).sum(1) / (torch.stack(den, dim=1).sum(1) + 1e-12)

    def forward(self, X, Y):
        r"""Args:
            x: A distortion tensor. Shape :math:`(N, C, H, W)`.
            y: A reference tensor. Shape :math:`(N, C, H, W)`.
            Order of input is important.
        """
        assert X.shape == Y.shape, 'Input and reference images should have the same shape, but got'
        f'{X.shape} and {Y.shape}'
        score = self.vif(X, Y)
        return score
