r"""DeepDC: Deep Distance Correlation as a Perceptual Image Quality Evaluator

Reference:
@article{zhu2024adaptive,
  title={DeepDC: Deep Distance Correlation as a Perceptual Image Quality Evaluator},
  author={Zhu, Hanwei and Chen, Baoliang and Zhu, Lingyu and Wang, Shiqi and Weisi, Lin},
  journal={arXiv preprint arXiv:2211.04927},
  year={2024},
}

Reference url: https://github.com/h4nwei/DeepDC
"""

import torch
import torch.nn as nn
from collections import OrderedDict
import torchvision
from torchvision import models, transforms
from pyiqa.utils.registry import ARCH_REGISTRY

names = {
    'vgg19': [
        'image',
        'conv1_1',
        'relu1_1',
        'conv1_2',
        'relu1_2',
        'pool1',
        'conv2_1',
        'relu2_1',
        'conv2_2',
        'relu2_2',
        'pool2',
        'conv3_1',
        'relu3_1',
        'conv3_2',
        'relu3_2',
        'conv3_3',
        'relu3_3',
        'conv3_4',
        'relu3_4',
        'pool3',
        'conv4_1',
        'relu4_1',
        'conv4_2',
        'relu4_2',
        'conv4_3',
        'relu4_3',
        'conv4_4',
        'relu4_4',
        'pool4',
        'conv5_1',
        'relu5_1',
        'conv5_2',
        'relu5_2',
        'conv5_3',
        'relu5_3',
        'conv5_4',
        'relu5_4',
        'pool5',
    ],
}


class MultiVGGFeaturesExtractor(nn.Module):
    def __init__(
        self,
        target_features=('conv1_2', 'conv2_2', 'conv3_4', 'conv4_4', 'conv5_4'),
        use_input_norm=True,
        requires_grad=False,
    ):  # ALL FALSE is the best for COS_Similarity; Correlation: use_norm = True
        super(MultiVGGFeaturesExtractor, self).__init__()
        self.use_input_norm = use_input_norm
        self.target_features = target_features

        model = torchvision.models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        names_key = 'vgg19'

        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)

        self.target_indexes = [
            names[names_key].index(k) - 1 for k in self.target_features
        ]
        self.features = nn.Sequential(
            *list(model.features.children())[: (max(self.target_indexes) + 1)]
        )

        if not requires_grad:
            for k, v in self.features.named_parameters():
                v.requires_grad = False
            self.features.eval()

    def forward(self, x):
        # assume input range is [0, 1]
        if self.use_input_norm:
            x = (x - self.mean) / self.std

        y = OrderedDict()
        if 'image' in self.target_features:
            y.update({'image': x})
        for key, layer in self.features._modules.items():
            x = layer(x)
            # x = self._normalize_tensor(x)
            if int(key) in self.target_indexes:
                y.update({self.target_features[self.target_indexes.index(int(key))]: x})
        return y

    def _normalize_tensor(sefl, in_feat, eps=1e-10):
        norm_factor = torch.sqrt(torch.sum(in_feat**2, dim=1, keepdim=True))
        return in_feat / (norm_factor + eps)


@ARCH_REGISTRY.register()
class DeepDC(nn.Module):
    def __init__(
        self,
        features_to_compute=('conv1_2', 'conv2_2', 'conv3_4', 'conv4_4', 'conv5_4'),
    ):
        super(DeepDC, self).__init__()
        self.MSE = torch.nn.MSELoss()
        self.features_extractor = MultiVGGFeaturesExtractor(
            target_features=features_to_compute
        ).eval()

    def forward(self, x, y):
        r"""Compute IQA using DeepDC model.

        Args:
            - x: An input tensor with (N, C, H, W) shape. RGB channel order for colour images.
            - y: An reference tensor with (N, C, H, W) shape. RGB channel order for colour images.

        Returns:
            Value of DeepDC model.

        """
        targets, inputs = x, y
        inputs_fea = self.features_extractor(inputs)

        with torch.no_grad():
            targets_fea = self.features_extractor(targets)

        dc_scores = []

        for _, key in enumerate(inputs_fea.keys()):
            inputs_dcdm = self._DCDM(inputs_fea[key])
            targets_dcdm = self._DCDM(targets_fea[key])
            dc_scores.append(self.Distance_Correlation(inputs_dcdm, targets_dcdm))

        dc_scores = torch.stack(dc_scores, dim=1)

        score = 1 - dc_scores.mean(dim=1, keepdim=True)

        return score

    # double-centered distance matrix (dcdm)
    def _DCDM(self, x):
        if len(x.shape) == 4:
            batchSize, dim, h, w = x.data.shape
            M = h * w
        elif len(x.shape) == 3:
            batchSize, M, dim = x.data.shape
        x = x.reshape(batchSize, dim, M)
        t = torch.log((1.0 / (torch.tensor(dim) * torch.tensor(dim))))

        I = (
            torch.eye(dim, dim, device=x.device)
            .view(1, dim, dim)
            .repeat(batchSize, 1, 1)
            .type(x.dtype)
        )
        I_M = torch.ones(batchSize, dim, dim, device=x.device).type(x.dtype)
        x_pow2 = x.bmm(x.transpose(1, 2))
        dcov = I_M.bmm(x_pow2 * I) + (x_pow2 * I).bmm(I_M) - 2 * x_pow2

        dcov = torch.clamp(dcov, min=0.0)
        dcov = torch.exp(t) * dcov
        dcov = torch.sqrt(dcov + 1e-5)
        dcdm = (
            dcov
            - 1.0 / dim * dcov.bmm(I_M)
            - 1.0 / dim * I_M.bmm(dcov)
            + 1.0 / (dim * dim) * I_M.bmm(dcov).bmm(I_M)
        )

        return dcdm

    def Distance_Correlation(self, matrix_A, matrix_B):
        Gamma_XY = torch.sum(matrix_A * matrix_B, dim=[1, 2])
        Gamma_XX = torch.sum(matrix_A * matrix_A, dim=[1, 2])
        Gamma_YY = torch.sum(matrix_B * matrix_B, dim=[1, 2])
        c = 1e-6
        correlation_r = (Gamma_XY + c) / (torch.sqrt(Gamma_XX * Gamma_YY) + c)
        return correlation_r


def prepare_image(image, resize=True):
    if resize and min(image.size) > 256:
        image = transforms.functional.resize(image, 256)
    image = transforms.ToTensor()(image)
    return image.unsqueeze(0)


if __name__ == '__main__':
    from PIL import Image
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', type=str, default='../images/r0.png')
    parser.add_argument('--dist', type=str, default='../images/r1.png')
    args = parser.parse_args()

    ref = prepare_image(Image.open(args.ref).convert('RGB'))
    dist = prepare_image(Image.open(args.dist).convert('RGB'))
    assert ref.shape == dist.shape

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepDC().to(device)
    ref = ref.to(device)
    dist = dist.to(device)
    score = model(ref, dist)
    print(score.item())
    # score: 0.3347
