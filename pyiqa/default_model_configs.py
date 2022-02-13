import cv2
import numpy as np
import os
import math
import torch
import torchvision as tv
from PIL import Image
from collections import OrderedDict

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

DEFAULT_CONFIGS = OrderedDict({
        'CKDN': {
            'metric_opts': {
                'type': 'CKDN',
                'pretrained_model_path': './experiments/pretrained_models/CKDN/model_best.pth.tar',
                'use_diff_preprocess': False,
                },
            'metric_mode': 'FR',
            'preprocess_x': tv.transforms.Compose([
                tv.transforms.Resize(int(math.floor(288/0.875)), tv.transforms.InterpolationMode.BICUBIC),
                tv.transforms.CenterCrop(288),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
                ]),
            'preprocess_y': tv.transforms.Compose([
                tv.transforms.Resize(int(math.floor(288/0.875)), tv.transforms.InterpolationMode.NEAREST),
                tv.transforms.CenterCrop(288),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
                ]),
            },
        'LPIPS': {
            'metric_opts': {
                'type': 'LPIPS',
                'net': 'alex',
                'version': '0.1',
                'pretrained_model_path': './experiments/pretrained_models/LPIPS/v0.1/alex.pth',
                },
            'metric_mode': 'FR',
            },
        'DISTS': {
            'metric_opts': {
                'type': 'DISTS',
                'pretrained_model_path': './experiments/pretrained_models/DISTS/weights.pt',
                },
            'metric_mode': 'FR',
            },
        'SSIM': {
            'metric_opts': {
                'type': 'SSIM',
                'downsample': False,
                'test_y_channel': True,
                },
            'metric_mode': 'FR',
            },
        'PSNR': {
            'metric_opts': {
                'type': 'PSNR',
                'test_y_channel': False,
                },
            'metric_mode': 'FR',
            },
        'FSIM': {
            'metric_opts': {
                'type': 'FSIM',
                'chromatic': True,
                },
            'metric_mode': 'FR',
            },
        'MS_SSIM': {
            'metric_opts': {
                'type': 'MS_SSIM',
                'downsample': False,
                'test_y_channel': True,
                'is_prod': True,
                },
            'metric_mode': 'FR',
            },
        'VIF': {
            'metric_opts': {
                'type': 'VIF',
                },
            'metric_mode': 'FR',
            },
        'GMSD': {
            'metric_opts': {
                'type': 'GMSD',
                'test_y_channel': True,
                },
            'metric_mode': 'FR',
            },
        'NLPD': {
            'metric_opts': {
                'type': 'NLPD',
                'channels': 1,
                'test_y_channel': True,
                },
            'metric_mode': 'FR',
            },
        'VSI': {
            'metric_opts': {
                'type': 'VSI',
                },
            'metric_mode': 'FR',
            },
        'CW_SSIM': {
            'metric_opts': {
                'type': 'CW_SSIM',
                'channels': 1,
                'level': 4,
                'ori': 8,
                'test_y_channel': True,
                },
            'metric_mode': 'FR',
            },
        'MAD': {
            'metric_opts': {
                'type': 'MAD',
                'test_y_channel': True,
                },
            'metric_mode': 'FR',
            },
        'NIQE': {
            'metric_opts': {
                'type': 'NIQE',
                'test_y_channel': True,
                'pretrained_model_path': './experiments/pretrained_models/NIQE/modelparameters.mat',
                },
            'metric_mode': 'NR',
            },
        'MUSIQ': {
            'metric_opts': {
                'type': 'MUSIQ',
                'num_class': 10,
                'pretrained_model_path': './experiments/pretrained_models/MUSIQ/musiq_ava_ckpt.pth',
                },
            'metric_mode': 'NR',
            },
        })

