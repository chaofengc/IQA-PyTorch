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
        'ckdn': {
            'metric_opts': {
                'type': 'CKDN',
                'pretrained_model_path': './experiments/pretrained_models/CKDN/model_best.pth',
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
        'lpips': {
            'metric_opts': {
                'type': 'LPIPS',
                'net': 'alex',
                'version': '0.1',
                'pretrained_model_path': './experiments/pretrained_models/LPIPS/v0.1/alex.pth',
                },
            'metric_mode': 'FR',
            },
        'dists': {
            'metric_opts': {
                'type': 'DISTS',
                'pretrained_model_path': './experiments/pretrained_models/DISTS/weights.pt',
                },
            'metric_mode': 'FR',
            },
        'ssim': {
            'metric_opts': {
                'type': 'SSIM',
                'downsample': False,
                'test_y_channel': True,
                },
            'metric_mode': 'FR',
            },
        'psnr': {
            'metric_opts': {
                'type': 'PSNR',
                'test_y_channel': False,
                },
            'metric_mode': 'FR',
            },
        'fsim': {
            'metric_opts': {
                'type': 'FSIM',
                'chromatic': True,
                },
            'metric_mode': 'FR',
            },
        'ms_ssim': {
            'metric_opts': {
                'type': 'MS_SSIM',
                'downsample': False,
                'test_y_channel': True,
                'is_prod': True,
                },
            'metric_mode': 'FR',
            },
        'vif': {
            'metric_opts': {
                'type': 'VIF',
                },
            'metric_mode': 'FR',
            },
        'gmsd': {
            'metric_opts': {
                'type': 'GMSD',
                'test_y_channel': True,
                },
            'metric_mode': 'FR',
            },
        'nlpd': {
            'metric_opts': {
                'type': 'NLPD',
                'channels': 1,
                'test_y_channel': True,
                },
            'metric_mode': 'FR',
            },
        'vsi': {
            'metric_opts': {
                'type': 'VSI',
                },
            'metric_mode': 'FR',
            },
        'cw_ssim': {
            'metric_opts': {
                'type': 'CW_SSIM',
                'channels': 1,
                'level': 4,
                'ori': 8,
                'test_y_channel': True,
                },
            'metric_mode': 'FR',
            },
        'mad': {
            'metric_opts': {
                'type': 'MAD',
                'test_y_channel': True,
                },
            'metric_mode': 'FR',
            },
        'niqe': {
            'metric_opts': {
                'type': 'NIQE',
                'test_y_channel': True,
                'pretrained_model_path': './experiments/pretrained_models/NIQE/modelparameters.mat',
                },
            'metric_mode': 'NR',
            },
        'brisque': {
            'metric_opts': {
                'type': 'BRISQUE',
                'test_y_channel': True,
                'pretrained_model_path': './experiments/pretrained_models/BRISQUE/brisque_svm_weights.pt',
                },
            'metric_mode': 'NR',
            },
        'musiq': {
            'metric_opts': {
                'type': 'MUSIQ',
                'num_class': 10,
                'pretrained_model_path': './experiments/pretrained_models/MUSIQ/musiq_ava_ckpt.pth',
            },
            'metric_mode': 'NR',
            },
        'dbcnn': {
            'metric_opts': {
                'type': 'DBCNN',
                'pretrained_scnn_path': './experiments/pretrained_models/DBCNN/DBCNN_scnn.pth',
            },
            'metric_mode': 'NR',
            },
        })

