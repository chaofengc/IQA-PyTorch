import math
import torchvision as tv
from collections import OrderedDict
import fnmatch
import re

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

DEFAULT_CONFIGS = OrderedDict({
        'ckdn': {
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
        })


def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def list_models(filter='', exclude_filters=''):
    """ Return list of available model names, sorted alphabetically
    Args:
        filter (str) - Wildcard filter string that works with fnmatch
        exclude_filters (str or list[str]) - Wildcard filters to exclude models after including them with filter
    Example:
        model_list('*ssim*') -- returns all models including 'ssim'
    """
    all_models = DEFAULT_CONFIGS.keys()
    if filter:
        models = []
        include_filters = filter if isinstance(filter, (tuple, list)) else [filter]
        for f in include_filters:
            include_models = fnmatch.filter(all_models, f)  # include these models
            if len(include_models):
                models = set(models).union(include_models)
    else:
        models = all_models
    if exclude_filters:
        if not isinstance(exclude_filters, (tuple, list)):
            exclude_filters = [exclude_filters]
        for xf in exclude_filters:
            exclude_models = fnmatch.filter(models, xf)  # exclude these models
            if len(exclude_models):
                models = set(models).difference(exclude_models)
    return list(sorted(models, key=_natural_key))
