from collections import OrderedDict
import fnmatch
import re

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

DEFAULT_CONFIGS = OrderedDict({
    'ahiq': {
        'metric_opts': {
            'type': 'AHIQ',
        },
        'metric_mode': 'FR',
    },
    'ckdn': {
        'metric_opts': {
            'type': 'CKDN',
        },
        'metric_mode': 'FR',
    },
    'lpips': {
        'metric_opts': {
            'type': 'LPIPS',
            'net': 'alex',
            'version': '0.1',
        },
        'metric_mode': 'FR',
        'lower_better': True,
    },
    'lpips-vgg': {
        'metric_opts': {
            'type': 'LPIPS',
            'net': 'vgg',
            'version': '0.1',
        },
        'metric_mode': 'FR',
        'lower_better': True,
    },
    'dists': {
        'metric_opts': {
            'type': 'DISTS',
        },
        'metric_mode': 'FR',
        'lower_better': True,
    },
    'ssim': {
        'metric_opts': {
            'type': 'SSIM',
            'downsample': False,
            'test_y_channel': True,
        },
        'metric_mode': 'FR',
    },
    'ssimc': {
        'metric_opts': {
            'type': 'SSIM',
            'downsample': False,
            'test_y_channel': False,
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
    'psnry': {
        'metric_opts': {
            'type': 'PSNR',
            'test_y_channel': True,
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
        'lower_better': True,
    },
    'nlpd': {
        'metric_opts': {
            'type': 'NLPD',
            'channels': 1,
            'test_y_channel': True,
        },
        'metric_mode': 'FR',
        'lower_better': True,
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
        'lower_better': True,
    },
    # =============================================================
    'niqe': {
        'metric_opts': {
            'type': 'NIQE',
            'test_y_channel': True,
        },
        'metric_mode': 'NR',
        'lower_better': True,
    },
    'ilniqe': {
        'metric_opts': {
            'type': 'ILNIQE',
        },
        'metric_mode': 'NR',
        'lower_better': True,
    },
    'brisque': {
        'metric_opts': {
            'type': 'BRISQUE',
            'test_y_channel': True,
        },
        'metric_mode': 'NR',
        'lower_better': True,
    },
    'nrqm': {
        'metric_opts': {
            'type': 'NRQM',
        },
        'metric_mode': 'NR',
    },
    'pi': {
        'metric_opts': {
            'type': 'PI',
        },
        'metric_mode': 'NR',
        'lower_better': True,
    },
    'cnniqa': {
        'metric_opts': {
            'type': 'CNNIQA',
            'pretrained': 'koniq10k'
        },
        'metric_mode': 'NR',
    },
    'musiq': {
        'metric_opts': {
            'type': 'MUSIQ',
            'pretrained': 'koniq10k'
        },
        'metric_mode': 'NR',
    },
    'musiq-ava': {
        'metric_opts': {
            'type': 'MUSIQ',
            'pretrained': 'ava'
        },
        'metric_mode': 'NR',
    },
    'musiq-koniq': {
        'metric_opts': {
            'type': 'MUSIQ',
            'pretrained': 'koniq10k'
        },
        'metric_mode': 'NR',
    },
    'musiq-paq2piq': {
        'metric_opts': {
            'type': 'MUSIQ',
            'pretrained': 'paq2piq'
        },
        'metric_mode': 'NR',
    },
    'musiq-spaq': {
        'metric_opts': {
            'type': 'MUSIQ',
            'pretrained': 'spaq'
        },
        'metric_mode': 'NR',
    },
    'nima': {
        'metric_opts': {
            'type': 'NIMA',
            'pretrained': 'ava',
            'base_model_name': 'inception_resnet_v2',
        },
        'metric_mode': 'NR',
    },
    'nima-vgg16-ava': {
        'metric_opts': {
            'type': 'NIMA',
            'pretrained': 'ava',
            'base_model_name': 'vgg16',
        },
        'metric_mode': 'NR',
    },
    'pieapp': {
        'metric_opts': {
            'type': 'PieAPP',
        },
        'metric_mode': 'FR',
        'lower_better': True,
    },
    'paq2piq': {
        'metric_opts': {
            'type': 'PAQ2PIQ',
        },
        'metric_mode': 'NR',
    },
    'dbcnn': {
        'metric_opts': {
            'type': 'DBCNN',
            'pretrained': 'koniq'
        },
        'metric_mode': 'NR',
    },
    'fid': {
        'metric_opts': {
            'type': 'FID',
        },
        'metric_mode': 'NR'
    },
    'maniqa': {
        'metric_opts': {
            'type': 'MANIQA',
        },
        'metric_mode': 'NR',
    },

})
