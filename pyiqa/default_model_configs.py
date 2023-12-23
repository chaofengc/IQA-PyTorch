from collections import OrderedDict

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
    'stlpips': {
        'metric_opts': {
            'type': 'STLPIPS',
            'net': 'alex',
            'variant': 'shift_tolerant',
        },
        'metric_mode': 'FR',
        'lower_better': True,
    },
    'stlpips-vgg': {
        'metric_opts': {
            'type': 'STLPIPS',
            'net': 'vgg',
            'variant': 'shift_tolerant',
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
            'num_classes': 10,
            'base_model_name': 'inception_resnet_v2',
        },
        'metric_mode': 'NR',
    },
    'nima-koniq': {
        'metric_opts': {
            'type': 'NIMA',
            'train_dataset': 'koniq',
            'num_classes': 1,
            'base_model_name': 'inception_resnet_v2',
        },
        'metric_mode': 'NR',
    },
    'nima-spaq': {
        'metric_opts': {
            'type': 'NIMA',
            'train_dataset': 'spaq',
            'num_classes': 1,
            'base_model_name': 'inception_resnet_v2',
        },
        'metric_mode': 'NR',
    },
    'nima-vgg16-ava': {
        'metric_opts': {
            'type': 'NIMA',
            'num_classes': 10,
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
        'metric_mode': 'NR',
        'lower_better': True,
    },
    'maniqa': {
        'metric_opts': {
            'type': 'MANIQA',
            'train_dataset': 'koniq',
            'scale': 0.8,
        },
        'metric_mode': 'NR',
    },
    'maniqa-koniq': {
        'metric_opts': {
            'type': 'MANIQA',
            'train_dataset': 'koniq',
            'scale': 0.8,
        },
        'metric_mode': 'NR',
    },
    'maniqa-pipal': {
        'metric_opts': {
            'type': 'MANIQA',
            'train_dataset': 'pipal',
        },
        'metric_mode': 'NR',
    },
    'maniqa-kadid': {
        'metric_opts': {
            'type': 'MANIQA',
            'train_dataset': 'kadid',
            'scale': 0.8,
        },
        'metric_mode': 'NR',
    },
    'clipiqa': {
        'metric_opts': {
            'type': 'CLIPIQA',
        },
        'metric_mode': 'NR',
    },
    'clipiqa+': {
        'metric_opts': {
            'type': 'CLIPIQA',
            'model_type': 'clipiqa+',
        },
        'metric_mode': 'NR',
    },
    'clipiqa+_vitL14_512': {
        'metric_opts': {
            'type': 'CLIPIQA',
            'model_type': 'clipiqa+_vitL14_512',
            'backbone': 'ViT-L/14',
            'pos_embedding': True,
        },
        'metric_mode': 'NR',
    },
    'clipiqa+_rn50_512': {
        'metric_opts': {
            'type': 'CLIPIQA',
            'model_type': 'clipiqa+_rn50_512',
            'backbone': 'RN50',
            'pos_embedding': True,
        },
        'metric_mode': 'NR',
    },
    'tres': {
        'metric_opts': {
            'type': 'TReS',
            'train_dataset': 'koniq',
        },
        'metric_mode': 'NR',
    },
    'tres-koniq': {
        'metric_opts': {
            'type': 'TReS',
            'train_dataset': 'koniq',
        },
        'metric_mode': 'NR',
    },
    'tres-flive': {
        'metric_opts': {
            'type': 'TReS',
            'train_dataset': 'flive',
        },
        'metric_mode': 'NR',
    },
    'hyperiqa': {
        'metric_opts': {
            'type': 'HyperNet',
        },
        'metric_mode': 'NR',
    },
    'uranker': {
        'metric_opts': {
            'type': 'URanker',
        },
        'metric_mode': 'NR',
    },
    'clipscore': {
        'metric_opts': {
            'type': 'CLIPScore',
        },
        'metric_mode': 'NR',  # Caption image similarity
    },
    'entropy': {
        'metric_opts': {
            'type': 'Entropy',
        },
        'metric_mode': 'NR',
    },
    'topiq_nr': {
        'metric_opts': {
            'type': 'CFANet',
            'semantic_model_name': 'resnet50',
            'model_name': 'cfanet_nr_koniq_res50',
            'use_ref': False,
        },
        'metric_mode': 'NR',
    },
    'topiq_nr-flive': {
        'metric_opts': {
            'type': 'CFANet',
            'semantic_model_name': 'resnet50',
            'model_name': 'cfanet_nr_flive_res50',
            'use_ref': False,
        },
        'metric_mode': 'NR',
    },
    'topiq_nr-spaq': {
        'metric_opts': {
            'type': 'CFANet',
            'semantic_model_name': 'resnet50',
            'model_name': 'cfanet_nr_spaq_res50',
            'use_ref': False,
        },
        'metric_mode': 'NR',
    },
    'topiq_nr-face': {
        'metric_opts': {
            'type': 'CFANet',
            'semantic_model_name': 'resnet50',
            'model_name': 'topiq_nr_gfiqa_res50',
            'use_ref': False,
            'test_img_size': 512,
        },
        'metric_mode': 'NR',
    },
    'topiq_fr': {
        'metric_opts': {
            'type': 'CFANet',
            'semantic_model_name': 'resnet50',
            'model_name': 'cfanet_fr_kadid_res50',
            'use_ref': True,
        },
        'metric_mode': 'FR',
    },
    'topiq_fr-pipal': {
        'metric_opts': {
            'type': 'CFANet',
            'semantic_model_name': 'resnet50',
            'model_name': 'cfanet_fr_pipal_res50',
            'use_ref': True,
        },
        'metric_mode': 'FR',
    },
    'topiq_iaa': {
        'metric_opts': {
            'type': 'CFANet',
            'semantic_model_name': 'swin_base_patch4_window12_384',
            'model_name': 'cfanet_iaa_ava_swin',
            'use_ref': False,
            'inter_dim': 512,
            'num_heads': 8,
            'num_class': 10,
        },
        'metric_mode': 'NR',
    },
    'topiq_iaa_res50': {
        'metric_opts': {
            'type': 'CFANet',
            'semantic_model_name': 'resnet50',
            'model_name': 'cfanet_iaa_ava_res50',
            'use_ref': False,
            'inter_dim': 512,
            'num_heads': 8,
            'num_class': 10,
            'test_img_size': 384,
        },
        'metric_mode': 'NR',
    },
    'laion_aes': {
        'metric_opts': {
            'type': 'LAIONAes',
        },
        'metric_mode': 'NR',
    },
    'liqe': {
            'metric_opts': {
                'type': 'LIQE',
                'pretrained': 'koniq'
            },
            'metric_mode': 'NR',
        },
    'liqe_mix': {
            'metric_opts': {
                'type': 'LIQE',
                'pretrained': 'mix'
            },
            'metric_mode': 'NR',
        },
})
