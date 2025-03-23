from collections import OrderedDict

# IMPORTANT NOTES !!!
#   - The score range (min, max) is only rough estimation, the actual score range may vary.

DEFAULT_CONFIGS = OrderedDict(
    {
        'ahiq': {
            'metric_opts': {
                'type': 'AHIQ',
            },
            'metric_mode': 'FR',
            'score_range': '~0, ~1',
        },
        'ckdn': {
            'metric_opts': {
                'type': 'CKDN',
            },
            'metric_mode': 'FR',
            'score_range': '0, 1',
        },
        'lpips': {
            'metric_opts': {
                'type': 'LPIPS',
                'net': 'alex',
                'version': '0.1',
            },
            'metric_mode': 'FR',
            'lower_better': True,
            'score_range': '0, 1',
        },
        'lpips-vgg': {
            'metric_opts': {
                'type': 'LPIPS',
                'net': 'vgg',
                'version': '0.1',
            },
            'metric_mode': 'FR',
            'lower_better': True,
            'score_range': '0, 1',
        },
        'lpips+': {
            'metric_opts': {
                'type': 'LPIPS',
                'net': 'alex',
                'version': '0.1',
                'semantic_weight_layer': 2,
            },
            'metric_mode': 'FR',
            'lower_better': True,
            'score_range': '0, 1',
        },
        'lpips-vgg+': {
            'metric_opts': {
                'type': 'LPIPS',
                'net': 'vgg',
                'version': '0.1',
                'semantic_weight_layer': 2,
            },
            'metric_mode': 'FR',
            'lower_better': True,
            'score_range': '0, 1',
        },
        'stlpips': {
            'metric_opts': {
                'type': 'STLPIPS',
                'net': 'alex',
                'variant': 'shift_tolerant',
            },
            'metric_mode': 'FR',
            'lower_better': True,
            'score_range': '0, 1',
        },
        'stlpips-vgg': {
            'metric_opts': {
                'type': 'STLPIPS',
                'net': 'vgg',
                'variant': 'shift_tolerant',
            },
            'metric_mode': 'FR',
            'lower_better': True,
            'score_range': '0, 1',
        },
        'dists': {
            'metric_opts': {
                'type': 'DISTS',
            },
            'metric_mode': 'FR',
            'lower_better': True,
            'score_range': '0, 1',
        },
        'deepdc': {
            'metric_opts': {
                'type': 'DeepDC',
            },
            'metric_mode': 'FR',
            'lower_better': True,
            'score_range': '0, 1',
        },
        'ssim': {
            'metric_opts': {
                'type': 'SSIM',
                'downsample': False,
                'test_y_channel': True,
            },
            'metric_mode': 'FR',
            'score_range': '0, 1',
        },
        'ssimc': {
            'metric_opts': {
                'type': 'SSIM',
                'downsample': False,
                'test_y_channel': False,
            },
            'metric_mode': 'FR',
            'score_range': '0, 1',
        },
        'psnr': {
            'metric_opts': {
                'type': 'PSNR',
                'test_y_channel': False,
            },
            'metric_mode': 'FR',
            'score_range': '~0, ~40',
        },
        'psnry': {
            'metric_opts': {
                'type': 'PSNR',
                'test_y_channel': True,
            },
            'metric_mode': 'FR',
            'score_range': '~0, ~60',
        },
        'fsim': {
            'metric_opts': {
                'type': 'FSIM',
                'chromatic': True,
            },
            'metric_mode': 'FR',
            'score_range': '0, 1',
        },
        'ms_ssim': {
            'metric_opts': {
                'type': 'MS_SSIM',
                'downsample': False,
                'test_y_channel': True,
                'is_prod': True,
            },
            'metric_mode': 'FR',
            'score_range': '0, 1',
        },
        'vif': {
            'metric_opts': {
                'type': 'VIF',
            },
            'metric_mode': 'FR',
            'score_range': '0, ~1',
        },
        'gmsd': {
            'metric_opts': {
                'type': 'GMSD',
                'test_y_channel': True,
            },
            'metric_mode': 'FR',
            'lower_better': True,
            'score_range': '0, ~1',
        },
        'nlpd': {
            'metric_opts': {
                'type': 'NLPD',
                'channels': 1,
                'test_y_channel': True,
            },
            'metric_mode': 'FR',
            'lower_better': True,
            'score_range': '0, 1',
        },
        'vsi': {
            'metric_opts': {
                'type': 'VSI',
            },
            'metric_mode': 'FR',
            'score_range': '0, ~1',
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
            'score_range': '0, 1',
        },
        'mad': {
            'metric_opts': {
                'type': 'MAD',
            },
            'metric_mode': 'FR',
            'lower_better': True,
            'score_range': '0, ~',
        },
        # =============================================================
        'piqe': {
            'metric_opts': {
                'type': 'PIQE',
            },
            'metric_mode': 'NR',
            'lower_better': True,
            'score_range': '0, 100',
        },
        'niqe': {
            'metric_opts': {
                'type': 'NIQE',
                'test_y_channel': True,
            },
            'metric_mode': 'NR',
            'lower_better': True,
            'score_range': '~0, ~100',
        },
        'niqe_matlab': {
            'metric_opts': {
                'type': 'NIQE',
                'test_y_channel': True,
                'version': 'matlab',
            },
            'metric_mode': 'NR',
            'lower_better': True,
            'score_range': '~0, ~100',
        },
        'ilniqe': {
            'metric_opts': {
                'type': 'ILNIQE',
            },
            'metric_mode': 'NR',
            'lower_better': True,
            'score_range': '~0, ~100',
        },
        'brisque': {
            'metric_opts': {
                'type': 'BRISQUE',
                'test_y_channel': True,
            },
            'metric_mode': 'NR',
            'lower_better': True,
            'score_range': '~0, ~150',
        },
        'brisque_matlab': {
            'metric_opts': {
                'type': 'BRISQUE',
                'test_y_channel': True,
                'version': 'matlab',
            },
            'metric_mode': 'NR',
            'lower_better': True,
            'score_range': '~0, ~100',
        },
        'nrqm': {
            'metric_opts': {
                'type': 'NRQM',
            },
            'metric_mode': 'NR',
            'score_range': '~0, ~10',
        },
        'pi': {
            'metric_opts': {
                'type': 'PI',
            },
            'metric_mode': 'NR',
            'lower_better': True,
            'score_range': '~0, ~',
        },
        'cnniqa': {
            'metric_opts': {'type': 'CNNIQA', 'pretrained': 'koniq10k'},
            'metric_mode': 'NR',
            'score_range': '~0, ~1',
        },
        'musiq': {
            'metric_opts': {'type': 'MUSIQ', 'pretrained': 'koniq10k'},
            'metric_mode': 'NR',
            'score_range': '~0, ~100',
        },
        'musiq-ava': {
            'metric_opts': {'type': 'MUSIQ', 'pretrained': 'ava'},
            'metric_mode': 'NR',
            'score_range': '1, 10',
        },
        'musiq-paq2piq': {
            'metric_opts': {'type': 'MUSIQ', 'pretrained': 'paq2piq'},
            'metric_mode': 'NR',
            'score_range': '~0, ~100',
        },
        'musiq-spaq': {
            'metric_opts': {'type': 'MUSIQ', 'pretrained': 'spaq'},
            'metric_mode': 'NR',
            'score_range': '~0, ~100',
        },
        'nima': {
            'metric_opts': {
                'type': 'NIMA',
                'num_classes': 10,
                'base_model_name': 'inception_resnet_v2',
            },
            'metric_mode': 'NR',
            'score_range': '0, 10',
        },
        'nima-koniq': {
            'metric_opts': {
                'type': 'NIMA',
                'train_dataset': 'koniq',
                'num_classes': 1,
                'base_model_name': 'inception_resnet_v2',
            },
            'metric_mode': 'NR',
            'score_range': '~0, ~1',
        },
        'nima-spaq': {
            'metric_opts': {
                'type': 'NIMA',
                'train_dataset': 'spaq',
                'num_classes': 1,
                'base_model_name': 'inception_resnet_v2',
            },
            'metric_mode': 'NR',
            'score_range': '~0, ~1',
        },
        'nima-vgg16-ava': {
            'metric_opts': {
                'type': 'NIMA',
                'num_classes': 10,
                'base_model_name': 'vgg16',
            },
            'metric_mode': 'NR',
            'score_range': '0, 10',
        },
        'pieapp': {
            'metric_opts': {
                'type': 'PieAPP',
            },
            'metric_mode': 'FR',
            'lower_better': True,
            'score_range': '~0, ~5',
        },
        'paq2piq': {
            'metric_opts': {
                'type': 'PAQ2PIQ',
            },
            'metric_mode': 'NR',
            'score_range': '~0, ~100',
        },
        'dbcnn': {
            'metric_opts': {'type': 'DBCNN', 'pretrained': 'koniq'},
            'metric_mode': 'NR',
            'score_range': '~0, ~1',
        },
        'fid': {
            'metric_opts': {
                'type': 'FID',
            },
            'metric_mode': 'NR',
            'lower_better': True,
            'score_range': '0, ~',
        },
        'fid_dinov2': {
            'metric_opts': {
                'type': 'FID',
                'backbone': 'dinov2',
            },
            'metric_mode': 'NR',
            'lower_better': True,
            'score_range': '0, ~',
        },
        'maniqa': {
            'metric_opts': {
                'type': 'MANIQA',
                'train_dataset': 'koniq',
                'scale': 0.8,
            },
            'metric_mode': 'NR',
            'score_range': '~0, ~1',
        },
        'maniqa-pipal': {
            'metric_opts': {
                'type': 'MANIQA',
                'train_dataset': 'pipal',
            },
            'metric_mode': 'NR',
            'score_range': '~0, ~1',
        },
        'maniqa-kadid': {
            'metric_opts': {
                'type': 'MANIQA',
                'train_dataset': 'kadid',
                'scale': 0.8,
            },
            'metric_mode': 'NR',
            'score_range': '~0, ~1',
        },
        'clipiqa': {
            'metric_opts': {
                'type': 'CLIPIQA',
            },
            'metric_mode': 'NR',
            'score_range': '0, 1',
        },
        'clipiqa+': {
            'metric_opts': {
                'type': 'CLIPIQA',
                'model_type': 'clipiqa+',
            },
            'metric_mode': 'NR',
            'score_range': '0, 1',
        },
        'clipiqa+_vitL14_512': {
            'metric_opts': {
                'type': 'CLIPIQA',
                'model_type': 'clipiqa+_vitL14_512',
                'backbone': 'ViT-L/14',
                'pos_embedding': True,
            },
            'metric_mode': 'NR',
            'score_range': '0, 1',
        },
        'clipiqa+_rn50_512': {
            'metric_opts': {
                'type': 'CLIPIQA',
                'model_type': 'clipiqa+_rn50_512',
                'backbone': 'RN50',
                'pos_embedding': True,
            },
            'metric_mode': 'NR',
            'score_range': '0, 1',
        },
        'tres': {
            'metric_opts': {
                'type': 'TReS',
                'train_dataset': 'koniq',
            },
            'metric_mode': 'NR',
            'score_range': '~0, ~100',
        },
        'tres-flive': {
            'metric_opts': {
                'type': 'TReS',
                'train_dataset': 'flive',
            },
            'metric_mode': 'NR',
            'score_range': '~0, ~100',
        },
        'hyperiqa': {
            'metric_opts': {
                'type': 'HyperNet',
            },
            'metric_mode': 'NR',
            'score_range': '~0, ~1',
        },
        'uranker': {
            'metric_opts': {
                'type': 'URanker',
            },
            'metric_mode': 'NR',
            'score_range': '~-1, ~2',
        },
        'clipscore': {
            'metric_opts': {
                'type': 'CLIPScore',
            },
            'metric_mode': 'NR',  # Caption image similarity
            'score_range': '0, 2.5',
        },
        'entropy': {
            'metric_opts': {
                'type': 'Entropy',
            },
            'metric_mode': 'NR',
            'score_range': '0, 8',
        },
        'topiq_nr': {
            'metric_opts': {
                'type': 'CFANet',
                'semantic_model_name': 'resnet50',
                'model_name': 'cfanet_nr_koniq_res50',
                'use_ref': False,
            },
            'metric_mode': 'NR',
            'score_range': '~0, ~1',
        },
        'topiq_nr-flive': {
            'metric_opts': {
                'type': 'CFANet',
                'semantic_model_name': 'resnet50',
                'model_name': 'cfanet_nr_flive_res50',
                'use_ref': False,
                'test_img_size': 384,
            },
            'metric_mode': 'NR',
            'score_range': '~0, ~1',
        },
        'topiq_nr-spaq': {
            'metric_opts': {
                'type': 'CFANet',
                'semantic_model_name': 'resnet50',
                'model_name': 'cfanet_nr_spaq_res50',
                'use_ref': False,
            },
            'metric_mode': 'NR',
            'score_range': '~0, ~1',
        },
        'topiq_nr-face': {
            'metric_opts': {
                'type': 'CFANet',
                'semantic_model_name': 'resnet50',
                'model_name': 'topiq_nr_cgfiqa_res50',
                'use_ref': False,
                'test_img_size': 512,
            },
            'metric_mode': 'NR',
            'score_range': '~0, ~1',
        },
        'topiq_nr_swin-face': {
            'metric_opts': {
                'type': 'CFANet',
                'semantic_model_name': 'swin_base_patch4_window12_384',
                'model_name': 'topiq_nr_cgfiqa_swin',
                'use_ref': False,
                'test_img_size': 384,
            },
            'metric_mode': 'NR',
            'score_range': '~0, ~1',
        },
        'topiq_nr-face-v1': {
            'metric_opts': {
                'type': 'CFANet',
                'semantic_model_name': 'resnet50',
                'model_name': 'topiq_nr_gfiqa_res50',
                'use_ref': False,
                'test_img_size': 512,
            },
            'metric_mode': 'NR',
            'score_range': '~0, ~1',
        },
        'topiq_fr': {
            'metric_opts': {
                'type': 'CFANet',
                'semantic_model_name': 'resnet50',
                'model_name': 'cfanet_fr_kadid_res50',
                'use_ref': True,
            },
            'metric_mode': 'FR',
            'score_range': '~0, ~1',
        },
        'topiq_fr-pipal': {
            'metric_opts': {
                'type': 'CFANet',
                'semantic_model_name': 'resnet50',
                'model_name': 'cfanet_fr_pipal_res50',
                'use_ref': True,
            },
            'metric_mode': 'FR',
            'score_range': '~0, ~1',
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
            'score_range': '1, 10',
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
            'score_range': '1, 10',
        },
        'laion_aes': {
            'metric_opts': {
                'type': 'LAIONAes',
            },
            'metric_mode': 'NR',
            'score_range': '~1, ~10',
        },
        'liqe': {
            'metric_opts': {'type': 'LIQE', 'pretrained': 'koniq'},
            'metric_mode': 'NR',
            'score_range': '1, 5',
        },
        'liqe_mix': {
            'metric_opts': {'type': 'LIQE', 'pretrained': 'mix'},
            'metric_mode': 'NR',
            'score_range': '1, 5',
        },
        'wadiqam_fr': {
            'metric_opts': {
                'type': 'WaDIQaM',
                'metric_type': 'FR',
                'model_name': 'wadiqam_fr_kadid',
            },
            'metric_mode': 'FR',
            'score_range': '~-1, ~0.1',
        },
        'wadiqam_nr': {
            'metric_opts': {
                'type': 'WaDIQaM',
                'metric_type': 'NR',
                'model_name': 'wadiqam_nr_koniq',
            },
            'metric_mode': 'NR',
            'score_range': '~-1, ~0.1',
        },
        'qalign': {
            'metric_opts': {
                'type': 'QAlign',
            },
            'metric_mode': 'NR',
            'score_range': '1, 5',
        },
        'qalign_8bit': {
            'metric_opts': {
                'type': 'QAlign',
                'dtype': '8bit',
            },
            'metric_mode': 'NR',
            'score_range': '1, 5',
        },
        'qalign_4bit': {
            'metric_opts': {
                'type': 'QAlign',
                'dtype': '4bit',
            },
            'metric_mode': 'NR',
            'score_range': '1, 5',
        },
        'compare2score': {
            'metric_opts': {
                'type': 'Compare2Score',
            },
            'metric_mode': 'NR',
            'score_range': '0, 100',
        },
        'unique': {
            'metric_opts': {
                'type': 'UNIQUE',
            },
            'metric_mode': 'NR',
            'score_range': '~-3, ~3',
        },
        'inception_score': {
            'metric_opts': {
                'type': 'InceptionScore',
            },
            'metric_mode': 'NR',
            'lower_better': False,
            'score_range': '0, ~',
        },
        'arniqa': {
            'metric_opts': {
                'type': 'ARNIQA',
                'regressor_dataset': 'koniq',
            },
            'metric_mode': 'NR',
            'score_range': '0, 1',
        },
        'arniqa-live': {
            'metric_opts': {
                'type': 'ARNIQA',
                'regressor_dataset': 'live',
            },
            'metric_mode': 'NR',
            'score_range': '0, 1',
        },
        'arniqa-csiq': {
            'metric_opts': {
                'type': 'ARNIQA',
                'regressor_dataset': 'csiq',
            },
            'metric_mode': 'NR',
            'score_range': '0, 1',
        },
        'arniqa-tid': {
            'metric_opts': {
                'type': 'ARNIQA',
                'regressor_dataset': 'tid',
            },
            'metric_mode': 'NR',
            'score_range': '0, 1',
        },
        'arniqa-kadid': {
            'metric_opts': {
                'type': 'ARNIQA',
                'regressor_dataset': 'kadid',
            },
            'metric_mode': 'NR',
            'score_range': '0, 1',
        },
        'arniqa-clive': {
            'metric_opts': {
                'type': 'ARNIQA',
                'regressor_dataset': 'clive',
            },
            'metric_mode': 'NR',
            'score_range': '0, 1',
        },
        'arniqa-flive': {
            'metric_opts': {
                'type': 'ARNIQA',
                'regressor_dataset': 'flive',
            },
            'metric_mode': 'NR',
            'score_range': '0, 1',
        },
        'arniqa-spaq': {
            'metric_opts': {
                'type': 'ARNIQA',
                'regressor_dataset': 'spaq',
            },
            'metric_mode': 'NR',
            'score_range': '0, 1',
        },
        'msswd': {
            'metric_opts': {
                'type': 'MS_SWD_learned',
            },
            'metric_mode': 'FR',
            'score_range': '0, ~10',
            'lower_better': True,
        },
        'qualiclip': {
            'metric_opts': {
                'type': 'QualiCLIP',
                'model_type': 'qualiclip',
            },
            'metric_mode': 'NR',
            'score_range': '0, 1',
        },
        'qualiclip+': {
            'metric_opts': {
                'type': 'QualiCLIP',
                'model_type': 'qualiclip+',
                'pretrained': True,
            },
            'metric_mode': 'NR',
            'score_range': '0, 1',
        },
        'qualiclip+-clive': {
            'metric_opts': {
                'type': 'QualiCLIP',
                'model_type': 'qualiclip+-clive',
                'pretrained': True,
            },
            'metric_mode': 'NR',
            'score_range': '0, 1',
        },
        'qualiclip+-flive': {
            'metric_opts': {
                'type': 'QualiCLIP',
                'model_type': 'qualiclip+-flive',
                'pretrained': True,
            },
            'metric_mode': 'NR',
            'score_range': '0, 1',
        },
        'qualiclip+-spaq': {
            'metric_opts': {
                'type': 'QualiCLIP',
                'model_type': 'qualiclip+-spaq',
                'pretrained': True,
            },
            'metric_mode': 'NR',
            'score_range': '0, 1',
        },
    }
)
