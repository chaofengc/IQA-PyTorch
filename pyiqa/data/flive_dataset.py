import numpy as np
import pickle
from PIL import Image
import cv2

import torch
from torch.utils import data as data
import torchvision.transforms as tf
from torchvision.transforms.functional import normalize

from pyiqa.data.data_util import read_meta_info_file
from pyiqa.data.transforms import transform_mapping 
from pyiqa.utils import FileClient, imfrombytes, img2tensor
from pyiqa.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class FLIVEDataset(data.Dataset):
    """General No Reference dataset with meta info file.

    Args:
        opt (dict): Config for train datasets with the following keys:
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(FLIVEDataset, self).__init__()
        self.opt = opt

        target_img_folder = opt['dataroot_target']
        self.paths_mos = read_meta_info_file(target_img_folder, opt['meta_info_file'])

        # read train/val/test splits
        split_file_path = opt.get('split_file', None)
        if split_file_path:
            split_index = opt.get('split_index', 1)
            with open(opt['split_file'], 'rb') as f:
                split_dict = pickle.load(f)
                if opt.get('override_phase', None) is None:
                    splits = split_dict[split_index][opt['phase']]
                else:
                    splits = split_dict[split_index][opt['override_phase']]

            if opt['phase'] == 'train':
                self.paths_mos = [self.paths_mos[i] for i in splits]
            else:
                # remove patches during validation and test
                self.paths_mos = [self.paths_mos[i] for i in splits]
                self.paths_mos = [[p, m] for p, m in self.paths_mos if not 'patches/' in p]

        dmos_max = opt.get('dmos_max', 0.)
        if dmos_max:
            self.use_dmos = True
            self.dmos_max = opt.get('dmos_max')
        else:
            self.use_dmos = False
        self.mos_max = opt.get('mos_max', 1.)

        transform_list = []
        augment_dict = opt.get('augment', None)
        if augment_dict is not None:
            for k, v in augment_dict.items():
                transform_list += transform_mapping(k, v)

        self.img_range = opt.get('img_range', 1.0)
        transform_list += [
            tf.ToTensor(),
        ]
        self.trans = tf.Compose(transform_list)

    def __getitem__(self, index):

        img_path = self.paths_mos[index][0]
        mos_label = self.paths_mos[index][1]
        img_pil = Image.open(img_path).convert('RGB')

        img_tensor = self.trans(img_pil) * self.img_range
        if self.use_dmos:
            mos_label = self.dmos_max - mos_label
        else:
            mos_label = mos_label / self.mos_max
        mos_label_tensor = torch.Tensor([mos_label])

        return {'img': img_tensor, 'mos_label': mos_label_tensor, 'img_path': img_path}

    def __len__(self):
        return len(self.paths_mos)
