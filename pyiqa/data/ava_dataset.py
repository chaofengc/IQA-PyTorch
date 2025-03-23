import numpy as np
import pickle
from PIL import Image
import os

import torch
from torch.utils import data as data

from pyiqa.utils.registry import DATASET_REGISTRY
import pandas as pd

from .base_iqa_dataset import BaseIQADataset

# avoid possible image read error in AVA dataset
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


@DATASET_REGISTRY.register()
class AVADataset(BaseIQADataset):
    """AVA dataset, proposed by

    Murray, Naila, Luca Marchesotti, and Florent Perronnin.
    "AVA: A large-scale database for aesthetic visual analysis."
    In 2012 IEEE conference on computer vision and pattern recognition (CVPR), pp. 2408-2415. IEEE, 2012.

    Args:
        opt (dict): Config for train datasets with the following keys:
            phase (str): 'train' or 'val'.
    """

    def init_path_mos(self, opt):
        super().init_path_mos(opt)
        target_img_folder = opt['dataroot_target']
        self.dataroot = target_img_folder

    def get_split(self, opt):
        split_index = opt.get('split_index', None)

        # compatible with previous version using split file
        # when using split file, previous version will use official_split or split_index=1
        if opt.get('split_file', None) is not None:
            split_index = 'official_split'

        if split_index is not None:
            # use val_num for validation
            val_num = opt.get('val_num', 2000)

            train_split_paths_mos = []
            val_split_paths_mos = []
            test_split_paths_mos = []
            for i in range(len(self.paths_mos)):
                if self.meta_info[split_index][i] == 0:  # 0 for train
                    train_split_paths_mos.append(self.paths_mos[i])
                elif self.meta_info[split_index][i] == 1:  # 1 for val
                    val_split_paths_mos.append(self.paths_mos[i])
                elif self.meta_info[split_index][i] == 2:  # 2 for test
                    test_split_paths_mos.append(self.paths_mos[i])

            if len(val_split_paths_mos) < val_num:
                val_num = val_num - len(val_split_paths_mos)
                val_split_paths_mos = (
                    val_split_paths_mos + train_split_paths_mos[-val_num:]
                )
                train_split_paths_mos = train_split_paths_mos[:-val_num]
            else:
                train_split_paths_mos = (
                    train_split_paths_mos + val_split_paths_mos[:-val_num]
                )
                val_split_paths_mos = val_split_paths_mos[-val_num:]

            if self.phase == 'train':
                self.paths_mos = train_split_paths_mos
            elif self.phase == 'val':
                self.paths_mos = val_split_paths_mos
            elif self.phase == 'test':
                self.paths_mos = test_split_paths_mos

        self.mean_mos = np.array([item[1] for item in self.paths_mos]).mean()

    def __getitem__(self, index):
        img_path = os.path.join(self.dataroot, self.paths_mos[index][0])
        mos_label = self.paths_mos[index][1]
        mos_dist = self.paths_mos[index][2:12]
        img_pil = Image.open(img_path).convert('RGB')
        width, height = img_pil.size

        img_tensor = self.trans(img_pil)
        img_tensor2 = self.trans(img_pil)
        mos_label_tensor = torch.Tensor([mos_label]) / 10.0
        mos_dist_tensor = torch.Tensor(mos_dist) / sum(mos_dist)

        if self.opt.get('list_imgs', False):
            tmp_tensor = torch.zeros((img_tensor.shape[0], 800, 800))
            h, w = img_tensor.shape[1:]
            tmp_tensor[..., :h, :w] = img_tensor
            return {
                'img': tmp_tensor,
                'mos_label': mos_label_tensor,
                'mos_dist': mos_dist_tensor,
                'org_size': torch.tensor([height, width]),
                'img_path': img_path,
                'mean_mos': torch.tensor(self.mean_mos),
            }
        else:
            return {
                'img': img_tensor,
                'img2': img_tensor2,
                'mos_label': mos_label_tensor,
                'mos_dist': mos_dist_tensor,
                'org_size': torch.tensor([height, width]),
                'img_path': img_path,
                'mean_mos': torch.tensor(self.mean_mos),
            }
