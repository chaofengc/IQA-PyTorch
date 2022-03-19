import numpy as np
import pickle
from PIL import Image
import cv2
import os

import torch
from torch.utils import data as data
import torchvision.transforms as tf

from pyiqa.data.transforms import transform_mapping
from pyiqa.utils.registry import DATASET_REGISTRY
import pandas as pd

# avoid possible image read error in AVA dataset
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


@DATASET_REGISTRY.register()
class AVADataset(data.Dataset):
    """AVA dataset, proposed by

    Murray, Naila, Luca Marchesotti, and Florent Perronnin.
    "AVA: A large-scale database for aesthetic visual analysis."
    In 2012 IEEE conference on computer vision and pattern recognition (CVPR), pp. 2408-2415. IEEE, 2012.

    Args:
        opt (dict): Config for train datasets with the following keys:
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(AVADataset, self).__init__()
        self.opt = opt

        target_img_folder = opt['dataroot_target']
        self.dataroot = target_img_folder
        self.paths_mos = pd.read_csv(opt['meta_info_file']).values.tolist()

        # read train/val/test splits
        split_file_path = opt.get('split_file', None)
        if split_file_path:
            split_index = opt.get('split_index', 1)
            with open(opt['split_file'], 'rb') as f:
                split_dict = pickle.load(f)
                splits = split_dict[split_index][opt['phase']]
            self.paths_mos = [self.paths_mos[i] for i in splits]

        transform_list = []
        augment_dict = opt.get('augment', None)
        if augment_dict is not None:
            for k, v in augment_dict.items():
                transform_list += transform_mapping(k, v)

        img_range = opt.get('img_range', 1.0)
        transform_list += [
            tf.ToTensor(),
            tf.Lambda(lambda x: x * img_range),
        ]
        self.trans = tf.Compose(transform_list)

    def __getitem__(self, index):

        img_path = os.path.join(self.dataroot, self.paths_mos[index][0])
        mos_label = self.paths_mos[index][1]
        mos_dist = self.paths_mos[index][2:12]
        img_pil = Image.open(img_path).convert('RGB')

        img_tensor = self.trans(img_pil)
        mos_label_tensor = torch.Tensor([mos_label])
        mos_dist_tensor = torch.Tensor(mos_dist) / sum(mos_dist)

        return {'img': img_tensor, 'mos_label': mos_label_tensor, 'mos_dist': mos_dist_tensor, 'img_path': img_path}

    def __len__(self):
        return len(self.paths_mos)
