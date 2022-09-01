import numpy as np
import pickle
from PIL import Image
import cv2
import os
import random
import itertools

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
            
            # use val_num for validation 
            val_num = 2000
            train_split = split_dict[split_index]['train'] 
            val_split = split_dict[split_index]['val'] 
            train_split = train_split + val_split[:-val_num]
            val_split = val_split[-val_num:]
            split_dict[split_index]['train'] = train_split
            split_dict[split_index]['val'] = val_split 

            if opt.get('override_phase', None) is None:
                splits = split_dict[split_index][opt['phase']]
            else:
                splits = split_dict[split_index][opt['override_phase']]
            
            self.paths_mos = [self.paths_mos[i] for i in splits] 
        
        self.mean_mos = np.array([item[1] for item in self.paths_mos]).mean()

        # self.paths_mos.sort(key=lambda x: x[1])
        # n = 32
        # n = 4
        # tmp_list = [self.paths_mos[i: i + n] for i in range(0, len(self.paths_mos), n)]
        # random.shuffle(tmp_list)
        # self.paths_mos = list(itertools.chain.from_iterable(tmp_list))

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
        width, height = img_pil.size        

        img_tensor = self.trans(img_pil)
        img_tensor2 = self.trans(img_pil)
        mos_label_tensor = torch.Tensor([mos_label])
        mos_dist_tensor = torch.Tensor(mos_dist) / sum(mos_dist)

        if self.opt.get('list_imgs', False):
            tmp_tensor = torch.zeros((img_tensor.shape[0], 800, 800)) 
            h, w = img_tensor.shape[1:]
            tmp_tensor[..., :h, :w] = img_tensor
            return {'img': tmp_tensor, 'mos_label': mos_label_tensor, 'mos_dist': mos_dist_tensor, 'org_size': torch.tensor([height, width]), 'img_path': img_path, 'mean_mos': torch.tensor(self.mean_mos)}
        else:
            return {'img': img_tensor, 'img2': img_tensor2, 'mos_label': mos_label_tensor, 'mos_dist': mos_dist_tensor, 'org_size': torch.tensor([height, width]), 'img_path': img_path, 'mean_mos': torch.tensor(self.mean_mos)}

    def __len__(self):
        return len(self.paths_mos)
