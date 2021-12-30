import numpy as np
import pickle
from PIL import Image

import torch
from torch.utils import data as data
import torchvision.transforms as tf
from torchvision.transforms.functional import normalize

from pyiqa.data.data_util import paths_mos_from_meta_info_file
from pyiqa.data.transforms import transform_mapping, augment, paired_random_crop
from pyiqa.utils import FileClient, imfrombytes, img2tensor
from pyiqa.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class LIVEChallengeDataset(data.Dataset):
    """The LIVE Challenge Dataset introduced by

    D. Ghadiyaram and A.C. Bovik, 
    "Massive Online Crowdsourced Study of Subjective and Objective Picture Quality," 
    IEEE Transactions on Image Processing, 2016
    url: https://live.ece.utexas.edu/research/ChallengeDB/index.html 
    
    Args:
        opt (dict): Config for train datasets with the following keys:
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(LIVEChallengeDataset, self).__init__()
        self.opt = opt

        target_img_folder = opt['dataroot_target']
        self.paths_mos = paths_mos_from_meta_info_file(target_img_folder, opt['meta_info_file']) 
        # remove first 7 training images as previous works
        self.paths_mos = self.paths_mos[7:] 

        # read train/val/test splits
        with open(opt['split_file'], 'rb') as f:
            split_dict = pickle.load(f)
            splits = split_dict[opt['split_index']][opt['phase']]
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

        img_path = self.paths_mos[index][0]
        mos_label = self.paths_mos[index][1]
        img_pil = Image.open(img_path)

        img_tensor = self.trans(img_pil)
        mos_label_tensor = torch.Tensor([mos_label])
        
        return {'img': img_tensor, 'mos_label': mos_label_tensor, 'img_path': img_path}

    def __len__(self):
        return len(self.paths_mos)
