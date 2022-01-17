import numpy as np
import pickle
from PIL import Image

import torch
from torch.utils import data as data
import torchvision.transforms as tf
from torchvision.transforms.functional import normalize

from pyiqa.data.data_util import read_meta_info_file 
from pyiqa.data.transforms import transform_mapping, augment, paired_random_crop
from pyiqa.utils import FileClient, imfrombytes, img2tensor
from pyiqa.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class GeneralFRDataset(data.Dataset):
    """General Full Reference dataset with meta info file.
    
    Args:
        opt (dict): Config for train datasets with the following keys:
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(GeneralFRDataset, self).__init__()
        self.opt = opt

        target_img_folder = opt['dataroot_target']
        ref_img_folder = opt.get('dataroot_ref', None)
        self.paths_mos = read_meta_info_file(target_img_folder, opt['meta_info_file'], mode='fr', ref_dir=ref_img_folder) 

        # read train/val/test splits
        split_file_path = opt.get('split_file', None)
        if split_file_path:
            with open(opt['split_file'], 'rb') as f:
                split_dict = pickle.load(f)
                splits = split_dict[opt['split_index']][opt['phase']]
            self.paths_mos = [self.paths_mos[i] for i in splits] 

        # TODO: paired transform
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
        ref_path = self.paths_mos[index][1]
        mos_label = self.paths_mos[index][2]
        img_pil = Image.open(img_path)
        ref_pil = Image.open(ref_path)

        img_tensor = self.trans(img_pil)
        ref_tensor = self.trans(ref_pil)
        mos_label_tensor = torch.Tensor([mos_label])
        
        return {'img': img_tensor, 'ref_img': ref_tensor, 'mos_label': mos_label_tensor, 'img_path': img_path, 'ref_img_path': ref_path}

    def __len__(self):
        return len(self.paths_mos)
