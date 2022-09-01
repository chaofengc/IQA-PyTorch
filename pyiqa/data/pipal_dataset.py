import numpy as np
import pickle
from PIL import Image

import torch
from torch.utils import data as data
import torchvision.transforms as tf
from torchvision.transforms.functional import normalize

from pyiqa.data.data_util import read_meta_info_file 
from pyiqa.data.transforms import transform_mapping, augment, PairedToTensor
from pyiqa.utils import FileClient, imfrombytes, img2tensor
from pyiqa.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class PIPALDataset(data.Dataset):
    """General Full Reference dataset with meta info file.
    
    Args:
        opt (dict): Config for train datasets with the following keys:
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PIPALDataset, self).__init__()
        self.opt = opt

        if opt.get('override_phase', None) is None:
            self.phase = opt['phase'] 
        else:
            self.phase = opt['override_phase'] 

        target_img_folder = opt['dataroot_target']
        ref_img_folder = opt.get('dataroot_ref', None)
        self.paths_mos = read_meta_info_file(target_img_folder, opt['meta_info_file'], mode='fr', ref_dir=ref_img_folder) 

        # read train/val/test splits
        split_file_path = opt.get('split_file', None)
        if split_file_path:
            split_index = opt.get('split_index', 1)
            with open(opt['split_file'], 'rb') as f:
                split_dict = pickle.load(f)
                splits = split_dict[split_index][self.phase]
            
            self.paths_mos = [self.paths_mos[i] for i in splits] 
        
        dmos_max = opt.get('dmos_max', 0.)
        if dmos_max:
            self.use_dmos = True
            self.dmos_max = opt.get('dmos_max') 
        else:
            self.use_dmos = False

        # do paired transform first and then do common transform
        paired_transform_list = []
        augment_dict = opt.get('augment', None)
        if augment_dict is not None:
            for k, v in augment_dict.items():
                paired_transform_list += transform_mapping(k, v)
        self.paired_trans = tf.Compose(paired_transform_list)

        common_transform_list = []
        self.img_range = opt.get('img_range', 1.0)
        common_transform_list += [
                PairedToTensor(),
                ]
        self.common_trans = tf.Compose(common_transform_list)


    def __getitem__(self, index):

        ref_path = self.paths_mos[index][0]
        img_path = self.paths_mos[index][1]
        mos_label = self.paths_mos[index][2]

        img_pil = Image.open(img_path).convert('RGB')
        ref_pil = Image.open(ref_path).convert('RGB')

        img_pil, ref_pil = self.paired_trans([img_pil, ref_pil])

        img_tensor = self.common_trans(img_pil) * self.img_range
        ref_tensor = self.common_trans(ref_pil) * self.img_range
        if self.use_dmos:
            mos_label = self.dmos_max - mos_label 
        mos_label_tensor = torch.Tensor([mos_label])
        
        return {'img': img_tensor, 'ref_img': ref_tensor, 'mos_label': mos_label_tensor, 'img_path': img_path, 'ref_img_path': ref_path}

    def __len__(self):
        return len(self.paths_mos)
