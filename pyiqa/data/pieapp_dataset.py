import pickle
from PIL import Image
import os

import torch
from torch.utils import data as data
import torchvision.transforms as tf

from pyiqa.data.transforms import transform_mapping
from pyiqa.utils.registry import DATASET_REGISTRY

import pandas as pd


@DATASET_REGISTRY.register()
class PieAPPDataset(data.Dataset):
    """The PieAPP Dataset introduced by:

    Prashnani, Ekta and Cai, Hong and Mostofi, Yasamin and Sen, Pradeep
    PieAPP: Perceptual Image-Error Assessment Through Pairwise Preference
    CVPR2018
    url: http://civc.ucsb.edu/graphics/Papers/CVPR2018_PieAPP/

    Args:
        opt (dict): Config for train datasets with the following keys:
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PieAPPDataset, self).__init__()
        self.opt = opt
        self.phase = opt['phase']

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

        ref_path = os.path.join(self.dataroot, self.paths_mos[index][0])
        distA_path = os.path.join(self.dataroot, self.paths_mos[index][1])
        distB_path = os.path.join(self.dataroot, self.paths_mos[index][2])

        ref_img_pil = Image.open(ref_path).convert('RGB')
        distA_pil = Image.open(distA_path).convert('RGB')
        distB_pil = Image.open(distB_path).convert('RGB')

        if self.phase == 'train':
            score = self.paths_mos[index][4]
            mos_label_tensor = torch.Tensor([score])
            distB_score = torch.Tensor([-1])
        elif self.phase == 'val':
            score = self.paths_mos[index][3]
            mos_label_tensor = torch.Tensor([score])
            distB_score = torch.Tensor([-1])
        elif self.phase == 'test':
            score = self.paths_mos[index][3]
            per_img_score = self.paths_mos[index][5]
            mos_label_tensor = torch.Tensor([score])
            distB_score = torch.Tensor([per_img_score])

        ref_tensor = self.trans(ref_img_pil)
        distA_tensor = self.trans(distA_pil)
        distB_tensor = self.trans(distB_pil)

        return {
            'distB_img': distB_tensor,
            'ref_img': ref_tensor,
            'distA_img': distA_tensor,
            'mos_label': mos_label_tensor,
            'distB_per_img_score': distB_score,
            'distB_path': distB_path,
            'ref_img_path': ref_path,
            'distA_path': distA_path
        }

    def __len__(self):
        return len(self.paths_mos)
