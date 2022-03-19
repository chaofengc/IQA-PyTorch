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
class BAPPSDataset(data.Dataset):
    """The BAPPS Dataset introduced by:

    Zhang, Richard and Isola, Phillip and Efros, Alexei A and Shechtman, Eli and Wang, Oliver
    The Unreasonable Effectiveness of Deep Features as a Perceptual Metric.
    CVPR2018
    url: https://github.com/richzhang/PerceptualSimilarity

    Args:
        opt (dict): Config for train datasets with the following keys:
            phase (str): 'train' or 'val'.
        mode (str):
            - 2afc: load 2afc triplet data
            - jnd: load jnd pair data
    """

    def __init__(self, opt):
        super(BAPPSDataset, self).__init__()
        self.opt = opt
        self.phase = opt['phase']
        self.dataset_mode = opt.get('mode', '2afc')

        target_img_folder = opt['dataroot_target']
        self.dataroot = target_img_folder

        self.paths_mos = pd.read_csv(opt['meta_info_file']).values.tolist()

        if self.dataset_mode == '2afc':
            self.paths_mos = [x for x in self.paths_mos if x[0] != 'jnd']
        elif self.dataset_mode == 'jnd':
            self.paths_mos = [x for x in self.paths_mos if x[0] == 'jnd']

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

        distA_path = os.path.join(self.dataroot, self.paths_mos[index][1])
        distB_path = os.path.join(self.dataroot, self.paths_mos[index][2])

        distA_pil = Image.open(distA_path).convert('RGB')
        distB_pil = Image.open(distB_path).convert('RGB')

        score = self.paths_mos[index][3]
        mos_label_tensor = torch.Tensor([score])
        distA_tensor = self.trans(distA_pil)
        distB_tensor = self.trans(distB_pil)

        is_jnd_data = self.paths_mos[index][0] == 'jnd'
        if not is_jnd_data:
            ref_path = os.path.join(self.dataroot, self.paths_mos[index][0])
            ref_img_pil = Image.open(ref_path).convert('RGB')
            ref_tensor = self.trans(ref_img_pil)

            return {
                'ref_img': ref_tensor,
                'distB_img': distB_tensor,
                'distA_img': distA_tensor,
                'mos_label': mos_label_tensor,
                'ref_img_path': ref_path,
                'distB_path': distB_path,
                'distA_path': distA_path
            }
        else:

            return {
                'distB_img': distB_tensor,
                'distA_img': distA_tensor,
                'mos_label': mos_label_tensor,
                'distB_path': distB_path,
                'distA_path': distA_path
            }

    def __len__(self):
        return len(self.paths_mos)
