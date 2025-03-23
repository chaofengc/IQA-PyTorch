import pickle
from PIL import Image
import os

import torch
from torch.utils import data as data
from pyiqa.utils.registry import DATASET_REGISTRY

import pandas as pd
from .general_fr_dataset import GeneralFRDataset


@DATASET_REGISTRY.register()
class PieAPPDataset(GeneralFRDataset):
    """The PieAPP Dataset introduced by:

    Prashnani, Ekta and Cai, Hong and Mostofi, Yasamin and Sen, Pradeep
    PieAPP: Perceptual Image-Error Assessment Through Pairwise Preference
    CVPR2018
    url: http://civc.ucsb.edu/graphics/Papers/CVPR2018_PieAPP/

    Args:
        opt (dict): Config for train datasets with the following keys:
            phase (str): 'train' or 'val'.
    """

    def init_path_mos(self, opt):
        self.dataroot = opt['dataroot_target']
        if self.phase == 'test':
            metadata = pd.read_csv(
                opt['meta_info_file'],
                usecols=[
                    'ref_img_path',
                    'dist_imgB_path',
                    'per_img score for dist_imgB',
                ],
            )
        else:
            metadata = pd.read_csv(opt['meta_info_file'])
        self.paths_mos = metadata.values.tolist()

    def get_split(self, opt):
        super().get_split(opt)
        # remove duplicates
        if self.phase == 'test':
            temp = []
            for item in self.paths_mos:
                if not item in temp:
                    temp.append(item)
            self.paths_mos = temp

    def __getitem__(self, index):
        ref_path = os.path.join(self.dataroot, self.paths_mos[index][0])
        if self.phase == 'test':
            distB_path = os.path.join(self.dataroot, self.paths_mos[index][1])
        else:
            distA_path = os.path.join(self.dataroot, self.paths_mos[index][1])
            distB_path = os.path.join(self.dataroot, self.paths_mos[index][2])

        distB_pil = Image.open(distB_path).convert('RGB')
        ref_img_pil = Image.open(ref_path).convert('RGB')

        if self.phase != 'test':
            distA_pil = Image.open(distA_path).convert('RGB')

            distA_pil, distB_pil, ref_img_pil = self.paired_trans(
                [distA_pil, distB_pil, ref_img_pil]
            )

            distA_tensor, distB_tensor, ref_tensor = self.common_trans(
                [distA_pil, distB_pil, ref_img_pil]
            )
        else:
            distB_pil, ref_img_pil = self.paired_trans([distB_pil, ref_img_pil])
            distB_tensor, ref_tensor = self.common_trans([distB_pil, ref_img_pil])

        if self.phase == 'train':
            score = self.paths_mos[index][4]
            mos_label_tensor = torch.Tensor([score])
            distB_score = torch.Tensor([-1])
        elif self.phase == 'val':
            score = self.paths_mos[index][4]
            mos_label_tensor = torch.Tensor([score])
            distB_score = torch.Tensor([-1])
        elif self.phase == 'test':
            per_img_score = self.paths_mos[index][2]
            distB_score = torch.Tensor([per_img_score])

        if self.phase == 'test':
            return {
                'img': distB_tensor,
                'ref_img': ref_tensor,
                'mos_label': distB_score,
                'img_path': distB_path,
                'ref_img_path': ref_path,
            }
        else:
            return {
                'distB_img': distB_tensor,
                'ref_img': ref_tensor,
                'distA_img': distA_tensor,
                'mos_label': mos_label_tensor,
                'distB_per_img_score': distB_score,
                'distB_path': distB_path,
                'ref_img_path': ref_path,
                'distA_path': distA_path,
            }
