import pickle
from PIL import Image
import os

import torch
from torch.utils import data as data

from pyiqa.utils.registry import DATASET_REGISTRY
from .base_iqa_dataset import BaseIQADataset

import pandas as pd


@DATASET_REGISTRY.register()
class BAPPSDataset(BaseIQADataset):
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

    def init_path_mos(self, opt):
        if opt.get('override_phase', None) is None:
            self.phase = opt['phase']
        else:
            self.phase = opt['override_phase']

        self.dataset_mode = opt.get('mode', '2afc')

        target_img_folder = opt['dataroot_target']
        self.dataroot = target_img_folder

        self.paths_mos = pd.read_csv(opt['meta_info_file']).values.tolist()

    def get_split(self, opt):
        super().get_split(opt)

        val_types = opt.get('val_types', None)

        if self.dataset_mode == '2afc':
            self.paths_mos = [x for x in self.paths_mos if x[0] != 'jnd']
        elif self.dataset_mode == 'jnd':
            self.paths_mos = [x for x in self.paths_mos if x[0] == 'jnd']

        if val_types is not None:
            tmp_paths_mos = []
            for item in self.paths_mos:
                for vt in val_types:
                    if vt in item[1]:
                        tmp_paths_mos.append(item)
            self.paths_mos = tmp_paths_mos

    def __getitem__(self, index):
        is_jnd_data = self.paths_mos[index][0] == 'jnd'
        distA_path = os.path.join(self.dataroot, self.paths_mos[index][1])
        distB_path = os.path.join(self.dataroot, self.paths_mos[index][2])

        distA_pil = Image.open(distA_path).convert('RGB')
        distB_pil = Image.open(distB_path).convert('RGB')

        score = self.paths_mos[index][3]
        # original 0 means prefer p0, transfer to probability of p0
        mos_label_tensor = torch.Tensor([score])

        if not is_jnd_data:
            ref_path = os.path.join(self.dataroot, self.paths_mos[index][0])
            ref_img_pil = Image.open(ref_path).convert('RGB')

            distA_tensor, distB_tensor, ref_tensor = self.trans(
                [distA_pil, distB_pil, ref_img_pil]
            )
        else:
            distA_tensor, distB_tensor = self.trans([distA_pil, distB_pil])

        if not is_jnd_data:
            return {
                'ref_img': ref_tensor,
                'distB_img': distB_tensor,
                'distA_img': distA_tensor,
                'mos_label': mos_label_tensor,
                'img_path': ref_path,
                'distB_path': distB_path,
                'distA_path': distA_path,
            }
        else:
            return {
                'distB_img': distB_tensor,
                'distA_img': distA_tensor,
                'mos_label': mos_label_tensor,
                'distB_path': distB_path,
                'distA_path': distA_path,
            }
