from PIL import Image
from os import path as osp

import torch
from torch.utils import data as data

from pyiqa.utils.registry import DATASET_REGISTRY
from .base_iqa_dataset import BaseIQADataset


@DATASET_REGISTRY.register()
class GeneralNRDataset(BaseIQADataset):
    """General No Reference dataset with meta info file."""

    def init_path_mos(self, opt):
        super().init_path_mos(opt)

        target_img_folder = opt['dataroot_target']

        self.paths_mos = []
        for row in self.meta_info.values:
            img_path = osp.join(target_img_folder, row[0])
            mos_label = float(row[1])
            self.paths_mos.append([img_path, mos_label])

    def __getitem__(self, index):
        img_path = self.paths_mos[index][0]
        mos_label = float(self.paths_mos[index][1])
        img_pil = Image.open(img_path).convert('RGB')

        img_tensor = self.trans(img_pil) * self.img_range
        mos_label_tensor = torch.Tensor([mos_label])

        return {'img': img_tensor, 'mos_label': mos_label_tensor, 'img_path': img_path}
