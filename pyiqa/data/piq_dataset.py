import torch
import os
import csv
from PIL import Image

from pyiqa.data.data_util import read_meta_info_file 
from pyiqa.utils.registry import DATASET_REGISTRY
from pyiqa.utils import get_root_logger
from .general_nr_dataset import GeneralNRDataset


@DATASET_REGISTRY.register()
class PIQDataset(GeneralNRDataset):
    """General No Reference dataset with meta info file.
    """
    def init_path_mos(self, opt):
        logger = get_root_logger()
        target_img_folder = opt['dataroot_target']
        attr = opt.get('attribute', 'Overall')

        assert attr in ['Details', 'Exposure', 'Overall'], f'attribute should be in [Details, Exposure, Overall], got {attr}'

        logger.info(f'Training on PIQ2023 dataset with attribute [{attr}]')

        with open(opt['meta_info_file'], 'r') as fin:
            csvreader = csv.reader(fin)
            name_mos = list(csvreader)[1:]
        
        self.paths_mos = name_mos

        self.paths_mos = []
        for item in name_mos:
            if attr in item[0]:
                item[0] = os.path.join(target_img_folder, item[0])
                self.paths_mos.append(item)
    
    def get_split(self, opt):
        """Get split for PIQ2023 dataset:
            1: device split
            2: scene split 
        """
        logger = get_root_logger()
        split_index = opt.get('split_index', None)
        if split_index is not None:
            assert split_index in [1, 2], f'split indexes should be, 1: device split; 2: scene split'
            assert self.phase in ['train', 'test'], f'PIQDataset has no {self.phase} split'
            
            logger.info(f'Training on PIQ2023 dataset with split [{split_index}](1: device split; 2: scene split)')

            new_paths_mos = []
            for item in self.paths_mos:
                if self.phase == 'train' and item[split_index - 3] == 'Train':
                    new_paths_mos.append(item)
                elif self.phase == 'test' and item[split_index - 3] == 'Test':
                    new_paths_mos.append(item)
            
            self.paths_mos = new_paths_mos

    def __getitem__(self, index):

        img_path = self.paths_mos[index][0]
        mos_label = float(self.paths_mos[index][1])
        img_pil = Image.open(img_path).convert('RGB')

        img_tensor = self.trans(img_pil) * self.img_range
        mos_label_tensor = torch.Tensor([mos_label])

        scene_idx = int(self.paths_mos[index][-4])
                
        return {'img': img_tensor, 'mos_label': mos_label_tensor, 'img_path': img_path, 'scene_idx': scene_idx}
   