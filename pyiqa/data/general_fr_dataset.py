from PIL import Image

import torch
from torch.utils import data as data
import torchvision.transforms as tf

from pyiqa.data.data_util import read_meta_info_file 
from pyiqa.data.transforms import transform_mapping, PairedToTensor
from pyiqa.utils.registry import DATASET_REGISTRY

from .base_iqa_dataset import BaseIQADataset

@DATASET_REGISTRY.register()
class GeneralFRDataset(BaseIQADataset):
    """General Full Reference dataset with meta info file.
    """
    
    def init_path_mos(self, opt):
        target_img_folder = opt['dataroot_target']
        ref_img_folder = opt.get('dataroot_ref', None)
        self.paths_mos = read_meta_info_file(target_img_folder, opt['meta_info_file'], mode='fr', ref_dir=ref_img_folder) 

    def get_transforms(self, opt):
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
    
    def mos_normalize(self, opt):
        mos_range = opt.get('mos_range', None)
        mos_lower_better = opt.get('lower_better', None)
        mos_normalize = opt.get('mos_normalize', False)

        if mos_normalize:
            assert mos_range is not None and mos_lower_better is not None, 'mos_range and mos_lower_better should be provided when mos_normalize is True'

            def normalize(mos_label):
                mos_label = (mos_label - mos_range[0]) / (mos_range[1] - mos_range[0])
                if mos_lower_better:
                    mos_label = 1 - mos_label
                return mos_label

            self.paths_mos = [item[:2] + [normalize(item[2])] for item in self.paths_mos]
            self.logger.info(f'mos_label is normalized from {mos_range}, lower_better[{mos_lower_better}] to [0, 1], higher better.')

    def __getitem__(self, index):

        ref_path = self.paths_mos[index][0]
        img_path = self.paths_mos[index][1]
        mos_label = self.paths_mos[index][2]
        img_pil = Image.open(img_path).convert('RGB')
        ref_pil = Image.open(ref_path).convert('RGB')
        
        img_pil, ref_pil = self.paired_trans([img_pil, ref_pil])

        img_tensor = self.common_trans(img_pil) * self.img_range
        ref_tensor = self.common_trans(ref_pil) * self.img_range
        mos_label_tensor = torch.Tensor([mos_label])
        
        return {'img': img_tensor, 'ref_img': ref_tensor, 'mos_label': mos_label_tensor, 'img_path': img_path, 'ref_img_path': ref_path}
