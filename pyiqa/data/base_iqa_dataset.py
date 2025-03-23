import pandas as pd
import pickle

from torch.utils import data as data
import torchvision.transforms as tf

from pyiqa.data.transforms import transform_mapping, PairedToTensor
from pyiqa.utils import get_root_logger


class BaseIQADataset(data.Dataset):
    """General No Reference dataset with meta info file.

    Args:
        opt (dict): Config for train datasets with the following keys:
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        self.opt = opt
        self.logger = get_root_logger()

        if opt.get('override_phase', None) is None:
            self.phase = opt.get('phase', 'train')
        else:
            self.phase = opt['override_phase']

        assert self.phase in ['train', 'val', 'test'], (
            f'phase should be in [train, val, test], got {self.phase}'
        )

        # initialize datasets
        self.init_path_mos(opt)

        # mos normalization
        self.mos_normalize(opt)

        # read train/val/test splits
        self.get_split(opt)

        # get transforms
        self.get_transforms(opt)

    def init_path_mos(self, opt):
        self.meta_info = pd.read_csv(opt['meta_info_file'])
        self.paths_mos = self.meta_info.values.tolist()

    def get_split_with_file(self, opt):
        # read train/val/test splits
        split_file_path = opt.get('split_file', None)
        if split_file_path:
            split_index = opt.get('split_index', 1)
            with open(opt['split_file'], 'rb') as f:
                split_dict = pickle.load(f)
                splits = split_dict[split_index][self.phase]
            self.paths_mos = [self.paths_mos[i] for i in splits]

    def get_split(self, opt):
        """Read train/val/test splits"""
        # compatible with previous version using split file
        if opt.get('split_file', None) is not None:
            self.get_split_with_file(opt)
            return

        # get all split column names
        all_split_lists = [x for x in self.meta_info.columns.tolist() if 'split' in x]

        split_index = opt.get('split_index', None)

        if split_index is not None:
            if isinstance(split_index, str):
                split_name = split_index
            elif isinstance(split_index, int):
                split_ratio = opt.get('split_ratio', '802')
                split_name = f'ratio{split_ratio}_seed123_split_{split_index:02d}'

            assert split_name in all_split_lists, (
                f'The given split [{split_name}] is not available in {all_split_lists}'
            )

            split_paths_mos = []
            for i in range(len(self.paths_mos)):
                if self.meta_info[split_name][i] == self.phase:
                    split_paths_mos.append(self.paths_mos[i])
            self.paths_mos = split_paths_mos

    def mos_normalize(self, opt):
        mos_range = opt.get('mos_range', None)
        mos_lower_better = opt.get('lower_better', None)
        mos_normalize = opt.get('mos_normalize', False)

        if mos_normalize:
            assert mos_range is not None and mos_lower_better is not None, (
                'mos_range and mos_lower_better should be provided when mos_normalize is True'
            )

            def normalize(mos_label):
                mos_label = (mos_label - mos_range[0]) / (mos_range[1] - mos_range[0])
                # convert to higher better if lower better is true
                if mos_lower_better:
                    mos_label = 1 - mos_label
                return mos_label

            for item in self.paths_mos:
                item[1] = normalize(float(item[1]))
            self.logger.info(
                f'mos_label is normalized from {mos_range}, lower_better[{mos_lower_better}] to [0, 1], lower_better[False(higher better)].'
            )

    def get_transforms(self, opt):
        transform_list = []
        augment_dict = opt.get('augment', None)
        if augment_dict is not None:
            for k, v in augment_dict.items():
                transform_list += transform_mapping(k, v)

        self.img_range = opt.get('img_range', 1.0)
        transform_list += [
            PairedToTensor(),
        ]
        self.trans = tf.Compose(transform_list)

    def __getitem__(self, index):
        pass

    def __len__(self):
        return len(self.paths_mos)
