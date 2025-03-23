import os

from pyiqa.utils.registry import DATASET_REGISTRY
from .general_nr_dataset import GeneralNRDataset


@DATASET_REGISTRY.register()
class LIVEChallengeDataset(GeneralNRDataset):
    """The LIVE Challenge Dataset introduced by

    D. Ghadiyaram and A.C. Bovik,
    "Massive Online Crowdsourced Study of Subjective and Objective Picture Quality,"
    IEEE Transactions on Image Processing, 2016
    url: https://live.ece.utexas.edu/research/ChallengeDB/index.html

    Args:
        opt (dict): Config for train datasets with the following keys:
            phase (str): 'train' or 'val'.
    """

    def init_path_mos(self, opt):
        super().init_path_mos(opt)
        # remove first 7 training images as previous works
        self.paths_mos = self.paths_mos[7:]
