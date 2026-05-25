import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

from pyiqa.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class MetaIQADataset(Dataset):

    def __init__(self, opt):
        super(MetaIQADataset, self).__init__()
        self.opt = opt
        self.csv_file = opt.get('meta_info_file', opt.get('csv_file'))
        self.root_dir = opt.get('dataroot_target', opt.get('root_dir'))
        self.df = pd.read_csv(self.csv_file, sep=',')

        if self.opt['phase'] == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomCrop((224, 224)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])

        self.mos_normalize = opt.get('mos_normalize', True)
        self.mos_range = opt.get('mos_range', [0, 100])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, str(row['image']))
        img = Image.open(img_path).convert('RGB')

        img_tensor = self.transform(img)

        score_key = 'dmos' if 'dmos' in row else 'mos'
        rating = float(row[score_key])

        if self.mos_normalize:
            rating = (rating - self.mos_range[0]) / (self.mos_range[1] - self.mos_range[0])

        return {
            'target_img': img_tensor,
            'mos': torch.tensor([rating]).float().view(-1, 1)
        }