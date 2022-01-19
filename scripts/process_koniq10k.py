import os
import scipy.io as sio
import random
import numpy
import pickle
import csv
from tqdm import tqdm

def get_meta_info():
    """
    Train/Val/Test split file from official github: 
        https://github.com/subpic/koniq/blob/master/metadata/koniq10k_distributions_sets.csv
    """
    info_file = '../../datasets/koniq10k/koniq10k_distributions_sets.csv'
    
    save_meta_path = '../pyiqa/data/meta_info/meta_info_KonIQ10kDataset.csv'
    split_info = {'train': [], 'val': [], 'test': []}
    with open(info_file, 'r') as f, open(save_meta_path, 'w+') as sf:
        csvreader = csv.reader(f)
        head = next(csvreader)

        csvwriter = csv.writer(sf)
        new_head = ['img_name', 'mos', 'std', 'split']
        csvwriter.writerow(new_head)
        for idx, row in enumerate(csvreader):
            print(row)
            img_name = row[0] 
            mos = row[7]
            std = row[8]
            split = row[9]
            if split == 'training':
                split = 'train'
            elif split == 'validation':
                split = 'val'
            split_info[split].append(idx)
            csvwriter.writerow([img_name, mos, std, split])

    save_split_path = '../pyiqa/data/train_split_info/koniq10k_official.pkl'
    with open(save_split_path, 'wb') as sf:
        pickle.dump({1: split_info}, sf)
    
if __name__ == '__main__':
    get_meta_info()
    #  get_random_splits()
