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
    info_file = '../datasets/koniq10k/koniq10k_distributions_sets.csv'

    save_meta_path = './datasets/meta_info/meta_info_KonIQ10kDataset.csv'
    split_info = {'train': [], 'val': [], 'test': []}
    with open(info_file, 'r') as f, open(save_meta_path, 'w+') as sf:
        csvreader = csv.reader(f)
        head = next(csvreader)

        csvwriter = csv.writer(sf)
        new_head = ['img_name', 'mos', 'std', 'split', 'c1', 'c2', 'c3', 'c4', 'c5', 'c_total']
        csvwriter.writerow(new_head)
        for idx, row in enumerate(csvreader):
            print(row)
            split = row[9]
            if split == 'training':
                split = 'train'
                row[9] = 0
            elif split == 'validation':
                split = 'val'
                row[9] = 1
            elif split == 'test':
                row[9] = 2
            split_info[split].append(idx)
            new_row = [row[0]] + row[7:10] + row[1:7]
            print(new_row)
            csvwriter.writerow(new_row)

    save_split_path = './datasets/meta_info/koniq10k_official.pkl'
    with open(save_split_path, 'wb') as sf:
        pickle.dump({1: split_info}, sf)


if __name__ == '__main__':
    get_meta_info()
    #  get_random_splits()
