import os
import scipy.io as sio
import random
import numpy
import pickle
import csv
import pandas as pd


def get_meta_info():
    root_dir = '../datasets/CSIQ/'
    label_file = '../datasets/CSIQ/csiq_label.txt'

    name_dmos = [x.strip().split() for x in open(label_file).readlines()]

    save_meta_path = './datasets/meta_info/meta_info_CSIQDataset.csv'
    with open(save_meta_path, 'w') as f:
        writer = csv.writer(f)
        header = ['ref_name', 'dist_name', 'dmos']
        writer.writerow(header)
        for dis_name, dmos in name_dmos:
            ref_name = f"{dis_name.split('.')[0]}.png"
            writer.writerow([ref_name, dis_name, dmos])


def get_random_splits(seed=123):
    random.seed(seed)
    meta_info_file = './datasets/meta_info/meta_info_CSIQDataset.csv'
    save_path = f'./datasets/meta_info/csiq_{seed}.pkl'
    ratio = 0.8

    meta_info = pd.read_csv(meta_info_file)

    ref_img_list = list(set(meta_info['ref_name'].tolist()))
    ref_img_num = len(ref_img_list)
    num_splits = 10
    train_num = int(ratio * ref_img_num)

    split_info = {}
    for i in range(num_splits):
        split_info[i + 1] = {'train': [], 'val': [], 'test': []}

    for i in range(num_splits):
        random.shuffle(ref_img_list)
        train_ref_img_names = ref_img_list[:train_num]
        for j in range(meta_info.shape[0]):
            tmp_ref_name = meta_info.loc[j]['ref_name']
            if tmp_ref_name in train_ref_img_names:
                split_info[i + 1]['train'].append(j)
            else:
                split_info[i + 1]['val'].append(j)
        print(meta_info.shape[0], len(split_info[i + 1]['train']), len(split_info[i + 1]['val']))
    with open(save_path, 'wb') as sf:
        pickle.dump(split_info, sf)


if __name__ == '__main__':
    # get_meta_info()
    get_random_splits()
