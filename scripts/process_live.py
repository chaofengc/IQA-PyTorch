import os
import scipy.io as sio
import random
import numpy
import pickle
import csv
import pandas as pd


def get_meta_info():
    root_dir = '../datasets/LIVEIQA_release2/'

    dmos = sio.loadmat(os.path.join(root_dir, 'dmos.mat'))  # difference of mos: test - ref. lower is better
    mos = dmos['dmos'][0]
    org_flag = dmos['orgs'][0]

    refnames = sio.loadmat(os.path.join(root_dir, 'refnames_all.mat'))
    refnames = refnames['refnames_all'][0]

    sub_folders = ['jp2k'] * 227 + ['jpeg'] * 233 + ['wn'] * 174 + ['gblur'] * 174 + ['fastfading'] * 174
    sub_indexes = list(range(1, 228)) + list(range(1, 234)) + list(range(1, 175)) * 3

    save_meta_path = './datasets/meta_info/meta_info_LIVEIQADataset.csv'
    with open(save_meta_path, 'w') as f:
        csvwriter = csv.writer(f)
        header = ['ref_name', 'dist_name', 'mos']
        csvwriter.writerow(header)
        for i in range(len(sub_folders)):
            ref_name = f'refimgs/{refnames[i][0]}'
            dis_name = f'{sub_folders[i]}/img{sub_indexes[i]}.bmp'
            tmpmos = mos[i]
            if org_flag[i] != 1:
                csvwriter.writerow([ref_name, dis_name, tmpmos])


def get_random_splits(seed=123):
    random.seed(seed)
    meta_info_file = './datasets/meta_info/meta_info_LIVEIQADataset.csv'
    save_path = f'./datasets/meta_info/live_{seed}.pkl'
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
