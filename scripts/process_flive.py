import os
import scipy.io as sio
import random
import numpy
import pickle
import csv
import pandas as pd
from tqdm import tqdm
"""
    The FLIVE dataset introduced by:

        Zhenqiang Ying, Haoran Niu, Praful Gupta, Dhruv Mahajan, Deepti Ghadiyaram, Alan Bovik.
        "From Patches to Pictures (PaQ-2-PiQ): Mapping the Perceptual Space of Picture Quality.""
        CVPR2020.

    Reference github:
        [1] https://github.com/niu-haoran/FLIVE_Database
        [2] https://github.com/baidut/PaQ-2-PiQ

    Image/patch labels are in [1], please copy the following prepare script from [2] to [1]
    because there are bugs in the script of [1]

        https://github.com/baidut/PaQ-2-PiQ/blob/master/database_prep.ipynb

    Besides, the patch labels in [1] are not complete. 9 patches from EE371R are in

        https://github.com/baidut/PaQ-2-PiQ/tree/master/database/patches
"""


def get_meta_info():
    patch_label_file = '../../PaQ-2-PiQ/database/labels_patch.csv'
    img_label_file = '../../PaQ-2-PiQ/database/labels_image.csv'

    test_label_file = '../../PaQ-2-PiQ/database/labels640.csv'
    train_label_file = '../../PaQ-2-PiQ/database/labels=640_padded.csv'

    all_patch_label = pd.read_csv(patch_label_file)
    all_img_label = pd.read_csv(img_label_file)
    test_label = pd.read_csv(test_label_file)
    train_label = pd.read_csv(train_label_file)

    train_img_list = []
    val_img_list = []
    test_img_list = test_label['name_image'].tolist()

    for i in tqdm(range(train_label.shape[0])):
        name = train_label.loc[i]['name_image']
        is_valid = train_label.loc[i]['is_valid']
        if is_valid:
            val_img_list.append(name)

    test_img_key_list = [x.split('/')[1] for x in test_img_list]
    val_img_key_list = [x.split('/')[1] for x in val_img_list]

    save_meta_path = './datasets/meta_info/meta_info_FLIVEDataset.csv'
    split_info = {
        1: {
            'train': [],
            'val': [],
            'test': []
        },
    }

    with open(save_meta_path, 'w') as sf:
        csvwriter = csv.writer(sf)
        head = ['img_name/patch_name', 'mos', 'split']
        csvwriter.writerow(head)

        count = 0

        # get image info
        for i in tqdm(range(all_img_label.shape[0])):
            name = all_img_label.loc[i]['name']
            mos = all_img_label.loc[i]['mos']
            name_key = name.split('/')[1]
            if name in test_img_list:
                split = 2
                split_info[1]['test'].append(count)
            elif name in val_img_list:
                split = 1
                split_info[1]['val'].append(count)
            else:
                split = 0
                split_info[1]['train'].append(count)
            row = [name, mos, split]
            csvwriter.writerow(row)
            count += 1

        print(len(split_info[1]['train']), len(split_info[1]['val']), len(split_info[1]['test']))
        print(sum([len(split_info[1]['train']), len(split_info[1]['val']), len(split_info[1]['test'])]))

        # get patch info
        for i in tqdm(range(all_patch_label.shape[0])):
            name = all_patch_label.loc[i]['name']
            mos = all_patch_label.loc[i]['mos']
            name_key = name.split('/')[1].split('_patch_')[0]
            if name_key in test_img_key_list:
                split = 2
                split_info[1]['test'].append(count)
            elif name_key in val_img_key_list:
                split = 1
                split_info[1]['val'].append(count)
            else:
                split = 0
                split_info[1]['train'].append(count)
            row = [name, mos, split]
            csvwriter.writerow(row)
            count += 1

    print(all_img_label.shape[0], all_patch_label.shape[0])
    print(all_img_label.shape[0] + all_patch_label.shape[0])
    print(len(split_info[1]['train']), len(split_info[1]['val']), len(split_info[1]['test']))
    print(sum([len(split_info[1]['train']), len(split_info[1]['val']), len(split_info[1]['test'])]))
    save_split_path = './datasets/meta_info/flive_official.pkl'
    with open(save_split_path, 'wb') as sf:
        pickle.dump(split_info, sf)


if __name__ == '__main__':
    get_meta_info()
    #  get_random_splits()
