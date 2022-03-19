import os
import random
import numpy
import pickle
import csv
import pandas as pd


def get_meta_info(root_dir, save_meta_path):
    mos_file = os.path.join(root_dir, 'mos_with_names.txt')
    std_file = os.path.join(root_dir, 'mos_std.txt')

    mos_names = [x.strip().split() for x in open(mos_file).readlines()]
    std = [x.strip() for x in open(std_file).readlines()]

    with open(save_meta_path, 'w') as f:
        csvwriter = csv.writer(f)
        header = ['ref_name', 'dist_name', 'mos', 'std']
        csvwriter.writerow(header)
        for idx, ((mos, name), std) in enumerate(zip(mos_names, std)):
            ref_name = f'I{name[1:3]}.BMP'
            ref_name = ref_name.replace('I25.BMP', 'i25.bmp')
            img_path = os.path.join(root_dir, 'distorted_images', name)
            if not os.path.exists(img_path):
                name = name.replace('i', 'I')
            csvwriter.writerow([ref_name, name, mos, std])


def get_random_splits(meta_info_file, save_path, seed=123):
    random.seed(seed)
    # meta_info_file = './datasets/meta_info/meta_info_CSIQDataset.csv'
    # save_path = f'./datasets/meta_info/csiq_{seed}.pkl'
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
    # root_dir = '../datasets/tid2013/'
    # save_meta_path = './datasets/meta_info/meta_info_TID2013Dataset.csv'
    # get_meta_info(root_dir, save_meta_path)

    # root_dir = '../datasets/tid2008/'
    # save_meta_path = './datasets/meta_info/meta_info_TID2008Dataset.csv'
    # get_meta_info(root_dir, save_meta_path)

    meta_info_file = './datasets/meta_info/meta_info_TID2013Dataset.csv'
    save_path = './datasets/meta_info/tid2013_seed123.pkl'
    get_random_splits(meta_info_file, save_path)

    meta_info_file = './datasets/meta_info/meta_info_TID2008Dataset.csv'
    save_path = './datasets/meta_info/tid2008_seed123.pkl'
    get_random_splits(meta_info_file, save_path)
