import os
import scipy.io as sio
import random
import numpy
import pickle
import csv
import pandas as pd


def get_meta_info():
    root_dir = '../datasets/LIVEmultidistortiondatabase/'

    parts = ['Part 1', 'Part 2']
    sub_img_folders = ['blurjpeg', 'blurnoise']

    save_meta_path = './datasets/meta_info/meta_info_LIVEMDDataset.csv'
    f = open(save_meta_path, 'w')
    csvwriter = csv.writer(f)
    header = ['ref_name', 'dist_name', 'dmos']
    csvwriter.writerow(header)
    for p, subf in zip(parts, sub_img_folders):
        sub_root_dir = os.path.join(root_dir, p)

        img_list = sio.loadmat(os.path.join(sub_root_dir, 'Imagelists.mat'))
        dist_names = [x[0][0] for x in img_list['distimgs']]

        score = sio.loadmat(os.path.join(sub_root_dir, 'Scores.mat'))
        alldmos = score['DMOSscores'][0]

        for i in range(len(dist_names)):
            dis_name = f'{p}/{subf}/{dist_names[i]}'
            ref_name = f"{p}/{subf}/{dist_names[i].split('_')[0]}.bmp"
            dmos = alldmos[i]
            msg = f'{ref_name:<15}\t{dis_name:<15}\t{dmos:<15}\n'
            csvwriter.writerow([ref_name, dis_name, dmos])

    f.close()


def get_random_splits(seed=123):
    random.seed(seed)
    meta_info_file = './datasets/meta_info/meta_info_LIVEMDDataset.csv'
    save_path = f'./datasets/meta_info/livemd_{seed}.pkl'
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
