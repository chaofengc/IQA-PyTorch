import os
import scipy.io as sio
import random
import numpy
import pickle
import csv

def get_meta_info():
    root_dir = '../../datasets/LIVEIQA_release2/'

    dmos = sio.loadmat(os.path.join(root_dir, 'dmos.mat')) # difference of mos: test - ref. lower is better
    mos = dmos['dmos'][0]
    org_flag = dmos['orgs'][0]

    refnames = sio.loadmat(os.path.join(root_dir, 'refnames_all.mat'))
    refnames = refnames['refnames_all'][0]

    sub_folders = ['jp2k']*227 + ['jpeg']*233 + ['wn']*174 + ['gblur']*174 + ['fastfading']*174 
    sub_indexes = list(range(1, 228)) + list(range(1, 234)) + list(range(1, 175)) * 3 

    save_meta_path = '../pyiqa/data/meta_info/meta_info_LIVEIQADataset.csv'
    with open(save_meta_path, 'w') as f:
        csvwriter = csv.writer(f)
        header = ['ref_name', 'dist_name','mos']
        csvwriter.writerow(header)
        for i in range(len(sub_folders)):
            ref_name = refnames[i][0]
            dis_name = f'{sub_folders[i]}/img{sub_indexes[i]}' 
            tmpmos = mos[i]
            if org_flag[i] != 1:
                csvwriter.writerow([ref_name, dis_name, tmpmos])

def get_random_splits(seed=123):
    random.seed(seed)
    all_img_index = list(range(1162))
    num_splits = 10
    save_path = f'../pyiqa/data/train_split_info/livechallenge_{seed}.pkl'

    ratio = [0.8, 0.2] # train/val/test
    sep_index = int(round(0.8 * 1162))
    
    split_info = {}
    for i in range(num_splits):
        random.shuffle(all_img_index)
        split_info[i+1] = {
                'train': all_img_index[:sep_index],
                'val': all_img_index[sep_index:]
                }
    with open(save_path, 'wb') as sf:
        pickle.dump(split_info, sf)

if __name__ == '__main__':
    get_meta_info()
    #  get_random_splits()
