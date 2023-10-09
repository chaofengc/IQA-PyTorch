import os
import scipy.io as sio
import random
import numpy
import pickle
import csv
import pandas as pd


def get_random_splits(seed=123):
    random.seed(seed)
    

    meta_info_file = '../datasets/meta_info/meta_info_GFIQADataset.csv'
    meta_info = pd.read_csv(meta_info_file)
    img_list = meta_info['img_name'].tolist()

    total_num = len(img_list) 

    all_img_index = list(range(total_num))
    num_splits = 10
    save_path = '../datasets/meta_info/gfiqa_seed123.pkl'

    ratio = [0.7, 0.1, 0.2]  # train/val/test

    split_info = {}
    for i in range(num_splits):
        random.shuffle(all_img_index)
        sep1 = int(total_num * ratio[0])
        sep2 = sep1 + int(total_num * ratio[1])
        split_info[i + 1] = {
            'train': all_img_index[:sep1], 
            'val': all_img_index[sep1:sep2],
            'test': all_img_index[sep2:] 
            }

    with open(save_path, 'wb') as sf:
        pickle.dump(split_info, sf)


if __name__ == '__main__':
    get_random_splits()
