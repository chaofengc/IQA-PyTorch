import os
import scipy.io as sio
import random
import numpy
import pickle
import csv

def get_meta_info():
    root_dir = '../../datasets/CSIQ/'
    label_file = '../../datasets/CSIQ/csiq_label.txt'
    
    name_dmos = [x.strip().split() for x in open(label_file).readlines()]
    
    #  save_meta_path = '../pyiqa/data/meta_info/meta_info_CSIQDataset.txt'
    #  with open(save_meta_path, 'w') as f:
        #  for dis_name, dmos in name_dmos:
            #  ref_name = f"{dis_name.split('.')[0]}.png" 
            #  msg = f'{ref_name:<15}\t{dis_name:<15}\t{dmos:<15}\n'
            #  f.write(msg)

    save_meta_path = '../pyiqa/data/meta_info/meta_info_CSIQDataset.csv'
    with open(save_meta_path, 'w') as f:
        writer = csv.writer(f)
        header = ['ref_name', 'dist_name', 'dmos']
        writer.writerow(header)
        for dis_name, dmos in name_dmos:
            ref_name = f"{dis_name.split('.')[0]}.png" 
            writer.writerow([ref_name, dis_name, dmos])

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
