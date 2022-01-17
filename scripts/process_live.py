import os
import scipy.io as sio
import random
import numpy
import pickle

def get_meta_info():
    root_dir = '../../datasets/LIVEIQA_release2/'

    sub_folders = ['jp2k', 'jpeg', 'wn', 'gblur', 'fastfading'] 

    save_meta_path = '../pyiqa/data/meta_info/meta_info_LIVEIQADataset.txt'
    with open(save_meta_path, 'w') as f:
        for subf in sub_folders:
            subf_info = [x.strip().split() for x in open(os.path.join(root_dir, subf, 'info.txt')).readlines()]
            for si in subf_info:
                if len(si):
                    ref_name = si[0]
                    dis_name = si[1]
                    mos = float(si[2])
                    if mos != 0:
                        ref_name = f'refimgs/{ref_name}'
                        name = f'{subf}/{dis_name}'
                        msg = f'{ref_name:<15}\t{name:<15}\t{mos:<15}\n'
                        f.write(msg)

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
