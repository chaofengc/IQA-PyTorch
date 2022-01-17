import os
import random
import numpy
import pickle

def get_meta_info():
    root_dir = '../../datasets/tid2013/'
    mos_file = '../../datasets/tid2013/mos_with_names.txt'
    std_file = '../../datasets/tid2013/mos_std.txt'

    mos_names = [x.strip().split() for x in open(mos_file).readlines()]
    std = [x.strip() for x in open(std_file).readlines()]
    
    save_meta_path = '../pyiqa/data/meta_info/meta_info_TID2013Dataset.txt'
    with open(save_meta_path, 'w') as f:
        for idx, ((mos, name), std) in enumerate(zip(mos_names, std)):
            ref_name = f'I{name[1:3]}.BMP'
            ref_name = ref_name.replace('I25.BMP', 'i25.bmp')
            msg = f'{ref_name:<15}{name:<15}{mos:<15}{std}\n'
            #  msg = f'{ref_name}\t{name}\t{mos}\t{std}\n'
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
