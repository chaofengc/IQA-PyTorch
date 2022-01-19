import os
import random
import numpy
import pickle
import csv

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
    root_dir = '../../datasets/tid2013/'
    save_meta_path = '../pyiqa/data/meta_info/meta_info_TID2013Dataset.csv'
    get_meta_info(root_dir, save_meta_path)

    root_dir = '../../datasets/tid2008/'
    save_meta_path = '../pyiqa/data/meta_info/meta_info_TID2008Dataset.csv'
    get_meta_info(root_dir, save_meta_path)
    #  get_random_splits()
