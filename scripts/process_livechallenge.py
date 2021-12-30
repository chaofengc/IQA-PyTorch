import os
import scipy.io as sio
import random
import numpy
import pickle

def get_meta_info():
    root_dir = '../../datasets/LIVEC/'
    names = sio.loadmat(os.path.join(root_dir, 'Data', 'AllImages_release.mat')) 
    mos_labels = sio.loadmat(os.path.join(root_dir, 'Data', 'AllMOS_release.mat'))
    mos_std = sio.loadmat(os.path.join(root_dir, 'Data', 'AllStdDev_release.mat'))

    img_names = names['AllImages_release']
    mos_labels = mos_labels['AllMOS_release'][0]
    mos_std = mos_std['AllStdDev_release'][0]

    save_meta_path = '../datasets/meta_info_LIVEChallengeDataset.txt'
    with open(save_meta_path, 'w') as f:
        for idx, name_item in enumerate(img_names):
            img_name = name_item[0][0]
            mos = mos_labels[idx]
            std = mos_std[idx]
            f.write(f'{img_name:<10}\t{mos:<20}\t{std}\n')


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
    #  get_meta_info()
    get_random_splits()
