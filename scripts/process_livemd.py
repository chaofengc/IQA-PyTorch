import os
import scipy.io as sio
import random
import numpy
import pickle

def get_meta_info():
    root_dir = '../../datasets/LIVEmultidistortiondatabase/'

    parts = ['Part1', 'Part2']
    sub_img_folders = ['blurjpeg', 'blurnoise']

    save_meta_path = '../pyiqa/data/meta_info/meta_info_LIVEMDDataset.txt'
    f = open(save_meta_path, 'w') 
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
            #  print(msg)
            f.write(msg)

    f.close()

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
