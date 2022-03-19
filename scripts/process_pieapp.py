import os
import scipy.io as sio
import random
import numpy
import pickle
import csv
import pandas as pd
from tqdm import tqdm


def get_meta_info():

    root_path = '../datasets/PieAPP_dataset_CVPR_2018/'
    train_list_file = '../datasets/PieAPP_dataset_CVPR_2018/train_reference_list.txt'
    val_list_file = '../datasets/PieAPP_dataset_CVPR_2018/val_reference_list.txt'
    test_list_file = '../datasets/PieAPP_dataset_CVPR_2018/test_reference_list.txt'

    train_ref_list = [x.strip() for x in open(train_list_file).readlines()]
    val_ref_list = [x.strip() for x in open(val_list_file).readlines()]
    test_ref_list = [x.strip() for x in open(test_list_file).readlines()]

    save_meta_path = './datasets/meta_info/meta_info_PieAPPDataset.csv'
    split_info = {
        1: {
            'train': [],
            'val': [],
            'test': []
        },
    }

    with open(save_meta_path, 'w') as sf:
        csvwriter = csv.writer(sf)
        head = [
            'ref_img_path', 'dist_imgA_path', 'dist_imgB_path', 'raw preference for A', 'processed preference for A',
            'per_img score for dist_imgB', 'split'
        ]
        csvwriter.writerow(head)

        count = 0

        split = 'train'

        splits_str = ['train', 'val', 'test']
        split_lists = [train_ref_list, val_ref_list, test_ref_list]
        split_flags = [0, 1, 2]

        for sp_str, sp_ls, sp_flag in zip(splits_str, split_lists, split_flags):
            for ref_name in sp_ls:
                ref_raw_name = ref_name.split('.')[0]
                label_path = os.path.join(root_path, 'labels', sp_str, f'{ref_raw_name}_pairwise_labels.csv')
                name_label = pd.read_csv(label_path)

                if sp_str == 'test':
                    test_file_label = os.path.join(root_path, 'labels', sp_str, f'{ref_raw_name}_per_image_score.csv')
                    test_label = pd.read_csv(test_file_label)

                for i in range(name_label.shape[0]):
                    row = name_label.loc[i].tolist()
                    ref_path = f'reference_images/{sp_str}/{row[0]}'
                    if 'ref' in row[1]:
                        distA_path = f'reference_images/{sp_str}/{row[1]}'
                    else:
                        distA_path = f'distorted_images/{sp_str}/{ref_raw_name}/{row[1]}'
                    distB_path = f'distorted_images/{sp_str}/{ref_raw_name}/{row[2]}'

                    if sp_str == 'train':
                        new_row = [ref_path, distA_path, distB_path] + row[3:5] + [''] + [sp_flag]
                    elif sp_str == 'val':
                        new_row = [ref_path, distA_path, distB_path] + [row[3], '', '', sp_flag]
                    elif sp_str == 'test':
                        dist_keys = test_label[' distorted image'].tolist()
                        dist_scores = test_label[' score for distorted image']
                        dist_score_dict = {k: v for k, v in zip(dist_keys, dist_scores)}
                        per_img_score = dist_score_dict[row[2]]
                        new_row = [ref_path, distA_path, distB_path] + [row[3], '', per_img_score, sp_flag]
                    csvwriter.writerow(new_row)
                    split_info[1][sp_str].append(count)
                    count += 1

    save_split_path = './datasets/meta_info/pieapp_official.pkl'
    with open(save_split_path, 'wb') as sf:
        pickle.dump(split_info, sf)


if __name__ == '__main__':
    get_meta_info()
