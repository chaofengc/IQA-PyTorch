import os
import scipy.io as sio
import random
import numpy
import pickle
import csv
import pandas as pd
from tqdm import tqdm


def get_meta_info():
    mos_label_file = '../datasets/SPAQ/Annotations/MOS and Image attribute scores.xlsx'
    scene_label_file = '../datasets/SPAQ/Annotations/Scene category labels.xlsx'
    exif_label_file = '../datasets/SPAQ/Annotations/EXIF_tags.xlsx'

    mos_label = pd.read_excel(mos_label_file)
    scene_label = pd.read_excel(scene_label_file)
    exif_label = pd.read_excel(exif_label_file)

    new_head = mos_label.keys().tolist() + scene_label.keys().tolist()[1:] + exif_label.keys().tolist()[1:]
    new_head[-2] = 'Time0'
    new_head[-1] = 'Time1'

    save_meta_path = './datasets/meta_info/meta_info_SPAQDataset.csv'
    with open(save_meta_path, 'w+') as sf:
        csvwriter = csv.writer(sf)
        csvwriter.writerow(new_head)
        for ridx in range(mos_label.shape[0]):
            mos_row = mos_label.loc[ridx].tolist()
            scene_row = scene_label.loc[ridx].tolist()
            exif_row = exif_label.loc[ridx].tolist()
            print(mos_row, scene_row, exif_row)
            assert mos_row[0] == scene_row[0] == exif_row[0]
            row_label = mos_row + scene_row[1:] + exif_row[1:]
            csvwriter.writerow(row_label)


def get_random_splits(seed=123):
    random.seed(seed)
    total_num = 11125
    all_img_index = list(range(total_num))
    num_splits = 10
    save_path = f'./datasets/meta_info/spaq_seed{seed}.pkl'

    ratio = [0.8, 0.2]  # train/val/test
    sep_index = int(round(0.8 * total_num))

    split_info = {}
    for i in range(num_splits):
        random.shuffle(all_img_index)
        split_info[i + 1] = {'train': all_img_index[:sep_index], 'val': all_img_index[sep_index:]}
    with open(save_path, 'wb') as sf:
        pickle.dump(split_info, sf)


if __name__ == '__main__':
    # get_meta_info()
    get_random_splits()
