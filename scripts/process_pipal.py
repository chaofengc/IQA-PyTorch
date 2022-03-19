import os
import scipy.io as sio
import random
import numpy
import pickle
import csv
import pandas as pd
from tqdm import tqdm
from glob import glob


def get_meta_info():

    train_label_folder = '../datasets/PIPAL/Train_Label/'

    name_labels = []
    for f in sorted(glob(train_label_folder + '*.txt')):
        name_labels += [x.strip().split(',') for x in open(f).readlines()]

    #  new_head = ['dist_name', 'elo_score', 'hq_name']
    new_head = ['hq_name', 'dist_name', 'elo_score']

    save_meta_path = './datasets/meta_info/meta_info_PIPALDataset.csv'
    with open(save_meta_path, 'w') as sf:
        csvwriter = csv.writer(sf)
        csvwriter.writerow(new_head)
        for n, l in name_labels:
            dist_name = n
            elo_score = l
            hq_name = dist_name.split('_')[0] + '.bmp'
            csvwriter.writerow([hq_name, dist_name, elo_score])


#  def get_random_splits(seed=123):
#  random.seed(seed)

#  meta_path = './datasets/meta_info/meta_info_PIPALDataset.csv'
#  meta_info = pd.read_csv(meta_path)

#  hq_names = set(meta_info['hq_name'].tolist())
#  random.shuffle(hq_names)

#  train_names = hq_names[:175]
#  test_names = hq_names[175:]

#  split_info = {
#  '1': {'train': [], 'val': [], 'test': []},
#  }
#  for index in range(meta_info.shape[0]):

if __name__ == '__main__':
    get_meta_info()
    #  get_random_splits()
