import os
import scipy.io as sio
import random
import numpy
import pickle
import csv
from tqdm import tqdm

def get_meta_info():
    info_file = '../../datasets/kadid10k/dmos.csv'

    save_meta_path = '../pyiqa/data/meta_info/meta_info_KADID10kDataset.txt'
    with open(info_file, 'r') as f, open(save_meta_path, 'w+') as sf:
        csvreader = csv.reader(f)
        head = next(csvreader)
        print(head)
        for idx, row in enumerate(csvreader):
            print(row)
            dis_name = row[0] 
            ref_name = row[1]
            dmos = row[2]
            std = row[3]
            msg = f'{ref_name:<15}\t{dis_name:<15}\t{dmos:<15}\t{std:<15}\n'
            sf.write(msg)

if __name__ == '__main__':
    get_meta_info()
    #  get_random_splits()
