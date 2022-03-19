import os
import scipy.io as sio
import random
import numpy as np
import pickle
import csv
import pandas as pd
from tqdm import tqdm
from glob import glob

IMG_EXTENSIONS = [
    '.jpg',
    '.JPG',
    '.jpeg',
    '.JPEG',
    '.png',
    '.PNG',
    '.ppm',
    '.PPM',
    '.bmp',
    '.BMP',
    '.tif',
    '.TIF',
    '.tiff',
    '.TIFF',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float('inf')):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]


def get_meta_info():

    # 2afc triplets
    root_dir = '../datasets/PerceptualSimilarity/dataset/2afc'

    ref_dir = '../datasets/PerceptualSimilarity/dataset/2afc/*/*/ref/*.png'
    p0_dir = '../datasets/PerceptualSimilarity/dataset/2afc/*/*/p0/*.png'
    p1_dir = '../datasets/PerceptualSimilarity/dataset/2afc/*/*/p1/*.png'
    judge_dir = '../datasets/PerceptualSimilarity/dataset/2afc/*/*/judge/*.npy'
    ref_path_list = sorted(list(glob(ref_dir)))
    p0_path_list = sorted(list(glob(p0_dir)))
    p1_path_list = sorted(list(glob(p1_dir)))
    judge_path_list = sorted(list(glob(judge_dir)))

    # jnd pairs
    p0_dir = '../datasets/PerceptualSimilarity/dataset/jnd/val/*/p0/*.png'
    p1_dir = '../datasets/PerceptualSimilarity/dataset/jnd/val/*/p1/*.png'
    judge_dir = '../datasets/PerceptualSimilarity/dataset/jnd/val/*/same/*.npy'
    jnd_p0_path_list = sorted(list(glob(p0_dir)))
    jnd_p1_path_list = sorted(list(glob(p1_dir)))
    jnd_judge_path_list = sorted(list(glob(judge_dir)))

    save_meta_path = './datasets/meta_info/meta_info_BAPPSDataset.csv'
    split_info = {
        1: {
            'train': [],
            'val': [],
            'test': []
        },
    }

    with open(save_meta_path, 'w') as sf:
        csvwriter = csv.writer(sf)
        head = ['ref_img_path', 'p0_img_path', 'p1_img_path', 'judge/same', 'split']
        csvwriter.writerow(head)

        count = 0
        for ref_path, p0_path, p1_path, jd_path in tqdm(
                zip(ref_path_list, p0_path_list, p1_path_list, judge_path_list), total=len(ref_path_list)):
            ref_path = ref_path.split('dataset/')[-1]
            p0_path = p0_path.split('dataset/')[-1]
            p1_path = p1_path.split('dataset/')[-1]

            jd_label = np.load(jd_path)[0]

            if 'train' in ref_path:
                split = 0
                split_info[1]['train'].append(count)
            elif 'val' in ref_path:
                split = 1
                split_info[1]['val'].append(count)

            row = [ref_path, p0_path, p1_path, jd_label, split]
            csvwriter.writerow(row)
            count += 1

        for p0_path, p1_path, jd_path in tqdm(
                zip(jnd_p0_path_list, jnd_p1_path_list, jnd_judge_path_list), total=len(p0_path)):
            p0_path = p0_path.split('dataset/')[-1]
            p1_path = p1_path.split('dataset/')[-1]

            jd_label = float(np.load(jd_path))

            if 'train' in ref_path:
                split = 0
                split_info[1]['train'].append(count)
            elif 'val' in ref_path:
                split = 1
                split_info[1]['val'].append(count)

            row = ['jnd', p0_path, p1_path, jd_label, split]
            csvwriter.writerow(row)
            count += 1

    save_split_path = './datasets/meta_info/bapps_official.pkl'
    with open(save_split_path, 'wb') as sf:
        pickle.dump(split_info, sf)


if __name__ == '__main__':
    get_meta_info()
