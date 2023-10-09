import os
import random
import numpy
import pickle
import csv
import pandas as pd


def get_meta_info(root_dir, save_meta_path):
    attrs = ['Details', 'Exposure', 'Overall']
    
    rows_all = []
    for att in attrs:
        tmp_row = []
        # read labels
        lpath = f'{root_dir}/Scores_{att}.csv'
        lreader = csv.reader(open(lpath, 'r'))
        header = next(lreader)

        header_all = header + ['DeviceSplit', 'SceneSplit']

        # read train/test kksplits
        device_split = {}
        reader = csv.reader(open(f'{root_dir}/Device Split.csv'))
        next(reader)
        for item in reader:
            device_split[item[0]] = item[1] 
        
        scene_split = {}
        reader = csv.reader(open(f'{root_dir}/Scene Split.csv'))
        next(reader)
        for item in reader:
            scene_split[item[0]] = item[1] 
        
        for item in lreader:
            tmp_row = item
            img_name = tmp_row[0].split("\\")[1]

            if img_name in device_split:
                ds = device_split[img_name]
        
            for k, v in scene_split.items():
                if k in img_name:
                    ss = v
        
            tmp_row += [ds, ss]
            tmp_row[0] = tmp_row[0].replace('\\', '/')
            rows_all.append(tmp_row)
        
    with open(save_meta_path, 'w') as file:
        csv_writer = csv.writer(file)

        csv_writer.writerow(header_all)
        csv_writer.writerows(rows_all)


if __name__ == '__main__':
    get_meta_info('../datasets/PIQ', '../datasets/meta_info/meta_info_PIQDataset.csv')