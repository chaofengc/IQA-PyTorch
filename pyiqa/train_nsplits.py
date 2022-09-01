import datetime
import logging
import time
import torch
import os
import numpy as np
from os import path as osp

from pyiqa.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from pyiqa.models import build_model
from pyiqa.utils import (AvgTimer, MessageLogger, get_root_logger, get_time_str, get_env_info, make_exp_dirs, mkdir_and_rename)
from pyiqa.utils.options import copy_opt_file, dict2str, parse_options, make_paths
from pyiqa.train import init_tb_loggers, create_train_val_dataloader
from pyiqa.train import train_pipeline


def train_nsplits(root_path):
    torch.backends.cudnn.benchmark = True
    opt, args = parse_options(root_path, is_train=True)
    n_splits = opt['split_num']
    save_path = opt['save_final_results_path']
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    all_split_results = []
    prefix_name = opt['name']
    for i in range(n_splits):
        # update split specific options
        opt['name'] = prefix_name + f'_Split{i:02d}'
        make_paths(opt, root_path)
        for k in opt['datasets'].keys():
            opt['datasets'][k]['split_index'] = i + 1
        tmp_results = train_pipeline(root_path, opt, args)  
        all_split_results.append(tmp_results)
    
    with open(save_path, 'w') as sf:
        datasets = list(all_split_results[0].keys())
        metrics = list(all_split_results[0][datasets[0]].keys())
        print(datasets, metrics)
        sf.write('Val Datasets\tSplits\t{}\n'.format('\t'.join(metrics)))
        for ds in datasets:
            all_results = []
            for i in range(n_splits):
                results_msg = f'{ds}\t{i:02d}\t'
                tmp_metric_results = []
                for mt in metrics:
                    tmp_metric_results.append(all_split_results[i][ds][mt]['val'])
                    results_msg += f"{all_split_results[i][ds][mt]['val']:04f}\t"
                results_msg += f"@{all_split_results[i][ds][mt]['iter']:05d}\n"
                sf.write(results_msg)
                all_results.append(tmp_metric_results)
            results_avg = np.array(all_results).mean(axis=0)
            results_std =  np.array(all_results).std(axis=0)
            sf.write(f'Overall results in {ds}: {results_avg}\t{results_std}\n')
                    

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_nsplits(root_path)
