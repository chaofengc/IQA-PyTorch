import pyiqa
from pyiqa.utils import scandir_images
from time import time
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import gc
import argparse


def time_benchmark(device, col_name=None):
    save_file = './tests/Efficiency_benchmark.csv'
    time_sum = pd.read_csv(save_file, index_col='Method') 

    file_list = scandir_images('./tests/test_efficiency_img_dir') 
    exclude_filters = ['fid', 'inception_score', 'clipscore', 'topiq_nr-face']
    if device == 'cpu':
        exclude_filters += ['qalign', 'qalign_4bit', 'qalign_8bit']
    metric_list = pyiqa.list_models(exclude_filters=exclude_filters)

    pbar = tqdm(total=len(metric_list))
    for idx, metric_name in enumerate(metric_list):

        if device != 'cpu':
            torch.cuda.reset_peak_memory_stats()
        metric_func = pyiqa.create_metric(metric_name, device=device)

        process_time = []
        for img_path in file_list:

            pbar.set_description(f'Testing {metric_name} on {device}')
            torch.cuda.synchronize()
            start_time = time()
            metric_func(img_path, img_path)
            torch.cuda.synchronize()
            process_time.append(time() - start_time)
        
        # Get the peak memory allocated
        if device != 'cpu':
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # Convert to GB 
        
        pbar.update(1)
        del metric_func
        torch.cuda.empty_cache()
        gc.collect()

        avg_process_time = np.mean(process_time[1:])
        if col_name is not None:
            time_sum.loc[metric_name, col_name] = float(f'{avg_process_time:.4f}')
        else:
            time_sum.loc[metric_name, device] = float(f'{avg_process_time:.4f}')
        if device != 'cpu':
            time_sum.loc[metric_name, 'Peak GPU Mem (GB)'] = float(f'{peak_memory:.4f}')

    pbar.close()
    time_sum = time_sum.sort_values(by=['cuda_v100'], ascending=True)
    time_sum.to_csv(save_file)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', type=str, default='cuda', help='test device.')
    parser.add_argument('-c', '--col_name', type=str, default=None, help='test device.')

    args = parser.parse_args()

    time_benchmark(args.device, args.col_name)
