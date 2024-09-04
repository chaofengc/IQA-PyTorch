import pyiqa
from pyiqa.utils import scandir_images
from time import time
import pandas as pd
import numpy as np
from tqdm import tqdm


def time_benchmark(device):
    save_file = './tests/time_benchmark.csv'
    time_sum = pd.read_csv(save_file, index_col='Method') 

    file_list = scandir_images('./tests/test_img_dir_1080p') 
    exclude_filters = ['fid', 'inception_score', 'clipscore', 'topiq_nr-face']
    if device == 'cpu':
        exclude_filters += ['qalign', 'qalign_4bit', 'qalign_8bit']
    metric_list = pyiqa.list_models(exclude_filters=exclude_filters)
    
    pbar = tqdm(total=len(metric_list))
    for idx, metric_name in enumerate(metric_list):

        metric_func = pyiqa.create_metric(metric_name, device=device)

        process_time = []
        for img_path in file_list:

            pbar.set_description(f'Testing {metric_name} on {device}')
            start_time = time()
            metric_func(img_path, img_path)
            process_time.append(time() - start_time)
        
        pbar.update(1)

        avg_process_time = np.mean(process_time[1:])
        time_sum.loc[metric_name, device] = float(f'{avg_process_time:.4f}')

    pbar.close()
    time_sum = time_sum.sort_values(by=['cuda'], ascending=True)
    time_sum.to_csv(save_file)

if __name__ == '__main__':
    # time_benchmark('cuda')
    time_benchmark('cpu')
