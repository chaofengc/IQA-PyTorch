import argparse
import yaml
import csv
import pandas as pd
from itertools import chain

from pyiqa.data import build_dataset, build_dataloader
from pyiqa.default_model_configs import DEFAULT_CONFIGS
from pyiqa.utils.options import ordered_yaml
from pyiqa.metrics import calculate_plcc, calculate_srcc, calculate_krcc

from tqdm import tqdm
import torch
from pyiqa import create_metric


def flatten_list(list_of_list):
    if isinstance(list_of_list, list):
        if isinstance(list_of_list[0], list):
            return list(chain.from_iterable(list_of_list))
        else:
            return list_of_list
    else:
        return [list_of_list]

def str_to_bool(s: str) -> bool:
    true_values = {"true", "1", "yes", "y", "t", "on"}
    false_values = {"false", "0", "no", "n", "f", "off"}
    
    # Convert the string to lowercase and strip any leading/trailing whitespace
    s = s.strip().lower()
    
    if s in true_values:
        return True
    elif s in false_values:
        return False
    else:
        return s

def main():
    """benchmark test demo for pyiqa.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=str, nargs='+', default=None, help='metric name list.')
    parser.add_argument('-d', type=str, nargs='+', default=None, help='dataset name list.')
    parser.add_argument('--metric_opt', type=str, default=None, help='Path to custom metric option YAML file.')
    parser.add_argument('--extra_metric_opts', nargs='+', type=str, default=None, help='Extra options for all tested metrics.')
    parser.add_argument('--data_opt', type=str, default=None, help='Path to custom data option YAML file.')
    parser.add_argument('--batch_size', type=int, default=None, help='batch size for benchmark.')
    parser.add_argument('--split_file', type=str, default=None, help='split file for test.')
    parser.add_argument('--test_phase', type=str, default=None, help='phase for benchmark: val/test.')
    parser.add_argument('--save_result_path', type=str, default=None, help='file to save results.')
    parser.add_argument('--update_benchmark', type=str, default=None, help='update benchmark results.')
    parser.add_argument('--use_gpu', action='store_true', default=False, help='use gpu or not')
    args = parser.parse_args()

    metrics_to_test = []
    datasets_to_test = []
    if args.m is not None:
        metrics_to_test += args.m
    if args.d is not None:
        datasets_to_test += args.d

    if args.use_gpu:
        num_gpu = 1
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        num_gpu = 0
        device = torch.device('cpu')

    # ========== get metric and dataset options ===========
    # load default options first
    all_metric_opts = DEFAULT_CONFIGS
    with open('./pyiqa/default_dataset_configs.yml', mode='r') as f:
        all_data_opts = yaml.load(f, Loader=ordered_yaml()[0])

    # load custom options to test
    if args.metric_opt is not None:
        with open(args.metric_opt, mode='r') as f:
            custom_metric_opt = yaml.load(f, Loader=ordered_yaml()[0])
        all_metric_opts.update(custom_metric_opt)
        metrics_to_test += list(custom_metric_opt.keys())
    
    extra_opt_dict = {}
    if args.extra_metric_opts is not None:
        for extra_opt in args.extra_metric_opts:
            extra_opt = extra_opt.split('=')
            if len(extra_opt) == 2:
                extra_opt_dict[extra_opt[0]] = str_to_bool(extra_opt[1])

    if args.data_opt is not None:
        with open(args.data_opt, mode='r') as f:
            custom_data_opt = yaml.load(f, Loader=ordered_yaml()[0])
        all_data_opts.update(custom_data_opt)
        datasets_to_test += list(custom_data_opt.keys())

    # =====================================================

    save_result_path = args.save_result_path

    if save_result_path is not None:
        csv_file = open(save_result_path, 'w')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Metric name'] + [name + '(PLCC/SRCC/KRCC)' for name in datasets_to_test])
    
    update_benchmark_file = args.update_benchmark
    if update_benchmark_file is not None:
        benchmark = pd.read_csv(update_benchmark_file, index_col='Metric name') 

    for metric_name in metrics_to_test:
        # if metric_name exist in default config, load default config first
        metric_opts = all_metric_opts[metric_name]['metric_opts']
        metric_mode = all_metric_opts[metric_name]['metric_mode']
        lower_better = all_metric_opts[metric_name].get('lower_better', False)
        metric_opts.update(extra_opt_dict)
        if metric_name == 'pieapp':
            lower_better = False    # ground truth score is also lower better for pieapp test set
        iqa_model = create_metric(metric_name, device=device, metric_mode=metric_mode, **metric_opts)

        results_row = [metric_name]
        for dataset_name in datasets_to_test:
            data_opts = all_data_opts[dataset_name]
            data_opts.update({
                'num_worker_per_gpu': 8,
                'prefetch_mode': 'cpu',
                'num_prefetch_queue': 8,
            })
            if args.batch_size is not None:
                data_opts.update({
                    'batch_size_per_gpu': args.batch_size,
                })
            if args.split_file is not None:
                data_opts.update({
                    'split_file': args.split_file,
                })
            if args.split_file is not None and args.test_phase is not None:
                data_opts.update({
                    'phase': args.test_phase,
                })

            if 'phase' not in data_opts:
                data_opts['phase'] = 'test'
            dataset = build_dataset(data_opts)
            dataloader = build_dataloader(dataset, data_opts, num_gpu=num_gpu)
            gt_labels = []
            result_scores = []
            pbar = tqdm(total=len(dataloader), unit='image')
            pbar.set_description(f'Testing *{metric_name}* on ({dataset_name})')
            for data in dataloader:
                try:
                    if metric_mode == 'FR':
                        iqa_score = iqa_model(data['img'], data['ref_img'])
                    else:
                        iqa_score = iqa_model(data['img'])

                    if not torch.isnan(iqa_score).any():
                        iqa_score = iqa_score.squeeze().cpu().tolist()
                        gt_labels += flatten_list(data['mos_label'].cpu().tolist())
                        result_scores += flatten_list(iqa_score)
                except:
                    print(f'Error in testing {metric_name} on {dataset_name}: {data["img_path"]}')
                pbar.update(1)
            pbar.close()

            if lower_better:
                results_scores_for_cc = [-x for x in result_scores]
            else:
                results_scores_for_cc = result_scores

            plcc_score = abs(round(calculate_plcc(results_scores_for_cc, gt_labels), 4))
            srcc_score = abs(round(calculate_srcc(results_scores_for_cc, gt_labels), 4))
            krcc_score = abs(round(calculate_krcc(results_scores_for_cc, gt_labels), 4))
            results_row.append(f'{plcc_score}/{srcc_score}/{krcc_score}')
            print(
                f'Results of *{metric_name}* on ({dataset_name}) is [PLCC|SRCC|KRCC]: {plcc_score}, {srcc_score}, {krcc_score}'
            )
            if update_benchmark_file is not None:
                benchmark.loc[metric_name, f'{dataset_name}(PLCC/SRCC/KRCC)'] = f'{plcc_score}/{srcc_score}/{krcc_score}'

        if save_result_path is not None:
            csv_writer.writerow(results_row)
        
    if save_result_path is not None:
        csv_file.close()

    if update_benchmark_file is not None:
        benchmark = benchmark.sort_values(by=benchmark.columns[0], key=lambda x: x.str.split('/').str[0].astype(float))
        benchmark.to_csv(update_benchmark_file)

if __name__ == '__main__':
    main()
