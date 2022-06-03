import argparse
import yaml
import csv
from pyiqa.data import build_dataset, build_dataloader
from pyiqa.default_model_configs import DEFAULT_CONFIGS
from pyiqa.utils.options import ordered_yaml
from pyiqa.metrics import calculate_plcc, calculate_srcc, calculate_krcc
from tqdm import tqdm
import torch
from pyiqa import create_metric


def main():
    """benchmark test demo for pyiqa.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=str, nargs='+', default=None, help='metric name list.')
    parser.add_argument('-d', type=str, nargs='+', default=None, help='dataset name list.')
    parser.add_argument('--metric_opt', type=str, default=None, help='Path to custom metric option YAML file.')
    parser.add_argument('--data_opt', type=str, default=None, help='Path to custom data option YAML file.')
    parser.add_argument('--save_result_path', type=str, default=None, help='file to save results.')
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
    with open('./options/default_dataset_opt.yml', mode='r') as f:
        all_data_opts = yaml.load(f, Loader=ordered_yaml()[0])

    # load custom options to test
    if args.metric_opt is not None:
        with open(args.metric_opt, mode='r') as f:
            custom_metric_opt = yaml.load(f, Loader=ordered_yaml()[0])
        all_metric_opts.update(custom_metric_opt)
        metrics_to_test += list(custom_metric_opt.keys())

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

    for metric_name in metrics_to_test:
        # if metric_name exist in default config, load default config first
        metric_opts = all_metric_opts[metric_name]['metric_opts']
        metric_mode = all_metric_opts[metric_name]['metric_mode']
        lower_better = all_metric_opts[metric_name].get('lower_better', False)
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
            if 'phase' not in data_opts:
                data_opts['phase'] = 'test'
            dataset = build_dataset(data_opts)
            dataloader = build_dataloader(dataset, data_opts, num_gpu=num_gpu)
            gt_labels = []
            result_scores = []
            pbar = tqdm(total=len(dataloader), unit='image')
            pbar.set_description(f'Testing *{metric_name}* on ({dataset_name})')
            for data in dataloader:
                gt_labels.append(data['mos_label'].cpu().item())
                if metric_mode == 'FR':
                    iqa_score = iqa_model(data['img'], data['ref_img']).cpu().item()
                else:
                    iqa_score = iqa_model(data['img']).cpu().item()
                result_scores.append(iqa_score)
                pbar.update(1)
            pbar.close()

            if lower_better:
                results_scores_for_cc = [-x for x in result_scores]
            else:
                results_scores_for_cc = result_scores

            plcc_score = round(calculate_plcc(results_scores_for_cc, gt_labels), 4)
            srcc_score = round(calculate_srcc(results_scores_for_cc, gt_labels), 4)
            krcc_score = round(calculate_krcc(results_scores_for_cc, gt_labels), 4)
            results_row.append(f'{plcc_score}/{srcc_score}/{krcc_score}')
            print(
                f'Results of *{metric_name}* on ({dataset_name}) is [PLCC|SRCC|KRCC]: {plcc_score}, {srcc_score}, {krcc_score}'
            )
        if save_result_path is not None:
            csv_writer.writerow(results_row)
    if save_result_path is not None:
        csv_file.close()


if __name__ == '__main__':
    main()
