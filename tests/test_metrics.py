import os
import csv
import numpy as np
import torch
import pyiqa
import argparse
from pyiqa.utils.img_util import imread2tensor
from pyiqa.default_model_configs import DEFAULT_CONFIGS


def load_test_img_batch():
    img_dir = './ResultsCalibra/dist_dir'
    ref_dir = './ResultsCalibra/ref_dir'

    img_list = [x for x in sorted(os.listdir(img_dir))]
    ref_list = [x for x in sorted(os.listdir(ref_dir))]

    img_batch = []
    ref_batch = []
    for img_name, ref_name in zip(img_list, ref_list):
        img_path = os.path.join(img_dir, img_name)
        ref_path = os.path.join(ref_dir, ref_name)

        img_tensor = imread2tensor(img_path).unsqueeze(0)
        ref_tensor = imread2tensor(ref_path).unsqueeze(0)
        img_batch.append(img_tensor)
        ref_batch.append(ref_tensor)

    img_batch = torch.cat(img_batch, dim=0)
    ref_batch = torch.cat(ref_batch, dim=0)
    return img_batch, ref_batch


def load_org_results():
    results_path = './ResultsCalibra/results_original.csv'
    results = {}
    with open(results_path) as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            if row[0].startswith('vsi'):
                row[0] = 'vsi'
            results[row[0]] = row[1:]
    return results


def run_test(test_metric_names, use_cpu):
    device = torch.device('cuda' if torch.cuda.is_available() and not use_cpu else 'cpu')
    print(f'============> Testing on {device}')

    img_batch, ref_batch = load_test_img_batch()
    org_results = load_org_results()

    failed_metrics = []
    for metric_name in test_metric_names:
        print(f'============> Testing {metric_name} ... ')
        iqa_metric = pyiqa.create_metric(metric_name, as_loss=True, device=device)
        img_batch = img_batch.to(device)
        ref_batch = ref_batch.to(device)
        img_batch.requires_grad_()

        metric_mode = DEFAULT_CONFIGS[metric_name]['metric_mode']
        if metric_mode == 'FR':
            score = iqa_metric(img_batch, ref_batch)
        else:
            score = iqa_metric(img_batch)
        # Results check
        if metric_name in org_results.keys():
            org_score = np.array([float(x) for x in org_results[metric_name]])
            our_score = score.squeeze().data.cpu().numpy()
            diff = np.abs(np.abs(org_score) - np.abs(our_score))
            diff = diff * (diff > 0.01)  # remove small difference
            diff = (diff / (np.abs(org_score) + 1e-8)).mean()  # calculate relative error
            # assert diff < 0.01, f'Results average difference {diff*100:.2f}% is too big !!!'
            if diff > 0.01:
                failed_metrics.append(f'Metric {metric_name}, diff {diff}')
            print(f'============> Results average difference is {diff*100:.2f}%')
        else:
            print(f'============> No official results for {metric_name}')

        # Backward check
        if metric_name not in ['nrqm', 'pi']:
            score.mean().backward()

            grad_map = img_batch.grad
            nan_num = torch.isnan(grad_map).sum()
            if nan_num == 0:
                print(f'============> Gradient of {metric_name} is normal !')
            else:
                failed_metrics.append(f'Metric {metric_name}, gradient wrong with {nan_num}')
                print(f'============> Wrong gradient of {metric_name} with {nan_num} numbers !')

    for fm in failed_metrics:
        print(fm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--metric_names', type=str, nargs='+', default=None, help='metric name list.')
    parser.add_argument('--use_cpu', action='store_true', help='use cpu for test')
    args = parser.parse_args()

    if args.metric_names is not None:
        test_metric_names = args.metric_names
    else:
        test_metric_names = pyiqa.list_models()
        test_metric_names.remove('fid')  # do not test fid here
    run_test(test_metric_names, args.use_cpu)
