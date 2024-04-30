import argparse
from .api_helpers import create_metric, list_models
from pyiqa.utils import scandir_images

import os
import sys
from tqdm import tqdm
import numpy as np
from pprint import pprint


def main():
    parser = argparse.ArgumentParser(description='Test a metric') 
    parser.add_argument('metric', nargs='*', help='Metric name(s)')
    parser.add_argument('--target', type=str, help='Target files or folder')
    parser.add_argument('--ref', type=str, default=None, help='Reference files or folder')

    # metric specific options
    parser.add_argument('--isc_splits', type=int, default=10, help='Splits for inception score')
    parser.add_argument('--fid_mode', type=str, default='clean', help='Image resize mode for FID [clean, legacy_pytorch, legacy_tensorflow]')
    parser.add_argument('--verbose', action='store_true', help='Print verbose message')

    parser.add_argument('-ls', '--list_models', action='store_true', help='Print verbose message')

    args, unknown_args = parser.parse_known_args()

    if args.list_models:
        pprint(list_models(), compact=True)
        return 

    print(f"{'='*50} Loading metrics {'='*50}")
    metric_func_list = {}
    for metric in args.metric:
        metric_func = create_metric(metric)
        metric_func_list[metric] = metric_func
    print(f"{'='*50} Metrics loaded {'='*50}")

    results = {}
    # Test fid, inception_score
    if 'fid' in metric_func_list:
        metric_func = metric_func_list.pop('fid')
        result = metric_func(args.target, args.ref, mode=args.fid_mode, verbose=args.verbose)
        results['fid'] = result
    if 'inception_score' in metric_func_list:
        metric_func = metric_func_list.pop('inception_score')
        result = metric_func(args.target, splits=args.isc_splits, verbose=args.verbose)
        results['inception_score'] = result
    
    if os.path.isdir(args.target):
        target_list = scandir_images(args.target) 
        if args.ref is not None:
            ref_list = scandir_images(args.ref)
            assert len(target_list) == len(ref_list), 'Number of images in target and reference folders must be the same'
    else:
        target_list = [args.target]
        if args.ref is not None:
            ref_list = [args.ref]    

    for metric, func in metric_func_list.items():
        pbar = tqdm(total=len(target_list), unit='image')

        if args.ref is None or func.metric_mode == 'NR':
            tmp_result = []
            for target in target_list:
                tmp_result.append(func(target).item())

                pbar.update(1)
                pbar.set_description(f'Testing {metric} with input {target:>20}')
        else:
            tmp_result = []
            for target, ref in zip(target_list, ref_list):
                tmp_result.append(func(target, ref).item())

                pbar.update(1)
                pbar.set_description(f'Testing {metric} with input {target:>20}')

        pbar.close()
        results[metric] = np.mean(tmp_result) 
    
    pprint(results)

if __name__ == '__main__':
    main()