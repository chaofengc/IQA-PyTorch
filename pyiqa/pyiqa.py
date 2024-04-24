import argparse
from .api_helpers import create_metric, list_models
from pyiqa.utils import scandir_images

import os
import sys
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

    metric_func_list = {}
    results = {}
    for metric in args.metric:
        metric_func = create_metric(metric)
        if metric == 'fid':
            result = metric_func(args.target, args.ref, mode=args.fid_mode, verbose=args.verbose)
            results[metric] = result
        elif metric == 'inception_score':
            result = metric_func(args.target, splits=args.isc_splits, verbose=args.verbose)
            results[metric] = result
        else:
            metric_func_list[metric] = metric_func
    
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
        if args.ref is None:
            tmp_result = []
            for target in target_list:
                tmp_result.append(func(target).item())
        else:
            tmp_result = []
            for target, ref in zip(target_list, ref_list):
                tmp_result.append(func(target, ref).item())

        results[metric] = np.mean(tmp_result) 
    
    pprint(results)

if __name__ == '__main__':
    main()