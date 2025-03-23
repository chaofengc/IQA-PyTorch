import argparse
from pyiqa.api_helpers import create_metric, list_models
from pyiqa.utils import scandir_images

import os
from tqdm import tqdm
import numpy as np
from datetime import datetime
from pprint import pprint


def main():
    parser = argparse.ArgumentParser(description='Test a metric')

    # required input arguments
    parser.add_argument('metric', nargs='*', help='Metric name(s)')
    parser.add_argument('-t', '--target', type=str, help='Target files or folder')
    parser.add_argument(
        '-r', '--ref', type=str, default=None, help='Reference files or folder'
    )

    # metric specific options
    parser.add_argument(
        '--isc_splits', type=int, default=10, help='Splits for inception score'
    )
    parser.add_argument(
        '--fid_mode',
        type=str,
        default='clean',
        help='Image resize mode for FID [clean, legacy_pytorch, legacy_tensorflow]',
    )

    # common options
    parser.add_argument(
        '--device', type=str, default=None, help='Print verbose message'
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true', help='Print verbose message'
    )
    parser.add_argument(
        '-ls', '--list_models', action='store_true', help='Print verbose message'
    )

    args, unknown_args = parser.parse_known_args()

    if args.list_models:
        pprint(list_models(), compact=True)
        return

    def get_time():
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    print(f'[{get_time()}] ===> Loading metrics: {args.metric}')
    metric_func_list = {}
    for metric in args.metric:
        metric_func = create_metric(metric, device=args.device)
        metric_func_list[metric] = metric_func
    print(f'[{get_time()}] ===> Metrics loaded: {args.metric}')

    results = {}
    # Test fid, inception_score
    if 'fid' in metric_func_list:
        metric_func = metric_func_list.pop('fid')
        result = metric_func(
            args.target, args.ref, mode=args.fid_mode, verbose=args.verbose
        )
        results['fid'] = result
    if 'inception_score' in metric_func_list:
        metric_func = metric_func_list.pop('inception_score')
        result = metric_func(args.target, splits=args.isc_splits, verbose=args.verbose)
        results['inception_score'] = result

    if os.path.isdir(args.target):
        target_list = scandir_images(args.target)
        if args.ref is not None:
            ref_list = scandir_images(args.ref)
            assert len(target_list) == len(ref_list), (
                'Number of images in target and reference folders must be the same'
            )
    else:
        target_list = [args.target]
        if args.ref is not None:
            ref_list = [args.ref]

    for metric, func in metric_func_list.items():
        pbar = tqdm(total=len(target_list), unit='image')

        all_results = {}
        if args.ref is None or func.metric_mode == 'NR':
            tmp_result = []
            for target in target_list:
                score = func(target).item()
                tmp_result.append(score)
                if args.verbose:
                    all_results[target] = score

                pbar.update(1)
                pbar.set_description(
                    f'[{get_time()}] Testing {metric} with input {target:>20}'
                )
        else:
            tmp_result = []
            for target, ref in zip(target_list, ref_list):
                score = func(target, ref).item()
                tmp_result.append(score)
                if args.verbose:
                    all_results[f'{target} | {ref}'] = score

                pbar.update(1)
                pbar.set_description(
                    f'[{get_time()}] Testing {metric} with input {target:>20}'
                )

        pbar.close()
        all_results['mean'] = np.mean(tmp_result)
        results[metric] = all_results

    pprint(results)


if __name__ == '__main__':
    main()
