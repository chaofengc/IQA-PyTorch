import datetime
import logging
import time
import torch
import os
import numpy as np
from os import path as osp

from pyiqa.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from pyiqa.models import build_model
from pyiqa.utils import (AvgTimer, MessageLogger, get_root_logger, get_time_str, get_env_info, make_exp_dirs,
                         mkdir_and_rename)
from pyiqa.utils.options import copy_opt_file, dict2str, parse_options, make_paths
from pyiqa.train import init_tb_loggers, create_train_val_dataloader


def train_single_split_pipeline(root_path, opt, args):

    opt['root_path'] = root_path

    make_exp_dirs(opt)
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name'] and opt['rank'] == 0:
        os.makedirs(osp.join(opt['root_path'], 'tb_logger_archived'), exist_ok=True)
        mkdir_and_rename(osp.join(opt['root_path'], 'tb_logger', opt['name']))

    # copy the yml file to the experiment root
    copy_opt_file(args.opt, opt['path']['experiments_root'])

    # WARNING: should not use get_root_logger in the above codes, including the called functions
    # Otherwise the logger will not be properly initialized
    log_file = osp.join(opt['path']['log'], f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='pyiqa', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))
    # initialize wandb and tb loggers
    tb_logger = init_tb_loggers(opt)

    # create train and validation dataloaders
    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loaders, total_epochs, total_iters = result

    # create model
    model = build_model(opt)
    start_epoch = 0
    current_iter = 0

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # dataloader prefetcher
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f'Use {prefetch_mode} prefetch dataloader')
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
    else:
        raise ValueError(f'Wrong prefetch_mode {prefetch_mode}.' "Supported ones are: None, 'cuda', 'cpu'.")

    # training
    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_timer, iter_timer = AvgTimer(), AvgTimer()
    start_time = time.time()

    for epoch in range(start_epoch, total_epochs + 1):
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()

        while train_data is not None:
            data_timer.record()

            current_iter += 1
            if current_iter > total_iters:
                break
            # update learning rate
            model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
            # training
            model.feed_data(train_data)
            model.optimize_parameters(current_iter)
            iter_timer.record()
            if current_iter == 1:
                # reset start time in msg_logger for more accurate eta_time
                # not work in resume mode
                msg_logger.reset_start_time()
            # log
            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update({'time': iter_timer.get_avg_time(), 'data_time': data_timer.get_avg_time()})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)

            # save models and training states
            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(epoch, current_iter)

            if current_iter % opt['logger']['save_latest_freq'] == 0:
                logger.info('Saving latest models and training states.')
                model.save(epoch, current_iter, 'latest')

            # validation
            if opt.get('val') is not None and (current_iter % opt['val']['val_freq'] == 0):
                if len(val_loaders) > 1:
                    logger.warning('Multiple validation datasets are *only* supported by SRModel.')
                for val_loader in val_loaders:
                    model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])

            data_timer.start()
            iter_timer.start()
            train_data = prefetcher.next()
        # end of iter

    # end of epoch
    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest
    if tb_logger:
        tb_logger.close()

    return model.best_metric_results


def train_nsplits(root_path):
    torch.backends.cudnn.benchmark = True
    opt, args = parse_options(root_path, is_train=True)
    n_splits = opt['split_num']
    save_path = opt['save_final_results_path']
    all_split_results = []
    prefix_name = opt['name']
    for i in range(n_splits):
        # update split specific options
        opt['name'] = prefix_name + f'_Split{i:02d}'
        make_paths(opt, root_path)
        opt['datasets']['train']['split_index'] = i + 1
        opt['datasets']['val']['split_index'] = i + 1
        if 'test' in opt['datasets']:
            opt['datasets']['test']['split_index'] = i + 1
        tmp_results = train_single_split_pipeline(root_path, opt, args)
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
            avg_results = np.array(all_results).mean(axis=0)
            sf.write(f'Average results in {ds}: {avg_results}\n')


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_nsplits(root_path)
