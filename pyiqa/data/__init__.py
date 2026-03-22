import importlib
import numpy as np
import random
import re
import torch
import torch.utils.data
from copy import deepcopy
from functools import partial
from os import path as osp
from pathlib import Path

from pyiqa.data.prefetch_dataloader import PrefetchDataLoader
from pyiqa.utils import get_root_logger, scandir
from pyiqa.utils.dist_util import get_dist_info
from pyiqa.utils.registry import DATASET_REGISTRY

__all__ = ['build_dataset', 'build_dataloader']

_DATA_PACKAGE = __package__
_DATA_FOLDER = Path(__file__).resolve().parent
_ALL_DATASET_IMPORTED = False


def _camel_to_snake(name: str) -> str:
    s1 = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def _lazy_import_dataset(dataset_type: str) -> None:
    global _ALL_DATASET_IMPORTED

    stem = dataset_type[:-8] if dataset_type.endswith('_dataset') else dataset_type
    candidates = (
        f'{_DATA_PACKAGE}.{stem}_dataset',
        f'{_DATA_PACKAGE}.{stem.lower()}_dataset',
        f'{_DATA_PACKAGE}.{_camel_to_snake(stem)}_dataset',
    )

    for module_name in dict.fromkeys(candidates):
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as error:
            if error.name != module_name:
                raise
        if dataset_type in DATASET_REGISTRY:
            return

    if _ALL_DATASET_IMPORTED:
        return

    # Compatibility fallback: import all modules once.
    for file_path in _DATA_FOLDER.glob('*_dataset.py'):
        module_name = f'{_DATA_PACKAGE}.{file_path.stem}'
        try:
            importlib.import_module(module_name)
        except Exception:
            continue
    _ALL_DATASET_IMPORTED = True


def build_dataset(dataset_opt):
    """Build dataset from options.

    Args:
        dataset_opt (dict): Configuration for dataset. It must contain:
            name (str): Dataset name.
            type (str): Dataset type.
    """
    dataset_opt = deepcopy(dataset_opt)
    if dataset_opt['type'] not in DATASET_REGISTRY:
        _lazy_import_dataset(dataset_opt['type'])
    dataset = DATASET_REGISTRY.get(dataset_opt['type'])(dataset_opt)
    logger = get_root_logger()
    logger.info(
        f'Dataset [{dataset.__class__.__name__}] - {dataset_opt["name"]} is built.'
    )
    return dataset


def build_dataloader(
    dataset, dataset_opt, num_gpu=1, dist=False, sampler=None, seed=None
):
    """Build dataloader.

    Args:
        dataset (torch.utils.data.Dataset): Dataset.
        dataset_opt (dict): Dataset options. It contains the following keys:
            phase (str): 'train' or 'val'.
            num_worker_per_gpu (int): Number of workers for each GPU.
            batch_size_per_gpu (int): Training batch size for each GPU.
        num_gpu (int): Number of GPUs. Used only in the train phase.
            Default: 1.
        dist (bool): Whether in distributed training. Used only in the train
            phase. Default: False.
        sampler (torch.utils.data.sampler): Data sampler. Default: None.
        seed (int | None): Seed. Default: None
    """
    phase = dataset_opt['phase']
    rank, _ = get_dist_info()
    if phase == 'train':
        if dist:  # distributed training
            batch_size = dataset_opt['batch_size_per_gpu']
            num_workers = dataset_opt['num_worker_per_gpu']
        else:  # non-distributed training
            multiplier = 1 if num_gpu == 0 else num_gpu
            batch_size = dataset_opt['batch_size_per_gpu'] * multiplier
            num_workers = dataset_opt['num_worker_per_gpu'] * multiplier
        dataloader_args = dict(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            sampler=sampler,
            drop_last=True,
        )
        if sampler is None:
            dataloader_args['shuffle'] = True
        dataloader_args['worker_init_fn'] = (
            partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=seed)
            if seed is not None
            else None
        )
    elif phase in ['val', 'test']:  # validation
        batch_size = dataset_opt.get('batch_size_per_gpu', 1)
        num_workers = dataset_opt.get('num_worker_per_gpu', 0)
        dataloader_args = dict(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
    else:
        raise ValueError(
            f'Wrong dataset phase: {phase}. '
            "Supported ones are 'train', 'val' and 'test'."
        )

    dataloader_args['pin_memory'] = dataset_opt.get('pin_memory', False)
    dataloader_args['persistent_workers'] = dataset_opt.get('persistent_workers', False)

    prefetch_mode = dataset_opt.get('prefetch_mode')
    if prefetch_mode == 'cpu':  # CPUPrefetcher
        num_prefetch_queue = dataset_opt.get('num_prefetch_queue', 1)
        logger = get_root_logger()
        logger.info(
            f'Use {prefetch_mode} prefetch dataloader: num_prefetch_queue = {num_prefetch_queue}'
        )
        return PrefetchDataLoader(
            num_prefetch_queue=num_prefetch_queue, **dataloader_args
        )
    else:
        # prefetch_mode=None: Normal dataloader
        # prefetch_mode='cuda': dataloader for CUDAPrefetcher
        return torch.utils.data.DataLoader(**dataloader_args)


def worker_init_fn(worker_id, num_workers, rank, seed):
    # Set the worker seed to num_workers * rank + worker_id + seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
