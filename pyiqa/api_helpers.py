"""Public helpers for constructing metrics and querying dataset metadata."""

import fnmatch
import re
from pyiqa.default_model_configs import DEFAULT_CONFIGS
import yaml
import os

from pyiqa.utils import get_root_logger
from pyiqa.models.inference_model import InferenceModel


def create_metric(metric_name, as_loss=False, device=None, **kwargs):
    """Create a configured IQA metric.

    Args:
        metric_name (str): Name returned by :func:`list_models`.
        as_loss (bool): Enable gradient propagation and loss reduction where
            supported. This does not make every metric differentiable.
        device (str or torch.device, optional): Device for the metric. Defaults
            to CUDA when available, otherwise CPU.
        **kwargs: Options that override the selected metric's default
            architecture options.

    Returns:
        InferenceModel: A callable metric module exposing ``metric_mode``,
        ``lower_better``, and ``score_range`` metadata.
    """
    assert metric_name in DEFAULT_CONFIGS.keys(), (
        f'Metric {metric_name} not implemented yet.'
    )
    metric = InferenceModel(metric_name, as_loss=as_loss, device=device, **kwargs)
    logger = get_root_logger()
    logger.info(f'Metric [{metric.net.__class__.__name__}] is created.')
    return metric


def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def list_models(metric_mode=None, filter='', exclude_filters=''):
    """Return available metric names in natural alphanumeric order.

    Args:
        metric_mode (str, optional): Restrict results to ``'FR'`` or ``'NR'``.
            ``None`` includes all configured metrics.
        filter (str or sequence[str]): ``fnmatch`` wildcard(s) selecting names.
            Multiple patterns are combined.
        exclude_filters (str or sequence[str]): Wildcard(s) removed from the
            selected results after inclusion filtering.

    Returns:
        list[str]: Matching metric names, sorted naturally (for example,
        ``metric2`` appears before ``metric10``).
    """
    if metric_mode is None:
        all_models = DEFAULT_CONFIGS.keys()
    else:
        assert metric_mode in ['FR', 'NR'], (
            f'Metric mode only support [FR, NR], but got {metric_mode}'
        )
        all_models = [
            key
            for key in DEFAULT_CONFIGS.keys()
            if DEFAULT_CONFIGS[key]['metric_mode'] == metric_mode
        ]

    if filter:
        models = []
        include_filters = filter if isinstance(filter, (tuple, list)) else [filter]
        for f in include_filters:
            include_models = fnmatch.filter(all_models, f)  # include these models
            if len(include_models):
                models = set(models).union(include_models)
    else:
        models = all_models
    if exclude_filters:
        if not isinstance(exclude_filters, (tuple, list)):
            exclude_filters = [exclude_filters]
        for xf in exclude_filters:
            exclude_models = fnmatch.filter(models, xf)  # exclude these models
            if len(exclude_models):
                models = set(models).difference(exclude_models)
    return list(sorted(models, key=_natural_key))


def get_dataset_info(dataset_name=None):
    """Read default dataset loader configurations.

    Args:
        dataset_name (str, optional): Return only this dataset's configuration.
            If omitted, return the full mapping.

    Returns:
        dict: Dataset configuration or the complete dataset configuration map.
    """
    dataset_info = yaml.safe_load(
        open(
            f'{os.path.dirname(os.path.abspath(__file__))}/default_dataset_configs.yml',
            'r',
        )
    )
    if dataset_name == None:
        return dataset_info
    else:
        assert dataset_name in dataset_info.keys(), (
            f'Dataset {dataset_name} not implemented yet.'
        )
        return dataset_info[dataset_name]
