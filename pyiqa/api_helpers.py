import fnmatch
import re
from .default_model_configs import DEFAULT_CONFIGS

from .utils import get_root_logger
from .models.inference_model import InferenceModel


def create_metric(metric_name, as_loss=False, device=None, **kwargs):
    assert metric_name in DEFAULT_CONFIGS.keys(), f'Metric {metric_name} not implemented yet.' 
    metric = InferenceModel(metric_name, as_loss, device, **kwargs)
    logger = get_root_logger()
    logger.info(f'Metric [{metric.net.__class__.__name__}] is created.')
    return metric


def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def list_models(metric_mode=None, filter='', exclude_filters=''):
    """ Return list of available model names, sorted alphabetically
    Args:
        filter (str) - Wildcard filter string that works with fnmatch
        exclude_filters (str or list[str]) - Wildcard filters to exclude models after including them with filter
    Example:
        model_list('*ssim*') -- returns all models including 'ssim'
    """
    if metric_mode is None:
        all_models = DEFAULT_CONFIGS.keys()
    else:
        assert metric_mode in ['FR', 'NR'], f'Metric mode only support [FR, NR], but got {metric_mode}'
        all_models = [key for key in DEFAULT_CONFIGS.keys() if DEFAULT_CONFIGS[key]['metric_mode'] == metric_mode]

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
