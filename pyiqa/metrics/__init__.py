from copy import deepcopy

from pyiqa.utils.registry import METRIC_REGISTRY
from .correlation_coefficient import calculate_srcc, calculate_plcc, calculate_krcc

__all__ = [
    'calculate_srcc',
    'calculate_plcc',
    'calculate_krcc',
]


def calculate_metric(data, opt):
    """Calculate metric from data and options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    metric_type = opt.pop('type')
    metric = METRIC_REGISTRY.get(metric_type)(*data, **opt)
    return metric
