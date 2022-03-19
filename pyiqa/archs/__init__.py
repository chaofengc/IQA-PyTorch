import importlib
from copy import deepcopy
from os import path as osp
from collections import OrderedDict

from pyiqa.utils import get_root_logger, scandir
from pyiqa.utils.registry import ARCH_REGISTRY

from pyiqa.default_model_configs import DEFAULT_CONFIGS

__all__ = ['build_network', 'create_metric']

# automatically scan and import arch modules for registry
# scan all the files under the 'archs' folder and collect files ending with
# '_arch.py'
arch_folder = osp.dirname(osp.abspath(__file__))
arch_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(arch_folder) if v.endswith('_arch.py')]
# import all the arch modules
_arch_modules = [importlib.import_module(f'pyiqa.archs.{file_name}') for file_name in arch_filenames]


def create_metric(metric_name, eval=True, **opt):
    net_opts = OrderedDict()
    if metric_name in DEFAULT_CONFIGS.keys():
        # load default setting first
        default_opt = DEFAULT_CONFIGS[metric_name]['metric_opts']
        net_opts.update(default_opt)
    # then update with custom setting
    net_opts.update(opt)
    network_type = net_opts.pop('type')
    net = ARCH_REGISTRY.get(network_type)(**net_opts)
    net.lower_better = DEFAULT_CONFIGS[metric_name].get('lower_better', False)
    if eval:
        net.eval()
    logger = get_root_logger()
    logger.info(f'Metric [{net.__class__.__name__}] is created.')
    return net


def build_network(opt):
    opt = deepcopy(opt)
    network_type = opt.pop('type')
    net = ARCH_REGISTRY.get(network_type)(**opt)
    logger = get_root_logger()
    logger.info(f'Network [{net.__class__.__name__}] is created.')
    return net
