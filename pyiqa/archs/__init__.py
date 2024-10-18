import importlib
from copy import deepcopy
from os import path as osp

from pyiqa.utils import get_root_logger, scandir
from pyiqa.utils.registry import ARCH_REGISTRY


__all__ = ['build_network']

# automatically scan and collect arch module filenames for lazy import
arch_folder = osp.dirname(osp.abspath(__file__))
arch_filenames = sorted([osp.splitext(osp.basename(v))[0] for v in scandir(arch_folder) if v.endswith('_arch.py')])

def lazy_import_module(module_name):
    while arch_filenames:
        if module_name in ARCH_REGISTRY:
            break
        importlib.import_module(f'pyiqa.archs.{arch_filenames.pop()}')

def build_network(opt):
    opt = deepcopy(opt)
    network_type = opt.pop('type')
    
    # Lazy import the required module
    if network_type not in ARCH_REGISTRY:
        lazy_import_module(network_type)
    
    net = ARCH_REGISTRY.get(network_type)(**opt)
    logger = get_root_logger()
    logger.info(f'Network [{net.__class__.__name__}] is created.')
    return net