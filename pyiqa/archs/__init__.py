import importlib
import copy
import re
from pathlib import Path

from pyiqa.utils import get_root_logger
from pyiqa.utils.registry import ARCH_REGISTRY


__all__ = ['build_network']


_ARCH_PACKAGE = __package__
_ARCH_FOLDER = Path(__file__).resolve().parent
_ALL_ARCH_IMPORTED = False


def _camel_to_snake(name: str) -> str:
    s1 = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def _lazy_import_arch(network_type: str) -> None:
    global _ALL_ARCH_IMPORTED

    stem = network_type[:-5] if network_type.endswith('_arch') else network_type
    candidates = (
        f'{_ARCH_PACKAGE}.{stem}_arch',
        f'{_ARCH_PACKAGE}.{stem.lower()}_arch',
        f'{_ARCH_PACKAGE}.{_camel_to_snake(stem)}_arch',
    )

    for module_name in dict.fromkeys(candidates):
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as error:
            # Ignore only when the candidate module itself is missing.
            if error.name != module_name:
                raise
        if network_type in ARCH_REGISTRY:
            return

    if _ALL_ARCH_IMPORTED:
        return

    # Compatibility fallback: import all arch modules once for shared-module class names.
    for file_path in _ARCH_FOLDER.glob('*_arch.py'):
        module_name = f'{_ARCH_PACKAGE}.{file_path.stem}'
        try:
            importlib.import_module(module_name)
        except Exception:
            # Some optional modules may fail to import; continue loading others.
            continue

    _ALL_ARCH_IMPORTED = True


def build_network(opt):
    """
    Build a network based on the provided options.

    Args:
        opt (dict): Dictionary containing network options. Must include the 'type' key.

    Returns:
        nn.Module: The constructed network.

    Example:
        >>> net = build_network(opt)
        >>> print(net)
    """
    opt = copy.deepcopy(opt)
    network_type = opt.pop('type')

    logger = get_root_logger()
    # Deterministic lazy import without class-name cache files.
    if network_type not in ARCH_REGISTRY:
        _lazy_import_arch(network_type)

    net = ARCH_REGISTRY.get(network_type)(**opt)
    logger.info(f'Network [{net.__class__.__name__}] is created.')
    return net
