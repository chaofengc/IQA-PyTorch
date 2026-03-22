import importlib
from copy import deepcopy
import re
from pathlib import Path

from pyiqa.utils import get_root_logger
from pyiqa.utils.registry import MODEL_REGISTRY

__all__ = ['build_model']

_MODEL_PACKAGE = __package__
_MODEL_FOLDER = Path(__file__).resolve().parent
_ALL_MODEL_IMPORTED = False


def _camel_to_snake(name: str) -> str:
    s1 = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def _lazy_import_model(model_type: str) -> None:
    global _ALL_MODEL_IMPORTED

    stem = model_type[:-6] if model_type.endswith('_model') else model_type
    candidates = (
        f'{_MODEL_PACKAGE}.{stem}_model',
        f'{_MODEL_PACKAGE}.{stem.lower()}_model',
        f'{_MODEL_PACKAGE}.{_camel_to_snake(stem)}_model',
    )

    for module_name in dict.fromkeys(candidates):
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as error:
            if error.name != module_name:
                raise
        if model_type in MODEL_REGISTRY:
            return

    if _ALL_MODEL_IMPORTED:
        return

    # Compatibility fallback: import all modules once.
    for file_path in _MODEL_FOLDER.glob('*_model.py'):
        module_name = f'{_MODEL_PACKAGE}.{file_path.stem}'
        try:
            importlib.import_module(module_name)
        except Exception:
            continue
    _ALL_MODEL_IMPORTED = True


def build_model(opt):
    """Build model from options.

    Args:
        opt (dict): Configuration. It must contain:
            model_type (str): Model type.
    """
    opt = deepcopy(opt)
    if opt['model_type'] not in MODEL_REGISTRY:
        _lazy_import_model(opt['model_type'])
    model = MODEL_REGISTRY.get(opt['model_type'])(opt)
    logger = get_root_logger()
    logger.info(f'Model [{model.__class__.__name__}] is created.')
    return model
