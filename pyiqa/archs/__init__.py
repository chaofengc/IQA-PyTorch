from os import path as osp
import json
from typing import Dict, Optional
from pathlib import Path
import importlib
import inspect
import copy

from pyiqa.utils import get_root_logger
from pyiqa.utils.registry import ARCH_REGISTRY


__all__ = ['build_network']


class ClassMapper:
    """
    ClassMapper is responsible for mapping class names to their corresponding file paths.
    It uses a cache file to store the mapping and refreshes it if necessary.

    Args:
        cache_file (str): JSON file to store the mapping. Default is 'class_mapping.json'.
    """

    def __init__(self, cache_file: str = 'class_mapping.json'):
        self.arch_folder = Path(osp.dirname(osp.abspath(__file__)))
        self.cache_file = self.arch_folder / cache_file
        self._mapping: Optional[Dict] = None

        self._load_cache()

    def _load_cache(self) -> Dict:
        """Load mapping from cache file."""
        if not osp.exists(self.cache_file):
            print(
                f'Warning: Cache file not found: {self.cache_file}. Refreshing cache...'
            )
            self.refresh()

        with open(self.cache_file, 'r') as f:
            self._mapping = json.load(f)

    def _save_cache(self, mapping: Dict) -> None:
        """Save mapping to cache file."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(mapping, f, indent=4)
        except Exception as e:
            print(f'Warning: Failed to save cache file: {e}')

    def _find_classes_in_file(self, file_path: Path) -> Dict[str, str]:
        """
        Find classes in a Python file that match our criteria.
        Returns a dict of {class_name: file_path}.

        Args:
            file_path (Path): Path to the Python file.

        Returns:
            Dict[str, str]: Mapping of class names to file paths.
        """
        classes = {}

        try:
            # Use importlib to load the module
            module = importlib.import_module(f'pyiqa.archs.{file_path.stem}')
            # Get all classes in the module
            classes_in_module = inspect.getmembers(module, inspect.isclass)

            for class_name, class_type in classes_in_module:
                classes[class_name] = file_path.stem

        except Exception as e:
            print(f'Warning: Failed to process {file_path}: {e}')

        return classes

    def _scan_architecture_files(self) -> Dict:
        """Scan architecture files and create mapping."""
        mapping = {}

        # Scan all architecture files
        for file_path in self.arch_folder.glob('*_arch.py'):
            file_classes = self._find_classes_in_file(file_path)
            mapping.update(file_classes)

        return mapping

    def get_mapping(self) -> Dict:
        """
        Get the class to filename mapping.

        Returns:
            Dict: Mapping of class names to relative file paths.
        """
        # Scan files and create new mapping
        self._mapping = self._scan_architecture_files()
        self._save_cache(self._mapping)

        return self._mapping

    def get_file_for_class(self, class_name: str) -> Optional[str]:
        """
        Get the file path for a specific class.

        Args:
            class_name (str): Name of the class to find.

        Returns:
            Optional[str]: Relative path to the file containing the class, or None if not found.
        """
        return self._mapping.get(class_name)

    def refresh(self) -> Dict:
        """
        Force refresh the mapping.

        Returns:
            Dict: Updated mapping dictionary.
        """
        return self.get_mapping()


class_mapper = ClassMapper()


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
    # Lazy import with class mapper
    if network_type not in ARCH_REGISTRY:
        file_name = class_mapper.get_file_for_class(network_type)
        if file_name is None:
            logger.info(
                f'Class [{network_type}] not found in cache. Refreshing class mapper file cache.'
            )
            class_mapper.refresh()
            file_name = class_mapper.get_file_for_class(network_type)

        importlib.import_module(f'pyiqa.archs.{file_name}')

    net = ARCH_REGISTRY.get(network_type)(**opt)
    logger.info(f'Network [{net.__class__.__name__}] is created.')
    return net
