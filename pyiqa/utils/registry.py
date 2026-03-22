# Modified from: https://github.com/facebookresearch/fvcore/blob/master/fvcore/common/registry.py  # noqa: E501
import difflib
import importlib
import warnings

try:
    from importlib import metadata as importlib_metadata
except ImportError:  # pragma: no cover
    importlib_metadata = None


class Registry:
    """
    The registry that provides name -> object mapping, to support third-party
    users' custom modules.

    To create a registry (e.g. a backbone registry):

    .. code-block:: python

        BACKBONE_REGISTRY = Registry('BACKBONE')

    To register an object:

    .. code-block:: python

        @BACKBONE_REGISTRY.register()
        class MyBackbone():
            ...

    Or:

    .. code-block:: python

        BACKBONE_REGISTRY.register(MyBackbone)
    """

    def __init__(self, name, entry_point_group=None):
        """
        Args:
            name (str): the name of this registry.
            entry_point_group (str | None): Optional plugin group name.
        """
        self._name = name
        self._obj_map = {}
        self._lazy_map = {}
        self._entry_point_group = entry_point_group
        self._entry_points_loaded = False

    def _do_register(self, name, obj, override=False):
        if (not override) and (name in self._obj_map):
            raise AssertionError(
                f"An object named '{name}' was already registered "
                f"in '{self._name}' registry!"
            )
        self._obj_map[name] = obj
        self._lazy_map.pop(name, None)

    def _register_with_aliases(self, obj, name=None, aliases=None, override=False):
        primary_name = name or obj.__name__
        names = [primary_name]
        if aliases:
            names.extend(aliases)

        for item_name in names:
            self._do_register(item_name, obj, override=override)

    @staticmethod
    def _split_import_path(import_path):
        if ':' in import_path:
            module_name, attr_name = import_path.split(':', 1)
            return module_name.strip(), attr_name.strip()
        return import_path.strip(), None

    def _resolve_lazy(self, name):
        import_path = self._lazy_map.get(name)
        if import_path is None:
            return

        module_name, attr_name = self._split_import_path(import_path)
        module = importlib.import_module(module_name)

        if attr_name:
            obj = getattr(module, attr_name)
            if name not in self._obj_map:
                self._do_register(name, obj, override=True)

    def _load_entry_points_once(self):
        if self._entry_points_loaded:
            return
        self._entry_points_loaded = True

        if not self._entry_point_group:
            return
        if importlib_metadata is None:  # pragma: no cover
            return

        try:
            entry_points = importlib_metadata.entry_points()
            if hasattr(entry_points, 'select'):
                group_eps = entry_points.select(group=self._entry_point_group)
            else:  # pragma: no cover
                group_eps = entry_points.get(self._entry_point_group, [])
        except Exception as error:
            warnings.warn(
                f"Failed to query entry points for '{self._entry_point_group}': {error}",
                RuntimeWarning,
            )
            return

        for ep in group_eps:
            try:
                loaded = ep.load()
                if callable(loaded):
                    loaded()
            except Exception as error:
                warnings.warn(
                    f"Failed to load registry plugin '{ep.name}' from '{self._entry_point_group}': {error}",
                    RuntimeWarning,
                )

    def register_lazy(self, name, import_path, aliases=None, override=False):
        """Register an object lazily from an import path.

        Args:
            name (str): Name used in this registry.
            import_path (str): 'module.submodule' or 'module.submodule:ClassOrFn'.
            aliases (list[str] | tuple[str] | None): Optional alias names.
            override (bool): Whether to override existing registrations.
        """
        names = [name]
        if aliases:
            names.extend(aliases)

        for item_name in names:
            if (not override) and (item_name in self._obj_map or item_name in self._lazy_map):
                raise AssertionError(
                    f"An object named '{item_name}' was already registered "
                    f"in '{self._name}' registry!"
                )
            self._lazy_map[item_name] = import_path

    def register(self, obj=None, name=None, aliases=None, override=False):
        """Register the given object.

        Supports decorator and direct-call forms.
        """
        if aliases is not None and not isinstance(aliases, (list, tuple)):
            raise TypeError('aliases must be a list or tuple of strings.')
        if aliases is not None and not all(isinstance(alias, str) for alias in aliases):
            raise TypeError('Each alias must be a string.')
        if name is not None and not isinstance(name, str):
            raise TypeError('name must be a string.')

        if obj is None:
            # used as a decorator
            def deco(func_or_class):
                self._register_with_aliases(
                    func_or_class,
                    name=name,
                    aliases=aliases,
                    override=override,
                )
                return func_or_class

            return deco

        # used as a function call
        self._register_with_aliases(obj, name=name, aliases=aliases, override=override)
        return obj

    def get(self, name):
        self._resolve_lazy(name)
        self._load_entry_points_once()
        self._resolve_lazy(name)

        ret = self._obj_map.get(name)
        if ret is None:
            candidates = list(self._obj_map.keys()) + list(self._lazy_map.keys())
            hint = ''
            if candidates:
                close = difflib.get_close_matches(name, candidates, n=5)
                if close:
                    hint = f" Did you mean one of: {', '.join(close)}?"
            raise KeyError(
                f"No object named '{name}' found in '{self._name}' registry!"
                + hint
            )
        return ret

    def __contains__(self, name):
        if name in self._obj_map or name in self._lazy_map:
            return True
        self._load_entry_points_once()
        return name in self._obj_map or name in self._lazy_map

    def __iter__(self):
        self._load_entry_points_once()
        return iter(self._obj_map.items())

    def keys(self):
        self._load_entry_points_once()
        return self._obj_map.keys()


DATASET_REGISTRY = Registry('dataset', entry_point_group='pyiqa.datasets')
ARCH_REGISTRY = Registry('arch', entry_point_group='pyiqa.archs')
MODEL_REGISTRY = Registry('model', entry_point_group='pyiqa.models')
LOSS_REGISTRY = Registry('loss', entry_point_group='pyiqa.losses')
METRIC_REGISTRY = Registry('metric', entry_point_group='pyiqa.metrics')
