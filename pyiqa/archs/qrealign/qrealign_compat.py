"""Back-compat shim so transformers 5.0/5.1 can run the vendored ``qwen3_5`` model.

``qwen3_5`` (the Qwen3.5-VL backbone behind Q-ReAlign) is natively supported only
in transformers >= 5.2. This module makes the *vendored* modeling code also run on
transformers 5.0/5.1 (and even 4.57) by patching the handful of internal symbols
that were added/renamed between those releases. It is **idempotent** and only
patches what is missing, so importing it on transformers >= 5.2 is a harmless no-op.

Import this module BEFORE importing/using ``qwen3_5``. Public helpers:
  - ``ensure_qwen3_5()``    -- make ``qwen3_5`` loadable (native if present, else vendored)
  - ``load_processor(ckpt)``-- version-robust image-only Qwen3-VL processor loader
"""
import sys, types, contextlib, importlib, os
import transformers

# 1) transformers.initialization (only used by _init_weights, never on load).
#    The real submodule may exist but not be a lazy attr on the package, so
#    import-test it -- stubbing over the real one breaks transformers itself.
try:
    importlib.import_module("transformers.initialization"); _have_init = True
except Exception:
    _have_init = False
if not _have_init:
    import torch
    init = types.ModuleType("transformers.initialization")
    init.ones_ = torch.nn.init.ones_; init.zeros_ = torch.nn.init.zeros_
    def _copy_(dst, src):
        with torch.no_grad(): dst.copy_(src)
        return dst
    init.copy_ = _copy_
    sys.modules["transformers.initialization"] = init; transformers.initialization = init

# 2) utils.output_capturing.capture_outputs -> identity decorator
try:
    oc = importlib.import_module("transformers.utils.output_capturing")
except Exception:
    oc = types.ModuleType("transformers.utils.output_capturing")
    sys.modules["transformers.utils.output_capturing"] = oc
if not hasattr(oc, "capture_outputs"):
    def capture_outputs(fn): return fn
    oc.capture_outputs = capture_outputs

# 3) utils.generic: merge_with_config_defaults / maybe_autocast / is_flash_attention_requested
g = importlib.import_module("transformers.utils.generic")
if not hasattr(g, "merge_with_config_defaults"):
    def merge_with_config_defaults(fn): return fn
    g.merge_with_config_defaults = merge_with_config_defaults
if not hasattr(g, "maybe_autocast"):
    import torch
    @contextlib.contextmanager
    def maybe_autocast(device_type="cuda", enabled=True, **kw):
        if enabled and device_type == "cuda" and torch.cuda.is_available():
            with torch.autocast(device_type=device_type, **kw): yield
        else:
            yield
    g.maybe_autocast = maybe_autocast
if not hasattr(g, "is_flash_attention_requested"):
    def is_flash_attention_requested(config):
        impl = getattr(config, "_attn_implementation", None) or getattr(config, "attn_implementation", None)
        return bool(impl) and "flash" in str(impl)
    g.is_flash_attention_requested = is_flash_attention_requested

# 4) utils.torch_compilable_check -> no-op
u = importlib.import_module("transformers.utils")
if not hasattr(u, "torch_compilable_check"):
    def torch_compilable_check(*a, **k): return None
    u.torch_compilable_check = torch_compilable_check

# 5) integrations.use_kernelized_func -> decorator factory returning fn unchanged
integ = importlib.import_module("transformers.integrations")
if not hasattr(integ, "use_kernelized_func"):
    def use_kernelized_func(*da, **dk):
        def deco(fn): return fn
        return deco
    integ.use_kernelized_func = use_kernelized_func

# 6) configuration_utils.PreTrainedConfig alias (renamed from PretrainedConfig in 5.x)
cu = importlib.import_module("transformers.configuration_utils")
if not hasattr(cu, "PreTrainedConfig") and hasattr(cu, "PretrainedConfig"):
    cu.PreTrainedConfig = cu.PretrainedConfig

# 7) AttentionInterface.get_interface (added in 5.2; older versions use .get)
try:
    from transformers.modeling_utils import AttentionInterface as _AI
    if not hasattr(_AI, "get_interface"):
        def get_interface(self, key, default=None):
            fn = self.get(key)
            return fn if fn is not None else default
        _AI.get_interface = get_interface
except Exception:
    pass

# 8) masking_utils.create_causal_mask: 5.2 renamed input_embeds -> inputs_embeds
import inspect as _inspect
_mu = importlib.import_module("transformers.masking_utils")
_ccm = _mu.create_causal_mask
if "inputs_embeds" not in _inspect.signature(_ccm).parameters:
    def _ccm_compat(*args, **kw):
        if "inputs_embeds" in kw and "input_embeds" not in kw:
            kw["input_embeds"] = kw.pop("inputs_embeds")
        return _ccm(*args, **kw)
    _mu.create_causal_mask = _ccm_compat

# 9) modeling_rope_utils.RopeParameters (5.x typing construct; annotation-only)
_mr = importlib.import_module("transformers.modeling_rope_utils")
if not hasattr(_mr, "RopeParameters"):
    _mr.RopeParameters = dict

# 10) auto_docstring: 4.57's version crashes parsing 5.2-style union annotations.
#     It is purely cosmetic -> replace with a no-op handling @deco and @deco(...).
def _noop_auto_docstring(obj=None, *a, **k):
    if callable(obj) and not a and not k:
        return obj            # used as @auto_docstring
    def deco(fn): return fn   # used as @auto_docstring(...)
    return deco
for _m in ["transformers.utils", "transformers.utils.auto_docstring", "transformers"]:
    try:
        _mod = importlib.import_module(_m)
        if hasattr(_mod, "auto_docstring"):
            _mod.auto_docstring = _noop_auto_docstring
    except Exception:
        pass

# 11) PretrainedConfig.to_json_string: 4.57 cannot serialize set-valued config
#     fields (5.2 config stores some as sets). Coerce sets -> sorted lists.
_cu = importlib.import_module("transformers.configuration_utils")
_PC = getattr(_cu, "PreTrainedConfig", None) or getattr(_cu, "PretrainedConfig", None)
if _PC is not None and not getattr(_PC, "_qrealign_setpatch", False):
    _orig_to_dict = _PC.to_dict
    def _desetify(o):
        if isinstance(o, set): return sorted(o, key=str)
        if isinstance(o, dict): return {k: _desetify(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return type(o)(_desetify(v) for v in o)
        return o
    def to_dict(self):
        return _desetify(_orig_to_dict(self))
    _PC.to_dict = to_dict
    _PC._qrealign_setpatch = True

# 12) Alias qwen3_5 -> qwen3_vl in the video/image/processing auto-maps so the
#     manual processor path resolves on transformers < 5.2 (image-only Q-ReAlign
#     never invokes the video path).
for _modname, _attr in [
    ("transformers.models.auto.video_processing_auto", "VIDEO_PROCESSOR_MAPPING_NAMES"),
    ("transformers.models.auto.image_processing_auto", "IMAGE_PROCESSOR_MAPPING_NAMES"),
    ("transformers.models.auto.processing_auto", "PROCESSOR_MAPPING_NAMES"),
]:
    try:
        _nm = getattr(importlib.import_module(_modname), _attr)
        if "qwen3_5" not in _nm and "qwen3_vl" in _nm:
            _nm["qwen3_5"] = _nm["qwen3_vl"]
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------
_QWEN35_SRC = os.path.join(os.path.dirname(__file__), "qwen3_5_src")


def _qwen3_5_is_native():
    """True if the installed transformers ships qwen3_5 (>= 5.2)."""
    try:
        importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
        return True
    except Exception:
        return False


def ensure_qwen3_5():
    """Make the ``qwen3_5`` architecture importable + Auto*-registered.

    On transformers >= 5.2 this is a no-op (native support). On 5.0/5.1 the
    vendored modeling files are injected into the installed transformers tree and
    registered with the Auto* factories. Returns ``(Qwen3_5Config, Qwen3_5ForConditionalGeneration)``.
    """
    if not _qwen3_5_is_native():
        import shutil
        tdir = os.path.dirname(transformers.__file__)
        dst = os.path.join(tdir, "models", "qwen3_5")
        try:
            os.makedirs(dst, exist_ok=True)
            for f in ["__init__.py", "configuration_qwen3_5.py",
                      "modeling_qwen3_5.py", "tokenization_qwen3_5.py"]:
                s = os.path.join(_QWEN35_SRC, f)
                if os.path.exists(s):
                    shutil.copy(s, os.path.join(dst, f))
        except OSError as e:
            raise RuntimeError(
                "qrealign needs qwen3_5 support: transformers < 5.2 detected and the "
                "vendored modeling files could not be installed into the transformers "
                f"package ({e}). Upgrade to transformers>=5.2, or make the transformers "
                "install writable."
            ) from e
    from transformers import AutoConfig, AutoModelForImageTextToText
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
    try:
        AutoConfig.register("qwen3_5", Qwen3_5Config)
        AutoModelForImageTextToText.register(Qwen3_5Config, Qwen3_5ForConditionalGeneration)
    except Exception:
        pass  # already registered (native, or a previous call)
    return Qwen3_5Config, Qwen3_5ForConditionalGeneration


def load_processor(ckpt):
    """Load the Q-ReAlign processor robustly across transformers 5.0 -> 5.x.

    Prefers the native ``AutoProcessor`` (works on 5.0+). Falls back to building an
    image-only ``Qwen3VLProcessor`` for releases whose ``AutoProcessor`` demands a
    video sub-processor that image-only Q-ReAlign never uses.
    """
    from transformers import AutoProcessor
    try:
        return AutoProcessor.from_pretrained(ckpt)
    except Exception:
        from transformers import AutoImageProcessor
        from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
        img = AutoImageProcessor.from_pretrained(ckpt)
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(ckpt)
        except Exception:
            # 4.57 lacks the 5.x TokenizersBackend class; qwen3_5's tokenizer is a
            # Qwen2-family BPE, so load it concretely (also satisfies the processor's
            # tokenizer-class check, which expects Qwen2Tokenizer/Qwen2TokenizerFast).
            from transformers import Qwen2TokenizerFast
            tok = Qwen2TokenizerFast.from_pretrained(ckpt)
        ctp = os.path.join(ckpt, "chat_template.jinja")
        ct = open(ctp).read() if os.path.exists(ctp) else None
        # Some releases require a real video_processor instance; a default one
        # satisfies the constructor and is never used for images.
        vp = None
        try:
            from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor
            vp = Qwen3VLVideoProcessor()
        except Exception:
            pass
        return Qwen3VLProcessor(image_processor=img, tokenizer=tok,
                                video_processor=vp, chat_template=ct)
