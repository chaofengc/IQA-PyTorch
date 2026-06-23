"""Vendored Qwen3.5 (``qwen3_5``) modeling + back-compat shim for the Q-ReAlign metric.

Q-ReAlign is built on the Qwen3.5-VL backbone, which transformers supports natively
only from 5.2 onward. This subpackage carries the modeling source and a shim so the
metric also runs on transformers 5.0/5.1. See ``qrealign_compat`` for details.
"""
from .qrealign_compat import ensure_qwen3_5, load_processor

__all__ = ["ensure_qwen3_5", "load_processor"]
