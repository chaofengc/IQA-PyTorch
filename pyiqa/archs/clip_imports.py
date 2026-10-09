"""Compatibility exports for the repository-local CLIP API and tokenizer."""

from . import clip_api as clip
from .clip_api import SimpleTokenizer

__all__ = ['clip', 'SimpleTokenizer']
