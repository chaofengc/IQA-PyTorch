import pytest

from pyiqa.archs.clip_tokenizer import SimpleTokenizer


def test_clip_tokenizer_encodes_and_decodes_known_text():
    tokenizer = SimpleTokenizer()

    tokens = tokenizer.encode('hello world')

    assert tokens == [3306, 1002]
    assert tokenizer.decode(tokens) == 'hello world '


def test_clip_tokenizer_handles_unicode_and_html_entities():
    tokenizer = SimpleTokenizer()

    tokens = tokenizer.encode('café &amp; tea')

    assert tokenizer.decode(tokens).strip() == 'café & tea'
