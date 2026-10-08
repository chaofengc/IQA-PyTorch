# CLIP tokenizer implementation and vocabulary are distributed under the
# OpenAI CLIP MIT license in LICENSE-OpenAI-CLIP.
import gzip
import html
import os
from functools import lru_cache
from typing import List, Union

import ftfy
import regex as re
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    Resize,
    ToTensor,
)

try:
    from torchvision.transforms import InterpolationMode

    _BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    _BICUBIC = Image.BICUBIC


@lru_cache()
def default_bpe():
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'bpe_simple_vocab_16e6.txt.gz',
    )


@lru_cache()
def bytes_to_unicode():
    byte_values = (
        list(range(ord('!'),
                   ord('~') + 1)) + list(range(ord('¡'),
                                               ord('¬') + 1)) + list(range(ord('®'),
                                                                           ord('ÿ') + 1)))
    unicode_values = byte_values[:]
    extra = 0
    for value in range(2**8):
        if value not in byte_values:
            byte_values.append(value)
            unicode_values.append(2**8 + extra)
            extra += 1
    return dict(zip(byte_values, (chr(value) for value in unicode_values)))


def _get_pairs(word):
    return set(zip(word[:-1], word[1:]))


def _basic_clean(text):
    return html.unescape(html.unescape(ftfy.fix_text(text))).strip()


def _whitespace_clean(text):
    return re.sub(r'\s+', ' ', text).strip()


class SimpleTokenizer:

    def __init__(self, bpe_path=default_bpe()):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {value: key for key, value in self.byte_encoder.items()}
        with gzip.open(bpe_path) as bpe_file:
            merges = bpe_file.read().decode('utf-8').split('\n')[1:49152 - 256 - 2 + 1]
        merges = [tuple(merge.split()) for merge in merges]
        vocabulary = list(self.byte_encoder.values())
        vocabulary += [value + '</w>' for value in vocabulary]
        vocabulary += [''.join(merge) for merge in merges]
        vocabulary.extend(['<|startoftext|>', '<|endoftext|>'])
        self.encoder = dict(zip(vocabulary, range(len(vocabulary))))
        self.decoder = {value: key for key, value in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
        self.pattern = re.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE,
        )

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + '</w>', )
        pairs = _get_pairs(word)
        while pairs:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            merged = []
            index = 0
            while index < len(word):
                try:
                    next_index = word.index(first, index)
                except ValueError:
                    merged.extend(word[index:])
                    break
                merged.extend(word[index:next_index])
                index = next_index
                if index < len(word) - 1 and word[index + 1] == second:
                    merged.append(first + second)
                    index += 2
                else:
                    merged.append(word[index])
                    index += 1
            word = tuple(merged)
            if len(word) == 1:
                break
            pairs = _get_pairs(word)
        result = ' '.join(word)
        self.cache[token] = result
        return result

    def encode(self, text):
        tokens = []
        text = _whitespace_clean(_basic_clean(text)).lower()
        for token in re.findall(self.pattern, text):
            token = ''.join(self.byte_encoder[value] for value in token.encode('utf-8'))
            tokens.extend(self.encoder[piece] for piece in self.bpe(token).split(' '))
        return tokens

    def decode(self, tokens):
        text = ''.join(self.decoder[token] for token in tokens)
        byte_values = bytearray(self.byte_decoder[value] for value in text)
        return byte_values.decode('utf-8', errors='replace').replace('</w>', ' ')


_tokenizer = SimpleTokenizer()


class _OpenAIClipModel(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.eval()

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__('model'), name)

    def encode_image(self, image):
        return self.model.encode_image(image, pos_embedding=True)

    def encode_text(self, text):
        return self.model.encode_text(text)

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits_per_image = self.logit_scale.exp() * image_features @ text_features.t()
        return logits_per_image, logits_per_image.t()


def available_models() -> List[str]:
    from .clip_model import available_models as _available_models

    return _available_models()


def _transform(image_size):
    return Compose([
        Resize(image_size, interpolation=_BICUBIC),
        CenterCrop(image_size),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        ),
    ])


def _convert_image_to_rgb(image):
    return image.convert('RGB')


def load(
    name: str,
    device: Union[str, torch.device] = 'cuda' if torch.cuda.is_available() else 'cpu',
    jit: bool = False,
    download_root: str = None,
):
    from .clip_model import load as _load_model

    model = _load_model(
        name,
        device=device,
        jit=jit,
        download_root=download_root or os.path.expanduser('~/.cache/clip'),
    )
    if not isinstance(model, torch.jit.ScriptModule):
        model = _OpenAIClipModel(model)
    try:
        image_size = model.visual.input_resolution
    except AttributeError:
        image_size = model.input_resolution
    image_size = int(image_size)
    return model, _transform(image_size)


def tokenize(
    texts: Union[str, List[str]],
    context_length: int = 77,
    truncate: bool = False,
) -> torch.LongTensor:
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder['<|startoftext|>']
    eot_token = _tokenizer.encoder['<|endoftext|>']
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    for index, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if not truncate:
                raise RuntimeError(f'Input {texts[index]} is too long for context length {context_length}')
            tokens = tokens[:context_length]
            tokens[-1] = eot_token
        result[index, :len(tokens)] = torch.tensor(tokens)
    return result


__all__ = ['SimpleTokenizer', 'available_models', 'load', 'tokenize']
