"""Local CLIP preprocessing, tokenizer, model-loading, and inference API. The tokenizer and vocabulary are distributed under the OpenAI CLIP MIT license noted below."""

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
    """Return the path to the packaged byte-pair-encoding vocabulary.

    Returns:
        Path to the packaged BPE vocabulary.
    """
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'bpe_simple_vocab_16e6.txt.gz',
    )


@lru_cache()
def bytes_to_unicode():
    """Build the reversible UTF-8-byte to printable-Unicode mapping used by CLIP BPE.

    Returns:
        Mapping from UTF-8 byte values to reversible Unicode characters.
    """
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
    """Return adjacent symbol pairs in a BPE token representation.

    Args:
        word: Single text token to apply byte-pair merges to.

    Returns:
        Set of adjacent symbol pairs.
    """
    return set(zip(word[:-1], word[1:]))


def _basic_clean(text):
    """Normalize text using HTML entity decoding and Unicode cleanup.

    Args:
        text: Text string or sequence of strings, depending on the API.

    Returns:
        Cleaned text or converted RGB image.
    """
    return html.unescape(html.unescape(ftfy.fix_text(text))).strip()


def _whitespace_clean(text):
    """Collapse repeated whitespace and trim leading and trailing spaces.

    Args:
        text: Text string or sequence of strings, depending on the API.

    Returns:
        Cleaned text or converted RGB image.
    """
    return re.sub(r'\s+', ' ', text).strip()


class SimpleTokenizer:

    """Byte-pair tokenizer that encodes and decodes text using the packaged CLIP vocabulary.

    """
    def __init__(self, bpe_path=default_bpe()):
        """Initialize the simple tokenizer and configure its layers, parameters, and optional pretrained state.

        Args:
            bpe_path: Filesystem path to an input file or checkpoint.
        """
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
        """Merge the token symbols using the tokenizer byte-pair ranks and cache the merged representation.

        Args:
            token: Token string to merge or map to vocabulary IDs.

        Returns:
            Space-separated string of merged BPE symbols.
        """
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
        """Clean, normalize, and byte-pair encode one text string into vocabulary IDs.

        Args:
            text: Text string or sequence of strings, depending on the API.


        Returns:
            List of integer BPE token IDs (without start/end-of-text markers).
        """
        tokens = []
        text = _whitespace_clean(_basic_clean(text)).lower()
        for token in re.findall(self.pattern, text):
            token = ''.join(self.byte_encoder[value] for value in token.encode('utf-8'))
            tokens.extend(self.encoder[piece] for piece in self.bpe(token).split(' '))
        return tokens

    def decode(self, tokens):
        """Convert BPE token IDs back to readable UTF-8 text.

        Args:
            tokens: Token IDs, usually an integer tensor shaped ``(B, L)``.


        Returns:
            Decoded string with word-ending markers converted to spaces.
        """
        text = ''.join(self.decoder[token] for token in tokens)
        byte_values = bytearray(self.byte_decoder[value] for value in text)
        return byte_values.decode('utf-8', errors='replace').replace('</w>', ' ')


_tokenizer = SimpleTokenizer()


class _OpenAIClipModel(nn.Module):

    """Thin module wrapper exposing the local CLIP model through the OpenAI-compatible inference API.

    """
    def __init__(self, model):
        """Initialize the open aiclip model and configure its layers, parameters, and optional pretrained state.

        Args:
            model: Model instance whose parameters or attributes are used by this helper.
        """
        super().__init__()
        self.model = model
        self.eval()

    def __getattr__(self, name):
        """Delegate attribute lookup to the wrapped model when the wrapper does not define the requested name.

        Args:
            name: Registered model name or identifier.

        Returns:
            The wrapped model attribute with the requested name.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__('model'), name)

    def encode_image(self, image):
        """Encode a batch of RGB images into the wrapped model shared embedding space.

        Args:
            image: RGB image tensor shaped ``(B, 3, H, W)``.

        Returns:
            Image embedding tensor shaped ``(B, embed_dim)``.
        """
        return self.model.encode_image(image, pos_embedding=True)

    def encode_text(self, text):
        """Encode CLIP token IDs into the wrapped model shared embedding space.

        Args:
            text: Integer CLIP token tensor shaped ``(B, L)``.

        Returns:
            Text embedding tensor shaped ``(B, embed_dim)``.
        """
        return self.model.encode_text(text)

    def forward(self, image, text):
        """Compute image-to-text and text-to-image similarity logits with the wrapped CLIP model.

        Args:
            image: RGB image tensor shaped ``(B_image, 3, H, W)``.
            text: Integer CLIP token tensor shaped ``(B_text, L)``.

        Returns:
            Pair of similarity-logit tensors shaped ``(B_image, B_text)`` and ``(B_text, B_image)``.
        """
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits_per_image = self.logit_scale.exp() * image_features @ text_features.t()
        return logits_per_image, logits_per_image.t()


def available_models() -> List[str]:
    """Return the names of CLIP checkpoints supported by the local API.

    Returns:
        Sequence of supported model names.
    """
    from .clip_model import available_models as _available_models

    return _available_models()


def _transform(image_size):
    """Compose RGB conversion, CLIP resize/center-crop, tensor conversion, and channel normalization for ``image_size``.

    Args:
        image_size: Requested spatial or sequence dimension, compatible with the model configuration.

    Returns:
        Torchvision transform composition for a PIL image.
    """
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
    """Convert a PIL image to RGB mode.

    Args:
        image: Batched image tensor, normally RGB in ``(B, 3, H, W)`` layout.

    Returns:
        Cleaned text or converted RGB image.
    """
    return image.convert('RGB')


def load(
    name: str,
    device: Union[str, torch.device] = 'cuda' if torch.cuda.is_available() else 'cpu',
    jit: bool = False,
    download_root: str = None,
):
    """Load a named CLIP model and its image preprocessing transform.

    Args:
        name: Registered model name or identifier.
        device: Target PyTorch device, such as CPU or CUDA.
        jit: Whether to load a TorchScript checkpoint when supported.
        download_root: Optional directory for cached model downloads.

    Returns:
        Computed result; type and shape follow the supplied inputs and model configuration.
    """
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
    """Tokenize one or more text strings into padded CLIP context-length token IDs.

    Args:
        texts: Sequence of strings to tokenize.
        context_length: Maximum number of tokens in the CLIP text context.
        truncate: Whether to truncate text that exceeds the requested context length.

    Returns:
        Integer token tensor with shape ``(N, context_length)``.
    """
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
