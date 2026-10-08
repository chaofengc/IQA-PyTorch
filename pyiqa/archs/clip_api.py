from functools import lru_cache

import torch

from .clip_model import available_models, load as load_model
from .clip_tokenizer import SimpleTokenizer


@lru_cache()
def _tokenizer():
    return SimpleTokenizer()


def tokenize(texts, context_length=77, truncate=False):
    """Convert one or more strings to OpenAI CLIP token IDs."""
    if isinstance(texts, str):
        texts = [texts]

    if context_length < 2:
        raise ValueError('context_length must be at least 2')

    tokenizer = _tokenizer()
    sot_token = tokenizer.encoder['<|startoftext|>']
    eot_token = tokenizer.encoder['<|endoftext|>']
    all_tokens = [[sot_token] + tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for row, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if not truncate:
                raise RuntimeError(
                    f'Input {texts[row]!r} is too long for context length {context_length}'
                )
            tokens = tokens[:context_length]
            tokens[-1] = eot_token
        result[row, : len(tokens)] = torch.tensor(tokens)

    return result


def load(
    name,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    jit=False,
    download_root=None,
):
    """Load a CLIP model and the standard PIL image preprocessing transform."""
    model = load_model(name, device=device, jit=jit, download_root=download_root)

    from torchvision.transforms import (
        CenterCrop,
        Compose,
        InterpolationMode,
        Normalize,
        Resize,
        ToTensor,
    )

    def convert_image_to_rgb(image):
        return image.convert('RGB')

    preprocess = Compose(
        [
            Resize(model.visual.input_resolution, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(model.visual.input_resolution),
            convert_image_to_rgb,
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    return model, preprocess


__all__ = ['available_models', 'load', 'tokenize']
