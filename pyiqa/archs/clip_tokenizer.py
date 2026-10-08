"""OpenAI CLIP's byte-pair tokenizer.

The BPE vocabulary and tokenizer algorithm are from OpenAI's MIT-licensed CLIP
implementation; see ``CLIP_LICENSE``.
"""

import gzip
import html
import os
from functools import lru_cache

import ftfy
import regex


@lru_cache()
def default_bpe():
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'bpe_simple_vocab_16e6.txt.gz'
    )


@lru_cache()
def bytes_to_unicode():
    byte_values = (
        list(range(ord('!'), ord('~') + 1))
        + list(range(ord('¡'), ord('¬') + 1))
        + list(range(ord('®'), ord('ÿ') + 1))
    )
    unicode_values = byte_values[:]
    extra = 0
    for value in range(2**8):
        if value not in byte_values:
            byte_values.append(value)
            unicode_values.append(2**8 + extra)
            extra += 1
    return dict(zip(byte_values, (chr(value) for value in unicode_values)))


def _get_pairs(word):
    return set(zip(word, word[1:]))


def _basic_clean(text):
    return html.unescape(html.unescape(ftfy.fix_text(text))).strip()


def _whitespace_clean(text):
    return regex.sub(r'\s+', ' ', text).strip()


class SimpleTokenizer:
    def __init__(self, bpe_path=None):
        if bpe_path is None:
            bpe_path = default_bpe()

        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {value: key for key, value in self.byte_encoder.items()}
        with gzip.open(bpe_path, 'rt', encoding='utf-8') as vocab_file:
            merges = vocab_file.read().split('\n')[1 : 49152 - 256 - 2 + 1]
        merges = [tuple(merge.split()) for merge in merges]

        vocab = list(self.byte_encoder.values())
        vocab.extend(value + '</w>' for value in self.byte_encoder.values())
        vocab.extend(''.join(merge) for merge in merges)
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {value: key for key, value in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {
            '<|startoftext|>': '<|startoftext|>',
            '<|endoftext|>': '<|endoftext|>',
        }
        self.pattern = regex.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            regex.IGNORECASE,
        )

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]

        word = tuple(token[:-1]) + (token[-1] + '</w>',)
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
        for token in regex.findall(self.pattern, text):
            token = ''.join(self.byte_encoder[value] for value in token.encode('utf-8'))
            tokens.extend(
                self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')
            )
        return tokens

    def decode(self, tokens):
        text = ''.join(self.decoder[token] for token in tokens)
        byte_values = bytearray(self.byte_decoder[value] for value in text)
        return byte_values.decode('utf-8', errors='replace').replace('</w>', ' ')
