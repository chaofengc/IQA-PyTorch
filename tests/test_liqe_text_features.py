"""LIQE text-helper regression tests without model construction or downloads.

Compile the actual helper from its AST so the unit tests need neither Torch nor
CLIP. The stand-ins check token/device selection and the existing normalization;
they do not establish neural inference or GPU numerical parity.
"""
import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest


SOURCE = Path(__file__).resolve().parents[1] / 'pyiqa' / 'archs' / 'liqe_arch.py'


def load_helper():
    tree = ast.parse(SOURCE.read_text())
    model = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'LIQE')
    method = next(node for node in model.body if isinstance(node, ast.FunctionDef) and node.name == 'get_text_features')
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace['get_text_features']


class Tokens:
    def __init__(self, name, device='cpu'):
        self.name = name
        self.device = device

    def to(self, device):
        return Tokens(self.name, device)


class Features:
    def __init__(self, values):
        self.values = values

    def norm(self, dim, keepdim):
        assert dim == 1
        assert keepdim is True
        return math.sqrt(sum(value * value for value in self.values))

    def __truediv__(self, denominator):
        return [value / denominator for value in self.values]


class Encoder:
    def __init__(self, device):
        self.device = device
        self.received = []

    def parameters(self):
        yield SimpleNamespace(device=self.device)

    def encode_text(self, tokens):
        self.received.append(tokens)
        if tokens.device != self.device:
            raise ValueError('tokens and encoder must have the same device')
        return Features([3.0, 4.0])


class LIQETextFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.helper = staticmethod(load_helper())

    def test_explicit_tokens_replace_constructor_prompts(self):
        encoder = Encoder('cpu')
        model = SimpleNamespace(clip_model=encoder, joint_texts=Tokens('original'))
        result = self.helper(model, Tokens('requested'))
        self.assertEqual(encoder.received[0].name, 'requested')
        self.assertEqual(result, [0.6, 0.8])

    def test_cache_loaded_model_does_not_need_joint_texts(self):
        encoder = Encoder('cpu')
        model = SimpleNamespace(clip_model=encoder)
        self.assertEqual(self.helper(model, Tokens('requested')), [0.6, 0.8])
        self.assertEqual(encoder.received[0].name, 'requested')

    def test_tokens_follow_encoder_device(self):
        for encoder_device, input_device in [('accelerator:0', 'cpu'), ('cpu', 'accelerator:0')]:
            with self.subTest(encoder_device=encoder_device, input_device=input_device):
                encoder = Encoder(encoder_device)
                model = SimpleNamespace(clip_model=encoder, joint_texts=Tokens('original'))
                self.assertEqual(self.helper(model, Tokens('requested', input_device)), [0.6, 0.8])
                self.assertEqual(encoder.received[0].device, encoder_device)
                self.assertEqual(encoder.received[0].name, 'requested')


if __name__ == '__main__':
    unittest.main()
