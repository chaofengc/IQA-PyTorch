r"""Q-ReAlign: a modern, model-agnostic Q-Align visual-quality scorer.

Q-ReAlign reproduces the Q-Align recipe on a Qwen3.5-VL backbone (``model_type:
qwen3_5``): the model is asked to rate quality, and the probability mass it places
on the discrete words ``excellent / good / fair / poor / bad`` is collapsed -- via
the fixed weighting ``[1.0, 0.75, 0.5, 0.25, 0.0]`` -- into a single scalar in
``[0, 1]`` (higher = better).

Three public checkpoints (selected with ``model=``):
    mini -> q-future/Q-ReAlign-Mini-0.8B   (fast, throughput champion)
    lite -> q-future/Q-ReAlign-Lite-4B
    pro  -> q-future/Q-ReAlign-Pro-9B       (best average SRCC/PLCC)

Requires ``transformers>=5.0`` (native qwen3_5 from 5.2; 5.0/5.1 run via the
vendored modeling + shim in ``pyiqa.archs.qrealign``).

Reference:
    Wu et al., "Q-Align: Teaching LMMs for Visual Scoring via Discrete Text-Defined
    Levels", arXiv:2312.17090. Q-ReAlign: https://github.com/Q-Future/Q-ReAlign
"""
import torch
from torch import nn
import torchvision.transforms.functional as TF

from pyiqa.utils.registry import ARCH_REGISTRY

# best -> worst; weights map each level to a point in [0, 1]
LEVELS = ["excellent", "good", "fair", "poor", "bad"]
WEIGHTS = [1.0, 0.75, 0.5, 0.25, 0.0]

# named checkpoints
MODELS = {
    "mini": "q-future/Q-ReAlign-Mini-0.8B",
    "lite": "q-future/Q-ReAlign-Lite-4B",
    "pro": "q-future/Q-ReAlign-Pro-9B",
}

# (prompt, answer-stem) per task -- matches the Q-ReAlign toolkit's default inference
TASKS = {
    "quality": ("Can you rate the quality of this picture?", "The quality of the image is"),
    "aesthetic": ("Can you rate the aesthetics of this picture?", "The aesthetics of the image is"),
}


def _level_token_ids(tokenizer, names):
    """First-token id of each level word, space-prefixed as it appears after the stem."""
    ids = []
    for w in names:
        tok = None
        for cand in (" " + w, w):
            t = tokenizer(cand, add_special_tokens=False)["input_ids"]
            if t:
                tok = t[0]
                break
        if tok is None:
            raise ValueError(f"level word {w!r} did not tokenize to any id")
        ids.append(tok)
    return ids


@ARCH_REGISTRY.register()
class QReAlign(nn.Module):
    """Q-ReAlign multimodal visual-quality scorer (Qwen3.5-VL backbone).

    Args:
        model (str): one of ``'mini'`` / ``'lite'`` / ``'pro'`` (see ``MODELS``),
            or any HuggingFace repo id / local path to a Q-ReAlign checkpoint.
        dtype (str): torch dtype passed to ``from_pretrained`` (default ``'auto'``
            -> bfloat16 weights).
        task (str): default scoring task, ``'quality'`` (IQA) or ``'aesthetic'`` (IAA).
    """

    def __init__(self, model="mini", dtype="auto", task="quality"):
        super().__init__()
        # Make qwen3_5 loadable (native on transformers>=5.2; vendored shim on 5.0/5.1).
        # Imported here -- not at module import -- so merely importing pyiqa never
        # patches transformers; only constructing this metric does.
        from .qrealign import qrealign_compat as compat

        self._check_transformers()
        compat.ensure_qwen3_5()

        self.default_task = task
        repo = MODELS.get(model, model)

        from transformers import AutoModelForImageTextToText

        self.processor = compat.load_processor(repo)
        self.model = AutoModelForImageTextToText.from_pretrained(repo, dtype=dtype).eval()

        # collapse the level-word logits into a scalar
        ids = _level_token_ids(self.processor.tokenizer, LEVELS)
        self.register_buffer("level_ids", torch.tensor(ids, dtype=torch.long), persistent=False)
        self.register_buffer("level_weights", torch.tensor(WEIGHTS), persistent=False)

        # the model only does a single forward pass -> silence sampling-flag noise
        gen = getattr(self.model, "generation_config", None)
        if gen is not None and not getattr(gen, "do_sample", False):
            gen.temperature = None
            gen.top_p = None

    @staticmethod
    def _check_transformers():
        import re
        import transformers
        m = re.match(r"(\d+)\.(\d+)", transformers.__version__ or "0.0")
        major, minor = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
        if (major, minor) < (5, 0):
            raise ImportError(
                "The 'qrealign' metric needs transformers>=5.0 (Qwen3.5 backbone); "
                f"found {transformers.__version__}. Please `pip install -U \"transformers>=5.2\"`."
            )

    @torch.no_grad()
    def forward(self, x, task_=None):
        """Score a batch of images.

        Args:
            x (torch.Tensor): ``(B, 3, H, W)`` in ``[0, 1]``.
            task_ (str): override the default task for this call.

        Returns:
            torch.Tensor: shape ``(B,)`` quality scores in ``[0, 1]`` (higher = better).
        """
        prompt, stem = TASKS[task_ or self.default_task]
        dev = self.level_weights.device
        scores = []
        for i in range(x.shape[0]):
            img = TF.to_pil_image(x[i].detach().cpu().clamp(0, 1))
            messages = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ]}]
            text = self.processor.apply_chat_template(messages, add_generation_prompt=True) + stem
            inputs = self.processor(text=[text], images=[img], return_tensors="pt").to(dev)
            logits = self.model(**inputs).logits[0, -1, self.level_ids]
            probs = logits.float().softmax(-1)
            scores.append((probs * self.level_weights).sum())
        return torch.stack(scores).reshape(-1)
