r"""Quality-Aware Image-Text Alignment for Real-World Image Quality Assessment

@article{agnolucci2024qualityaware,
      title={Quality-Aware Image-Text Alignment for Real-World Image Quality Assessment},
      author={Agnolucci, Lorenzo and Galteri, Leonardo and Bertini, Marco},
      journal={arXiv preprint arXiv:2403.11176},
      year={2024}
}

Reference:
    - Arxiv link: https://arxiv.org/abs/2403.11176
    - Official Github: https://github.com/miccunifi/QualiCLIP
"""

import torch
import torch.nn as nn
import clip
from clip.simple_tokenizer import SimpleTokenizer

from pyiqa.utils.registry import ARCH_REGISTRY
from pyiqa.archs.arch_util import (
    get_url_from_name,
    load_pretrained_network,
    load_file_from_url,
)
from pyiqa.archs.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from pyiqa.archs.clip_model import load

default_model_urls = {
    'qualiclip': get_url_from_name('QualiCLIP.pth'),
    'qualiclip+': get_url_from_name('QualiCLIP+_koniq.pth'),
    'qualiclip+-clive': get_url_from_name('QualiCLIP+_clive.pth'),
    'qualiclip+-flive': get_url_from_name('QualiCLIP+_flive.pth'),
    'qualiclip+-spaq': get_url_from_name('QualiCLIP+_spaq.pth'),
}


class PromptLearner(nn.Module):
    """
    PromptLearner class for learning prompts for QualiCLIP+. See https://github.com/IceClear/CLIP-IQA for reference.
    """

    def __init__(
        self, clip_model, prompt_pairs, n_ctx=16, ctx_init='', prompt_specific_ctx=False
    ) -> None:
        """
        Initialize the PromptLearner.

        Args:
            clip_model (nn.Module): The CLIP model.
            prompt_pairs (list): List of antonym prompt pairs.
            n_ctx (int): Number of context tokens. Default is 16.
            ctx_init (str): String used for initializing the context tokens. Default is ''.
            prompt_specific_ctx (bool): Whether to learn context tokens for each input prompt. Default is False.
        """
        super().__init__()

        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        tokenizer = SimpleTokenizer()

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace('_', ' ')
            n_ctx = len(ctx_init.split(' '))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                init_embedding = clip_model.token_embedding(prompt).type(dtype)
            if prompt_specific_ctx:
                init_ctx = init_embedding[:, 1 : 1 + n_ctx].repeat(
                    len(prompt_pairs), 1, 1
                )
            else:
                init_ctx = init_embedding[0, 1 : 1 + n_ctx]
            prompt_prefix = ctx_init
        else:
            if prompt_specific_ctx:
                init_ctx = torch.empty(len(prompt_pairs), n_ctx, ctx_dim, dtype=dtype)
            else:
                init_ctx = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(init_ctx, std=0.02)
            prompt_prefix = ' '.join(['X'] * n_ctx) + ' '

        self.ctx = nn.Parameter(init_ctx)  # to be optimized

        name_lens = [len(tokenizer.encode(prompt)) for prompt in prompt_pairs]
        prompts = [prompt_prefix + prompt for prompt in prompt_pairs]
        with torch.no_grad():
            self.tokenized_prompts = clip.tokenize(prompts)
            init_embedding = clip_model.token_embedding(self.tokenized_prompts).type(
                dtype
            )

        self.n_ctx = n_ctx
        self.n_cls = len(prompt_pairs)
        self.name_lens = name_lens

        self.register_buffer('token_prefix', init_embedding[:, :1, :])  # SOS
        self.register_buffer(
            'token_suffix', init_embedding[:, 1 + n_ctx :, :]
        )  # CLS, EOS

    def get_prompts_with_middle_class(self):
        """
        Get prompts with the original text embedding inserted in the middle.

        Returns:
            torch.Tensor: The generated prompts.
        """
        ctx = self.ctx.to(self.token_prefix)
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        half_n_ctx = self.n_ctx // 2
        prompts = []
        for i in range(self.n_cls):
            name_len = self.name_lens[i]
            prefix_i = self.token_prefix[i : i + 1, :, :]
            class_i = self.token_suffix[i : i + 1, :name_len, :]
            suffix_i = self.token_suffix[i : i + 1, name_len:, :]
            ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
            ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
            prompt = torch.cat(
                [
                    prefix_i,  # (1, 1, dim)
                    ctx_i_half1,  # (1, n_ctx//2, dim)
                    class_i,  # (1, name_len, dim)
                    ctx_i_half2,  # (1, n_ctx//2, dim)
                    suffix_i,  # (1, *, dim)
                ],
                dim=1,
            )
            prompts.append(prompt)
        prompts = torch.cat(prompts, dim=0)
        return prompts

    def forward(self, clip_model):
        """
        Forward pass for the PromptLearner.

        Args:
            clip_model (nn.Module): The CLIP model.

        Returns:
            torch.Tensor: The output features.
        """
        prompts = self.get_prompts_with_middle_class()
        x = prompts + clip_model.positional_embedding.type(clip_model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = clip_model.ln_final(x).type(clip_model.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = (
            x[torch.arange(x.shape[0]), self.tokenized_prompts.argmax(dim=-1)]
            @ clip_model.text_projection
        )

        return x


@ARCH_REGISTRY.register()
class QualiCLIP(nn.Module):
    """
    QualiCLIP model implementation following the original paper. QualiCLIP+ represents the version that employs prompt
    learning, similar to CLIP-IQA+ (https://arxiv.org/abs/2207.12396).
    """

    def __init__(
        self,
        model_type='qualiclip+',
        backbone='RN50',
        temperature=2,
        n_ctx=16,
        ctx_init='',
        prompt_specific_ctx=True,
        pretrained=True,
        pos_embedding=False,
    ) -> None:
        super().__init__()

        self.clip_model = [load(backbone, 'cpu')]  # avoid saving clip weights

        # antonym prompts used during training
        self.prompt_pairs = [
            'Good photo.',
            'Bad photo.',
            'Sharp image.',
            'Blurry image.',
            'Sharp edges.',
            'Blurry edges.',
            'High-resolution image.',
            'Low-resolution image.',
            'Noise-free image.',
            'Noisy image.',
            'High-quality image.',
            'Low-quality image.',
            'Good picture.',
            'Bad picture.',
        ]

        self.model_type = model_type
        self.temperature = temperature
        self.pos_embedding = pos_embedding
        if 'qualiclip+' in model_type:
            self.prompt_learner = PromptLearner(
                self.clip_model[0],
                self.prompt_pairs,
                n_ctx=n_ctx,
                ctx_init=ctx_init,
                prompt_specific_ctx=prompt_specific_ctx,
            )

        self.default_mean = torch.Tensor(OPENAI_CLIP_MEAN).view(1, 3, 1, 1)
        self.default_std = torch.Tensor(OPENAI_CLIP_STD).view(1, 3, 1, 1)

        checkpoint = torch.load(
            load_file_from_url(default_model_urls['qualiclip']),
            map_location='cpu',
            weights_only=True,
        )
        self.prompts_features = checkpoint[
            'prompts_features'
        ]  # Load the pre-computed normalized text features of the prompts
        del checkpoint['prompts_features']
        checkpoint = {k.replace('clip_model.', ''): v for k, v in checkpoint.items()}
        self.clip_model[0].load_state_dict(checkpoint)

        if pretrained and 'qualiclip+' in model_type:
            assert backbone == 'RN50', 'Only RN50 backbone is supported for QualiCLIP+'
            if model_type in default_model_urls.keys():
                load_pretrained_network(
                    self, default_model_urls[model_type], True, 'params'
                )
            else:
                raise ValueError(f'No pretrained model for {model_type}')

        for p in self.clip_model[0].parameters():
            p.requires_grad = False

    def forward(self, x):
        # preprocess image
        x = (x - self.default_mean.to(x)) / self.default_std.to(x)
        clip_model = self.clip_model[0].to(x)

        # get text features
        if self.model_type == 'qualiclip':
            self.prompts_features = self.prompts_features.to(x.device)
            text_features = self.prompts_features
        elif 'qualiclip+' in self.model_type:
            text_features = self.prompt_learner(clip_model)
        else:
            raise ValueError(f'Invalid model type: {self.model_type}')

        # compute logits
        logits, _ = clip_model(
            x, None, text_features=text_features, pos_embedding=self.pos_embedding
        )
        logits = logits.reshape(logits.shape[0], -1, 2)
        exp_logits = torch.exp(logits / self.temperature)
        probs = exp_logits / exp_logits.sum(dim=-1, keepdim=True)

        return probs[..., 0].mean(dim=1, keepdim=True)
