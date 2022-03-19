r"""Convert tensorflow weights of MUSIQ to pytorch

Reference: https://github.com/google-research/google-research/blob/master/musiq/run_predict_image.py

"""

import numpy as np
import torch
from pyiqa.archs.musiq_arch import MUSIQ


def check_same(x, y):
    return np.abs(y - x).mean() < np.abs(x.min())


# ckpt_path = './tmp/MUSIQ/musiq_koniq_ckpt.npz'
# save_path = './experiments/pretrained_models/MUSIQ/musiq_koniq_ckpt.pth'
ckpt_path = './tmp/MUSIQ/musiq_ava_ckpt.npz'
save_path = './experiments/pretrained_models/MUSIQ/musiq_ava_ckpt.pth'
# ckpt_path = './tmp/MUSIQ/musiq_paq2piq_ckpt.npz'
# save_path = './experiments/pretrained_models/MUSIQ/musiq_paq2piq_ckpt.pth'
# ckpt_path = './tmp/MUSIQ/musiq_spaq_ckpt.npz'
# save_path = './experiments/pretrained_models/MUSIQ/musiq_spaq_ckpt.pth'
# ckpt_path = './tmp/MUSIQ/musiq_imagenet_pretrain.npz'
# save_path = './experiments/pretrained_models/MUSIQ/musiq_imagenet_pretrain.pth'
tf_params = np.load(ckpt_path)
tf_keys = [k for k in tf_params.keys() if 'target' in k]

if 'ava' in ckpt_path:
    musiq_model = MUSIQ(num_class=10, pretrained=False)
else:
    musiq_model = MUSIQ(pretrained=False)
th_params = musiq_model.state_dict()
tf_params = np.load(ckpt_path)

tf_keys = [k for k in tf_params.keys() if 'target' in k]
th_keys = th_params.keys()


class TmpHead(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.imagenet_head = torch.nn.Linear(384, 1000)


if 'imagenet' in ckpt_path:
    th_params.update(TmpHead().state_dict())

total_converted_params = 0


def convert_module(tf_same_key_strs, th_same_key_strs=None):
    global total_converted_params
    """assign module with the same keywords"""
    if th_same_key_strs is None:
        th_same_key_strs = tf_same_key_strs

    tf_filter_keys = []
    th_filter_keys = []
    for tfk in tf_keys:
        keep_flag = True
        for sk in tf_same_key_strs:
            if sk.lower() not in tfk.lower():
                keep_flag = False
        if keep_flag:
            tf_filter_keys.append(tfk)

    for thk in th_keys:
        keep_flag = True
        for sk in th_same_key_strs:
            if sk.lower() not in thk.lower():
                keep_flag = False
        if keep_flag:
            th_filter_keys.append(thk)

    assert len(tf_filter_keys) == len(
        th_filter_keys), f'{tf_filter_keys}, {th_filter_keys}, {len(tf_filter_keys)}, {len(th_filter_keys)}'
    for tfk, thk in zip(sorted(tf_filter_keys), sorted(th_filter_keys)):
        print(f'Assign {tfk} to {thk}')
        tfw = tf_params[tfk]
        thw = th_params[thk]
        if 'conv' in tfk:
            tfw = tfw.transpose(3, 2, 0, 1)
        elif 'key/' in tfk or 'value/' in tfk or 'query/' in tfk:
            if 'kernel' in tfk:
                tfw = tfw.transpose(1, 2, 0)
                tfw = tfw.reshape(384, 384)
            elif 'bias' in tfk:
                tfw = tfw.reshape(-1)
        elif 'out/' in tfk:
            if 'kernel' in tfk:
                tfw = tfw.transpose(2, 0, 1)
                tfw = tfw.reshape(384, 384)
            elif 'bias' in tfk:
                tfw = tfw.reshape(-1)
        elif 'bias' in tfk or 'scale' in tfk:
            if len(tfw.shape) > 1:
                tfw = tfw.squeeze()
        elif len(tfw.shape) == 2:
            tfw = tfw.transpose()
        assert tfw.shape == thw.shape, f'shape not match, {tfw.shape}, {thw.shape}'
        th_params[thk].copy_(torch.from_numpy(tfw))
        assert check_same(tfw, th_params[thk]), f'value not match'
        total_converted_params = total_converted_params + 1


# first 5 conv layers
convert_module(['root'])
convert_module(['block1'])

# fc layers
convert_module(['target/embedding'], ['embedding'])
if 'imagenet' in ckpt_path:
    convert_module(['target/head'], ['imagenet_head'])
else:
    convert_module(['target/head'], ['head'])

# transformer layers
convert_module(['posembed_input'])
convert_module(['scaleembed_input'])
convert_module(['encoder_norm'])
convert_module(['cls'])
convert_module(['encoderblock_', 'norm'])
convert_module(['encoderblock_', 'mlp'])
convert_module(['encoderblock_', 'attention'])

print(
    f'Model param num: {len(tf_keys)}/tensorflow, {len(tf_keys)}/pytorch. Converted param num: {total_converted_params}'
)
print(f'Save model to {save_path}')
torch.save(th_params, save_path)
