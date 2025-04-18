from pyiqa.data import build_dataset, build_dataloader
import torch
import pytest
import os
import yaml

from pyiqa import load_dataset


with open('./pyiqa/default_dataset_configs.yml') as f:
    options = yaml.safe_load(f)

common_opt = {
    'name': 'test',
    'augment': {'resize': [224, 224]},
    'batch_size_per_gpu': 32,
    'num_worker_per_gpu': 8,
    'phase': 'train',
}

@pytest.mark.parametrize(
    ('test_dataset_name'),
    [
        (k)
        for k in options.keys()
        if os.path.exists(options[k]['dataroot_target'])
            and os.path.exists(options[k]['meta_info_file'])
    ],
)
def test_datasets_loading(test_dataset_name):
    dataset_opt = options[test_dataset_name]
    dataset_opt.update(common_opt)
    dataset = build_dataset(dataset_opt)
    dataloader = build_dataloader(dataset, dataset_opt)

    num_batches = 0
    max_num_test = 10
    for data in dataloader:
        if 'img' in data.keys():
            img_tensor = data['img']
            assert img_tensor.shape[1:] == torch.Size([3, 224, 224]), (
                f'input image shape should be [3, 224, 224], but got {img_tensor.shape[1:]}'
            )
        elif 'ref_img' in data.keys():
            ref_tensor = data['ref_img']
            assert ref_tensor.shape[1:] == torch.Size([3, 224, 224]), (
                f'reference image shape should be [3, 224, 224], but got {ref_tensor.shape[1:]}'
            )
        elif 'distA' in data.keys():
            ref_tensor = data['distA_img']
            assert ref_tensor.shape[1:] == torch.Size([3, 224, 224]), (
                f'reference image shape should be [3, 224, 224], but got {ref_tensor.shape[1:]}'
            )
        elif 'distB' in data.keys():
            ref_tensor = data['distB_img']
            assert ref_tensor.shape[1:] == torch.Size([3, 224, 224]), (
                f'reference image shape should be [3, 224, 224], but got {ref_tensor.shape[1:]}'
            )
        num_batches += 1
        if num_batches > max_num_test:
            break


@pytest.mark.parametrize(
    ('test_dataset_name'),
    [
        (k)
        for k in options.keys()
        if options[k]['type'] == 'GeneralFRDataset'
    ],
)
def test_fr_datasets_loading(test_dataset_name):
    idx0, idx1 = 0, 1
    if test_dataset_name == 'live':
        idx0, idx1 = -1, -2

    frdata = load_dataset(test_dataset_name)

    assert not torch.allclose(frdata[idx0]['img'], frdata[idx1]['img']), f'distortion images of the same image should be different! {test_dataset_name}'
    assert torch.allclose(frdata[idx0]['ref_img'], frdata[idx1]['ref_img']), f'reference images of the same image should be the same! {test_dataset_name}'