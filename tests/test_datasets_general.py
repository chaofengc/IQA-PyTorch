from pyiqa.data import build_dataset, build_dataloader
import torch
import pytest
import os
import yaml

with open('./options/default_dataset_opt.yml') as f:
    options = yaml.safe_load(f)

common_opt = {
    'name': 'test',
    'augment': {
        'resize': [224, 224]
    },
    'batch_size_per_gpu': 32,
    'num_worker_per_gpu': 8,
    'phase': 'train',
}

@pytest.mark.parametrize(('test_dataset_name'), [(k) for k in options.keys() if os.path.exists(options[k]['dataroot_target'] and os.path.exists(options[k]['meta_info_file']))])
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
            assert img_tensor.shape[1:] == torch.Size(
                [3, 224, 224]), f'input image shape should be [3, 224, 224], but got {img_tensor.shape[1:]}'
        elif 'ref_img' in data.keys():
            ref_tensor = data['ref_img']
            assert ref_tensor.shape[1:] == torch.Size(
                [3, 224, 224]), f'reference image shape should be [3, 224, 224], but got {ref_tensor.shape[1:]}'
        elif 'distA' in data.keys():
            ref_tensor = data['distA_img']
            assert ref_tensor.shape[1:] == torch.Size(
                [3, 224, 224]), f'reference image shape should be [3, 224, 224], but got {ref_tensor.shape[1:]}'
        elif 'distB' in data.keys():
            ref_tensor = data['distB_img']
            assert ref_tensor.shape[1:] == torch.Size(
                [3, 224, 224]), f'reference image shape should be [3, 224, 224], but got {ref_tensor.shape[1:]}'
        num_batches += 1 
        if num_batches > max_num_test:
            break

