from pyiqa.data import build_dataset, build_dataloader
import torch
from tqdm import tqdm

options = {
    'BAPPS': {
        'type': 'BAPPSDataset',
        'dataroot_target': './datasets/PerceptualSimilarity/dataset',
        'meta_info_file': './datasets/meta_info/meta_info_BAPPSDataset.csv',
    },
    'PieAPP': {
        'type': 'PieAPPDataset',
        'dataroot_target': './datasets/PieAPP_dataset_CVPR_2018/',
        'meta_info_file': './datasets/meta_info/meta_info_PieAPPDataset.csv',
    },
    'FLIVE': {
        'type': 'GeneralNRDataset',
        'dataroot_target': './datasets/FLIVE_Database/database',
        'meta_info_file': './datasets/meta_info/meta_info_FLIVEDataset.csv',
    },
    'PIPAL': {
        'type': 'GeneralFRDataset',
        'dataroot_target': './datasets/PIPAL/Dist_Imgs',
        'dataroot_ref': './datasets/PIPAL/Train_Ref',
        'meta_info_file': './datasets/meta_info/meta_info_PIPALDataset.csv',
    },
    'KonIQ10k++': {
        'type': 'GeneralNRDataset',
        'dataroot_target': './datasets/koniq10k/512x384',
        'meta_info_file': './datasets/meta_info/meta_info_KonIQ10k++Dataset.csv',
    },
    'AVA': {
        'type': 'AVADataset',
        'dataroot_target': './datasets/AVA_dataset/ava_images/',
        'meta_info_file': './datasets/meta_info/meta_info_AVADataset.csv',
    },
    'SPAQ': {
        'type': 'GeneralNRDataset',
        'dataroot_target': './datasets/SPAQ/TestImage',
        'meta_info_file': './datasets/meta_info/meta_info_SPAQDataset.csv',
    },
    'KADID10k': {
        'type': 'GeneralFRDataset',
        'dataroot_target': './datasets/kadid10k/images',
        'meta_info_file': './datasets/meta_info/meta_info_KADID10kDataset.csv',
    },
    'KonIQ10k': {
        'type': 'GeneralNRDataset',
        'dataroot_target': './datasets/koniq10k/512x384',
        'meta_info_file': './datasets/meta_info/meta_info_KonIQ10kDataset.csv',
    },
    'LIVEC': {
        'type': 'LIVEChallengeDataset',
        'dataroot_target': './datasets/LIVEC',
        'meta_info_file': './datasets/meta_info/meta_info_LIVEChallengeDataset.csv',
    },
    'LIVEM': {
        'type': 'GeneralFRDataset',
        'dataroot_target': './datasets/LIVEmultidistortiondatabase',
        'meta_info_file': './datasets/meta_info/meta_info_LIVEMDDataset.csv',
    },
    'LIVE': {
        'type': 'GeneralFRDataset',
        'dataroot_target': './datasets/LIVEIQA_release2',
        'meta_info_file': './datasets/meta_info/meta_info_LIVEIQADataset.csv',
    },
    'TID2013': {
        'type': 'GeneralFRDataset',
        'dataroot_target': './datasets/tid2013/distorted_images',
        'dataroot_ref': './datasets/tid2013/reference_images',
        'meta_info_file': './datasets/meta_info/meta_info_TID2013Dataset.csv',
    },
    'TID2008': {
        'type': 'GeneralFRDataset',
        'dataroot_target': './datasets/tid2008/distorted_images',
        'dataroot_ref': './datasets/tid2008/reference_images',
        'meta_info_file': './datasets/meta_info/meta_info_TID2008Dataset.csv',
    },
    'CSIQ': {
        'type': 'GeneralFRDataset',
        'dataroot_target': './datasets/CSIQ/dst_imgs',
        'dataroot_ref': './datasets/CSIQ/src_imgs',
        'meta_info_file': './datasets/meta_info/meta_info_CSIQDataset.csv',
    },
}

common_opt = {
    'name': 'test',
    'augment': {
        'resize': [224, 224]
    },
    'batch_size_per_gpu': 32,
    'num_worker_per_gpu': 8,
    'phase': 'train',
}


def test(test_dataset_name):
    print(f'========>>> Test dataset reader: {test_dataset_name}')
    assert test_dataset_name in options.keys()
    dataset_opt = options[test_dataset_name]
    dataset_opt.update(common_opt)
    dataset = build_dataset(dataset_opt)
    dataloader = build_dataloader(dataset, dataset_opt)

    for data in tqdm(dataloader):
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


if __name__ == '__main__':
    test('CSIQ')
    test('TID2008')
    test('TID2013')
    test('LIVE')
    test('LIVEM')
    test('LIVEC')
    test('KonIQ10k')
    test('KADID10k')
    test('SPAQ')
    test('AVA')
    test('KonIQ10k++')
    test('PIPAL')
    test('FLIVE')
    test('PieAPP')
    test('BAPPS')
