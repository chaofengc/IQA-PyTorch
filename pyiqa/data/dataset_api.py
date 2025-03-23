import os
import os.path as osp
from pathlib import Path
from tqdm import tqdm
import zipfile
import tarfile

from huggingface_hub import snapshot_download

from pyiqa import get_dataset_info
from pyiqa.data import build_dataset
from typing import Optional, Dict, Any


dataset_download_name = {
    'csiq': 'csiq.tgz',
    'tid2008': 'tid2008.tgz',
    'tid2013': 'tid2013.tgz',
    'live': 'live.tgz',
    'livem': 'livem.tgz',
    'livec': 'live_challenge.tgz',
    'koniq10k': 'koniq10k.tgz',
    'koniq10k-1024': 'koniq10k.tgz',
    'koniq10k++': 'koniq10k.tgz',
    'kadid10k': 'kadid10k.tgz',
    'spaq': 'spaq.tgz',
    'ava': 'ava.tgz',
    'pipal': 'pipal.tar',
    'flive': 'flive.tgz',
    'pieapp': 'pieapp.tgz',
    'bapps': 'bapps.tgz',
    'gfiqa': 'gfiqa-20k.tgz',
    'cgfiqa': 'CGFIQA.zip',
}


def extract_archive(archive_path: str, extract_path: Optional[str] = None) -> bool:
    """
    Extract .zip, .tar, or .tgz files with a progress bar.

    Args:
        archive_path (str): Path to the archive file
        extract_path (str, optional): Extraction destination. Defaults to current directory.

    Returns:
        bool: True if extraction successful, False otherwise
    """
    try:
        archive_path = Path(archive_path)
        if extract_path is None:
            extract_path = archive_path.parent

        # Create extraction directory if it doesn't exist
        os.makedirs(extract_path, exist_ok=True)

        # Handle ZIP files
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                # Get total size for progress bar
                total_size = sum(file.file_size for file in zip_ref.filelist)
                extracted_size = 0

                with tqdm(
                    total=total_size, unit='B', unit_scale=True, desc='Extracting ZIP'
                ) as pbar:
                    for file in zip_ref.filelist:
                        zip_ref.extract(file, extract_path)
                        extracted_size += file.file_size
                        pbar.update(file.file_size)

        # Handle TAR and TGZ files
        elif archive_path.suffix in ['.tar', '.tgz'] or archive_path.suffixes == [
            '.tar',
            '.gz',
        ]:
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                # Get total size for progress bar
                total_size = sum(member.size for member in tar_ref.getmembers())
                extracted_size = 0

                with tqdm(
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    desc='Extracting TAR/TGZ',
                ) as pbar:
                    for member in tar_ref.getmembers():
                        tar_ref.extract(member, extract_path)
                        extracted_size += member.size
                        pbar.update(member.size)

        else:
            raise ValueError(f'Unsupported archive format: {archive_path.suffix}')

        return True

    except Exception as e:
        print(f'Error extracting archive: {str(e)}')
        return False


def load_dataset(
    name: str,
    data_root: str = './datasets',
    force_download: bool = False,
    dataset_opts: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Any:
    """
    Load a dataset from specified location, downloading if necessary.

    Args:
        name (str): Name of the dataset to load
        data_root (str): Root directory for dataset storage
        force_download (bool): Whether to force download even if files exist
        dataset_opts (Optional[Dict[str, Any]]): Additional dataset options
        **kwargs (Any): Additional arguments passed to dataset configuration

    Returns:
        Any: Dataset object
    """
    print(f'Loading dataset {name} from {data_root} ...')

    # Get and update dataset configuration
    dataset_info = get_dataset_info(name)
    if dataset_opts is not None:
        dataset_info.update(dataset_opts or {})
    dataset_info.update(kwargs)

    # Update paths relative to data root
    dataset_info['dataroot_target'] = data_root / Path(
        *Path(dataset_info['dataroot_target']).parts[1:]
    )

    if 'dataroot_ref' in dataset_info:
        dataset_info['dataroot_ref'] = data_root / Path(
            *Path(dataset_info['dataroot_ref']).parts[1:]
        )

    dataset_info['meta_info_file'] = data_root / Path(
        *Path(dataset_info['meta_info_file']).parts[1:]
    )

    # Download metadata file if needed
    if not osp.exists(dataset_info['meta_info_file']):
        print(f'Downloading all metadata files to {data_root}/meta_info.')
        snapshot_download(
            'chaofengc/IQA-PyTorch-Datasets-metainfo',
            repo_type='dataset',
            local_dir=os.path.join(data_root, 'meta_info'),
        )

    # Check if dataset needs to be downloaded
    if force_download or not osp.exists(dataset_info['dataroot_target']):
        # Verify dataset availability
        assert name in dataset_download_name, (
            f'Dataset {name} is not available for download. '
            f'Currently available datasets are {list(get_dataset_info().keys())}'
        )

        # Download and extract dataset
        download_file_path = osp.join(data_root, dataset_download_name[name])
        print(f'Downloading dataset {name} to {download_file_path}')

        snapshot_download(
            'chaofengc/IQA-PyTorch-Datasets',
            repo_type='dataset',
            local_dir=data_root,
            allow_patterns=dataset_download_name[name],
        )

        extract_archive(download_file_path, data_root)

    # Build and return dataset
    dataset = build_dataset(dataset_info)
    return dataset
