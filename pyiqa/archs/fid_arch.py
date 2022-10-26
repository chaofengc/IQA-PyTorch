"""FID and clean-fid metric

Codes are borrowed from the clean-fid project:
    - https://github.com/GaParmar/clean-fid

Ref:
    [1] GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium. 
    Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, Sepp Hochreiter
    NeurIPS, 2017
    [2] On Aliased Resizing and Surprising Subtleties in GAN Evaluation
    Gaurav Parmar, Richard Zhang, Jun-Yan Zhu
    CVPR, 2022
"""

from email.policy import default
import os
from tqdm import tqdm
from glob import glob
import numpy as np
from scipy import linalg
from PIL import Image

import torch
from torch import nn
import torchvision

from .inception import InceptionV3
from pyiqa.utils.download_util import load_file_from_url
from pyiqa.utils.img_util import is_image_file
from pyiqa.utils.registry import ARCH_REGISTRY

default_model_urls = {
    'ffhq_clean_trainval70k_512.npz': 'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/ffhq_clean_trainval70k_512.npz',
    'ffhq_clean_trainval70k_512_kid.npz': 'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/ffhq_clean_trainval70k_512_kid.npz',
}


class ResizeDataset(torch.utils.data.Dataset):
    """
    A placeholder Dataset that enables parallelizing the resize operation
    using multiple CPU cores
    files: list of all files in the folder
    mode:
        - clean: use PIL resize before calculate features
        - legacy_pytorch: do not resize here, but before pytorch model
    """

    def __init__(self, files, mode, size=(299, 299)):
        self.files = files
        self.transforms = torchvision.transforms.ToTensor()
        self.size = size
        self.mode = mode

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = str(self.files[i])
        img_pil = Image.open(path).convert('RGB')

        if self.mode == 'clean':
            def resize_single_channel(x_np):
                img = Image.fromarray(x_np.astype(np.float32), mode='F')
                img = img.resize(self.size, resample=Image.BICUBIC)
                return np.asarray(img).clip(0, 255).reshape(*self.size, 1)

            img_np = np.array(img_pil)
            img_np = [resize_single_channel(img_np[:, :, idx]) for idx in range(3)]
            img_np = np.concatenate(img_np, axis=2).astype(np.float32)
            img_np = (img_np - 128) / 128
            img_t = torch.tensor(img_np).permute(2, 0, 1)
        else:
            img_np = np.array(img_pil).clip(0, 255)
            img_t = self.transforms(img_np)

        return img_t


def get_reference_statistics(name, res, mode="clean", split="test", metric="FID"):
    r"""
        Load precomputed reference statistics for commonly used datasets
    """
    base_url = "https://www.cs.cmu.edu/~clean-fid/stats"
    if split == "custom":
        res = "na"
    if metric == "FID":
        rel_path = (f"{name}_{mode}_{split}_{res}.npz").lower()
        url = f"{base_url}/{rel_path}"

        if rel_path in default_model_urls.keys():
            fpath = load_file_from_url(default_model_urls[rel_path])
        else:
            fpath = load_file_from_url(url)

        stats = np.load(fpath)
        mu, sigma = stats["mu"], stats["sigma"]
        return mu, sigma
    elif metric == "KID":
        rel_path = (f"{name}_{mode}_{split}_{res}_kid.npz").lower()
        url = f"{base_url}/{rel_path}"

        if rel_path in default_model_urls.keys():
            fpath = load_file_from_url(default_model_urls[rel_path])
        else:
            fpath = load_file_from_url(url)

        stats = np.load(fpath)
        return stats["feats"]


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Danica J. Sutherland.
    Params:
        mu1   : Numpy array containing the activations of a layer of the
                inception net (like returned by the function 'get_predictions')
                for generated samples.
        mu2   : The sample mean over activations, precalculated on an
                representative data set.
        sigma1: The covariance matrix over activations for generated samples.
        sigma2: The covariance matrix over activations, precalculated on an
                representative data set.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def kernel_distance(feats1, feats2, num_subsets=100, max_subset_size=1000):
    r"""
        Compute the KID score given the sets of features
    """
    n = feats1.shape[1]
    m = min(min(feats1.shape[0], feats2.shape[0]), max_subset_size)
    t = 0
    for _subset_idx in range(num_subsets):
        x = feats2[np.random.choice(feats2.shape[0], m, replace=False)]
        y = feats1[np.random.choice(feats1.shape[0], m, replace=False)]
        a = (x @ x.T / n + 1) ** 3 + (y @ y.T / n + 1) ** 3
        b = (x @ y.T / n + 1) ** 3
        t += (a.sum() - np.diag(a).sum()) / (m - 1) - b.sum() * 2 / m
    kid = t / num_subsets / m
    return float(kid)


def get_folder_features(fdir, model=None, num_workers=12,
                        batch_size=32,
                        device=torch.device("cuda"),
                        mode="clean",
                        description="",
                        verbose=True,
                        ):
    r"""
    Compute the inception features for a folder of image files
    """
    files = sorted([file for file in glob(os.path.join(fdir, '*')) if is_image_file(file)])

    if verbose:
        print(f"Found {len(files)} images in the folder {fdir}")

    dataset = ResizeDataset(files, mode=mode)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size, shuffle=False,
                                             drop_last=False, num_workers=num_workers)

    # collect all inception features
    if verbose:
        pbar = tqdm(dataloader, desc=description)
    else:
        pbar = dataloader

    if mode == 'clean':
        resize_input = normalize_input = False
    else:
        resize_input = normalize_input = True

    l_feats = []
    with torch.no_grad():
        for batch in pbar:
            feat = model(batch.to(device), resize_input, normalize_input)
            feat = feat[0].squeeze(-1).squeeze(-1).detach().cpu().numpy()
            l_feats.append(feat)
    np_feats = np.concatenate(l_feats)
    return np_feats


@ARCH_REGISTRY.register()
class FID(nn.Module):
    """FID and Clean-FID metric 
    Args:
        mode: [clean, legacy_pytorch]. Default: clean
    """

    def __init__(
        self,
        dims=2048,
    ) -> None:
        super().__init__()

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        self.model = InceptionV3(output_blocks=[block_idx])
        self.model.eval()

    def forward(self,
                fdir1=None,
                fdir2=None,
                mode='clean',
                dataset_name='FFHQ',
                dataset_res=1024,
                dataset_split='train',
                num_workers=12,
                batch_size=32,
                device=torch.device('cuda'),
                verbose=True,
                ):
        
        assert mode in ['clean', 'legacy_pytorch', 'legacy_tensorflow'], 'Invalid calculation mode, should be in [clean, legacy_pytorch, legacy_tensorflow]' 

        # if both dirs are specified, compute FID between folders
        if fdir1 is not None and fdir2 is not None:
            if not verbose:
                print("compute FID between two folders")
            fbname1 = os.path.basename(fdir1)
            np_feats1 = get_folder_features(fdir1, self.model, num_workers=num_workers, batch_size=batch_size,
                                            device=device, mode=mode, description=f"FID {fbname1}: ", verbose=verbose)

            fbname2 = os.path.basename(fdir2)
            np_feats2 = get_folder_features(fdir2, self.model, num_workers=num_workers, batch_size=batch_size,
                                            device=device, mode=mode, description=f"FID {fbname2}: ", verbose=verbose)

            mu1, sig1 = np.mean(np_feats1, axis=0), np.cov(np_feats1, rowvar=False)
            mu2, sig2 = np.mean(np_feats2, axis=0), np.cov(np_feats2, rowvar=False)
            return frechet_distance(mu1, sig1, mu2, sig2)

        # compute fid of a folder
        elif fdir1 is not None and fdir2 is None:
            if verbose:
                print(f"compute FID of a folder with {dataset_name}-{mode}-{dataset_split}-{dataset_res} statistics")
            fbname1 = os.path.basename(fdir1)
            np_feats1 = get_folder_features(fdir1, self.model, num_workers=num_workers, batch_size=batch_size,
                                            device=device, mode=mode, description=f"FID {fbname1}: ", verbose=verbose)

            # Load reference FID statistics (download if needed)
            ref_mu, ref_sigma = get_reference_statistics(
                dataset_name, dataset_res, mode=mode, split=dataset_split)

            mu1, sig1 = np.mean(np_feats1, axis=0), np.cov(np_feats1, rowvar=False)
            score = frechet_distance(mu1, sig1, ref_mu, ref_sigma)
            return score
        else:
            raise ValueError("invalid combination of arguments entered")
