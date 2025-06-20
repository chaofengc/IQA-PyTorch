# <img align="left" width="100" height="100" src="docs/pyiqa_logo.jpg"> PyTorch Toolbox for Image Quality Assessment

An IQA toolbox with pure python and pytorch. Please refer to [Awesome-Image-Quality-Assessment](https://github.com/chaofengc/Awesome-Image-Quality-Assessment) for a comprehensive survey of IQA methods and download links for IQA datasets.

<a href="https://colab.research.google.com/drive/14J3KoyrjJ6R531DsdOy5Bza5xfeMODi6?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a> 
[![PyPI](https://img.shields.io/pypi/v/pyiqa)](https://pypi.org/project/pyiqa/)
[![Downloads](https://static.pepy.tech/badge/pyiqa)](https://pepy.tech/project/pyiqa)
[![Documentation Status](https://readthedocs.org/projects/iqa-pytorch/badge/?version=latest)](https://iqa-pytorch.readthedocs.io/en/latest/?badge=latest)
[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/chaofengc/Awesome-Image-Quality-Assessment)
[![Citation](https://img.shields.io/badge/Citation-bibtex-green)](https://github.com/chaofengc/IQA-PyTorch/blob/main/README.md#bookmark_tabs-citation)
[![Zhihu](https://img.shields.io/badge/zhihu-white?logo=zhihu)](https://www.zhihu.com/column/c_1565424954811846656)

<!-- [![visitors](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fchaofengc%2FIQA-PyTorch&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=visitors&edge_flat=false)](https://hits.seeyoufarm.com) -->

## :open_book: Introduction

This is a comprehensive image quality assessment (IQA) toolbox built with **pure Python and PyTorch**. We provide reimplementation of many widely used full reference (FR) and no reference (NR) metrics, **with results calibrated against official MATLAB scripts when available**. With GPU acceleration, **our implementations are much faster than their Matlab counterparts.** Please refer to the following documents for details:  

<div align="center">

📦 [Model Cards](docs/ModelCard.md)  |  🗃️ [Dataset Cards](docs/Dataset_Preparation.md) | 🤗 [Datasets Download](https://huggingface.co/datasets/chaofengc/IQA-Toolbox-Datasets/tree/main) | 📚 [Documentation](https://iqa-pytorch.readthedocs.io/en/latest/) | 📈[Benchmark](https://github.com/chaofengc/IQA-PyTorch/tree/main?tab=readme-ov-file#performance-evaluation-protocol)

</div>

---

### :triangular_flag_on_post: Updates/Changelog
- 🎉**Jun, 2025**. Add `sfid`, a commonly used metric in generative models.
- 🎆**Jan, 2025**. Add `qualiclip`, `qualiclip+` and its variances trained on different datasets, refer to official repo [here](https://github.com/miccunifi/QualiCLIP). Thanks for the contribution from [Lorenzo Agnolucci](https://github.com/LorenzoAgnolucci) 🤗.
- 🪐**Dec, 2024**. Add `fid` with MMD distance, use it with `fid_metric(..., distance_type='mmd', kernel_type='rbf')`refer to [cmmd](https://github.com/google-research/google-research/tree/master/cmmd) for more details. Thanks to [Ina](https://github.com/Luciennnnnnn) for the contributions.
- 🌕**Dec, 2024**. Add `fid_dinov2`, refer to [dgm-eval](https://github.com/layer6ai-labs/dgm-eval) for more details. 
- 💫**Nov, 2024**. Add `pyiqa.load_dataset` for easy loading of several common datasets. 
- 🌟**Nov, 2024**. Add `compare2score` and `deepdc`. Thanks to [hanwei](https://github.com/h4nwei) for their great work 🤗, and please refer to their official papers for more details! 
- [**More**](docs/history_changelog.md)

---

## :zap: Quick Start

### Installation
```bash
# Install with pip
pip install pyiqa

# Install latest github version
pip uninstall pyiqa # if have older version installed already 
pip install git+https://github.com/chaofengc/IQA-PyTorch.git

# Install with git clone
git clone https://github.com/chaofengc/IQA-PyTorch.git
cd IQA-PyTorch
pip install -r requirements.txt
python setup.py develop
```

### Basic Usage 

You can simply use the package with commandline interface. 
```bash
# list all available metrics
pyiqa -ls

# test with default settings
pyiqa [metric_name(s)] -t [image_path or dir] -r [image_path or dir] --device [cuda or cpu] --verbose
```

### Advanced Usage with Codes

#### Test metrics 

```python
import pyiqa
import torch

# list all available metrics
print(pyiqa.list_models())

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# create metric with default setting
iqa_metric = pyiqa.create_metric('lpips', device=device)

# check if lower better or higher better
print(iqa_metric.lower_better)

# example for iqa score inference
# Tensor inputs, img_tensor_x/y: (N, 3, H, W), RGB, 0 ~ 1
score_fr = iqa_metric(img_tensor_x, img_tensor_y)

# img path as inputs.
score_fr = iqa_metric('./ResultsCalibra/dist_dir/I03.bmp', './ResultsCalibra/ref_dir/I03.bmp')

# For FID metric, use directory or precomputed statistics as inputs
# refer to clean-fid for more details: https://github.com/GaParmar/clean-fid
fid_metric = pyiqa.create_metric('fid')
score = fid_metric('./ResultsCalibra/dist_dir/', './ResultsCalibra/ref_dir')
score = fid_metric('./ResultsCalibra/dist_dir/', dataset_name="FFHQ", dataset_res=1024, dataset_split="trainval70k")
```

#### Use as loss functions

Note that gradient propagation is disabled by default. Set `as_loss=True` to enable it as a loss function. **Not all metrics support backpropagation, please refer to [Model Cards](docs/ModelCard.md) and be sure that you are using it in a `lower_better` way.**
```python
lpips_loss = pyiqa.create_metric('lpips', device=device, as_loss=True)

ssim_loss = pyiqa.create_metric('ssimc', device=device, as_loss=True)
loss = 1 - ssim_loss(img_tensor_x, img_tensor_y)   # ssim is not lower better
```

#### Use custom settings and weights 

We also provide a flexible way to use custom settings and weights in case you want to retrain or fine-tune the models. 

```python
iqa_metric = pyiqa.create_metric('topiq_nr', device=device, **custom_opts)

# Note that if you train the model with this package, the weights will be saved in weight_dict['params']. Otherwise, please set weight_keys=None.
iqa_metric.load_weights('path/to/weights.pth', weight_keys='params')
```

#### Example test script

Example test script with input directory/images and reference directory/images. 
```bash
# example for FR metric with dirs
python inference_iqa.py -m LPIPS[or lpips] -i ./ResultsCalibra/dist_dir[dist_img] -r ./ResultsCalibra/ref_dir[ref_img]

# example for NR metric with single image
python inference_iqa.py -m brisque -i ./ResultsCalibra/dist_dir/I03.bmp
```

#### Easy load of popular datasets

We offer an easy way to load popular IQA datasets through the configuration file `pyiqa/default_dataset_configs.yml`. The specified datasets will automatically download from the [huggingface IQA-PyTorch-Dataset](https://huggingface.co/datasets/chaofengc/IQA-PyTorch-Datasets). See example code below:
```python 
from pyiqa import get_dataset_info, load_dataset

# list all available datasets
print(get_dataset_info().keys())

# load dataset with default options and official split
dataset = load_dataset('koniq10k', data_root='./datasets', force_download=False, split_index='official_split', phase='test')
print(f'Loaded dataset, len={len(dataset)}, {dataset[0].keys()}')
print(dataset[0]['img'].shape)

# split_ratio: train/test/val
dataset = load_dataset('csiq', data_root='./datasets', force_download=False, split_index=1, split_ratio='622', phase='test')
print(f'Loaded dataset, len={len(dataset)}, {dataset[0].keys()}')
print(dataset[0]['img'].shape)

# or use dataset options
dataset_opts = {
  'split_index': 1,
  'split_ratio': '622',
  'phase': 'test',
  'augment': {
      'resize': 256,
      'center_crop': 224,
  }
}
dataset = load_dataset('csiq', data_root='./datasets', force_download=False, dataset_opts=dataset_opts)
print(f'Loaded dataset, len={len(dataset)}, {dataset[0].keys()}')
print(dataset[0]['img'].shape)
```
**Please refer to [Dataset Cards](docs/Dataset_Preparation.md) for more details about the `dataset_opts`.**


## :1st_place_medal: Benchmark Performances and Model Zoo

### Results Calibration

Please refer to the [results calibration](./ResultsCalibra/ResultsCalibra.md) to verify the correctness of the python implementations compared with official scripts in matlab or python.

### ⏬ Download Benchmark Datasets

For convenience, we upload all related datasets to [huggingface IQA-Toolbox-Dataset](https://huggingface.co/datasets/chaofengc/IQA-Toolbox-Datasets), and corresponding meta information files to [huggingface IQA-Toolbox-Dataset-metainfo](https://huggingface.co/datasets/chaofengc/IQA-Toolbox-Datasets-metainfo). 
Here are example codes to download them from huggingface:

>[!CAUTION]
> we only collect the datasets for academic, research, and educational purposes. It is important for the users to adhere to the usage guidelines, licensing terms, and conditions set forth by the original creators or owners of each dataset.

```python
import os
from huggingface_hub import snapshot_download

save_dir = './datasets'
os.makedirs(save_dir, exist_ok=True)

filename = "koniq10k.tgz"
snapshot_download("chaofengc/IQA-Toolbox-Datasets", repo_type="dataset", local_dir=save_dir, allow_patterns=filename, local_dir_use_symlinks=False)

os.system(f"tar -xzvf {save_dir}/{filename} -C {save_dir}")
```

Download meta information from Huggingface with `git clone` or update with `git pull`:
```
cd ./datasets
git clone https://huggingface.co/datasets/chaofengc/IQA-Toolbox-Datasets-metainfo meta_info

cd ./datasets/meta_info
git pull
```

Examples to specific dataset options can be found in `./pyiqa/default_dataset_configs.yml`. Details of the dataloader interface and meta information files can be found in [Dataset Preparation](docs/Dataset_Preparation.md)

### Performance Evaluation Protocol

**We use official models for evaluation if available.** Otherwise, we use the following settings to train and evaluate different models for simplicity and consistency:

| Metric Type   | Train     | Test                                       | Results                                                  | 
| ------------- | --------- | ------------------------------------------ | -------------------------------------------------------- |
| FR            | KADID-10k | CSIQ, LIVE, TID2008, TID2013               | [FR benchmark](tests/FR_benchmark_results.csv)   |
| NR            | KonIQ-10k | LIVEC, KonIQ-10k (official split), TID2013, SPAQ | [NR benchmark](tests/NR_benchmark_results.csv)   |
| Aesthetic IQA | AVA       | AVA (official split)                       | [IAA benchmark](tests/IAA_benchmark_results.csv) |
| Face IQA | [CGFIQA](https://github.com/DSL-FIQA/DSL-FIQA) | CGFIQA (official split)                    | [Face IQA benchmark](tests/Face_benchmark_results.csv) |
| Efficiency | CPU/GPU Time, GPU Memory | Average on $1080\times800$ image inputs | [Efficiency benchmark](tests/Efficiency_benchmark.csv) |

Results are calculated with:
- **PLCC without any correction**. Although test time value correction is common in IQA papers, we want to use the original value in our benchmark.
- **Full image single input.** We **do not** use multi-patch testing unless necessary.

Basically, we use the largest existing datasets for training, and cross dataset evaluation performance for fair comparison. The following models do not provide official weights, and are retrained by our scripts:

| Metric Type   | Reproduced Models |
| ------------- | ----------------------------- |
| FR            | `wadiqam_fr`  |
| NR            | `cnniqa`, `dbcnn`, `hyperiqa`,  `wadiqam_nr` |
| Aesthetic IQA | `nima`, `nima-vgg16-ava`      |

>[!NOTE]
>- Due to optimized training process, performance of some retrained approaches may be different with original paper.
>- Results of all **retrained models by ours** are normalized to [0, 1] and change to higher better for convenience.
>- Results of KonIQ-10k, AVA are both tested with official split.
>- NIMA is only applicable to AVA dataset now. We use `inception_resnet_v2` for default `nima`.
>- MUSIQ is not included in the IAA benchmark because we do not have train/split information of the official model.

### Benchmark Performance with Provided Script

Here is an example script to get performance benchmark on different datasets:
```bash
# NOTE: this script will test ALL specified metrics on ALL specified datasets
# Test default metrics on default datasets
python benchmark_results.py -m psnr ssim -d csiq tid2013 tid2008

# Test with your own options
python benchmark_results.py -m psnr --data_opt options/example_benchmark_data_opts.yml

python benchmark_results.py --metric_opt options/example_benchmark_metric_opts.yml tid2013 tid2008

python benchmark_results.py --metric_opt options/example_benchmark_metric_opts.yml --data_opt options/example_benchmark_data_opts.yml
```

## :hammer_and_wrench: Train



### Example Train Script

Example to train DBCNN on LIVEChallenge dataset
```bash
# train for single experiment
python pyiqa/train.py -opt options/train/DBCNN/train_DBCNN.yml

# train N splits for small datasets
python pyiqa/train_nsplits.py -opt options/train/DBCNN/train_DBCNN.yml
```

Example for distributed training
```bash
torchrun --nproc_per_node=2 --master_port=4321 pyiqa/train.py -opt options/train/CLIPIQA/train_CLIPIQA_koniq10k.yml --launcher pytorch
```

## :beers: Contribution

Any contributions to this repository are greatly appreciated. Please follow the [contribution instructions](docs/Instruction.md) for contribution guidance.

## :scroll: License

This work is licensed under a [NTU S-Lab License](https://github.com/chaofengc/IQA-PyTorch/blob/main/LICENSE_NTU-S-Lab) and <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a>

## :bookmark_tabs: Citation

If you find our codes helpful to your research, please consider to use the following citation:

```bib
@misc{pyiqa,
  title={{IQA-PyTorch}: PyTorch Toolbox for Image Quality Assessment},
  author={Chaofeng Chen and Jiadi Mo},
  year={2022},
  howpublished = "[Online]. Available: \url{https://github.com/chaofengc/IQA-PyTorch}"
}
```

Please also consider to cite our works on image quality assessment if it is useful to you:
```bib
@article{chen2024topiq,
  author={Chen, Chaofeng and Mo, Jiadi and Hou, Jingwen and Wu, Haoning and Liao, Liang and Sun, Wenxiu and Yan, Qiong and Lin, Weisi},
  title={TOPIQ: A Top-Down Approach From Semantics to Distortions for Image Quality Assessment}, 
  journal={IEEE Transactions on Image Processing}, 
  year={2024},
  volume={33},
  pages={2404-2418},
  doi={10.1109/TIP.2024.3378466}
}
``` 
```bib
@article{wu2024qalign,
  title={Q-Align: Teaching LMMs for Visual Scoring via Discrete Text-Defined Levels},
  author={Wu, Haoning and Zhang, Zicheng and Zhang, Weixia and Chen, Chaofeng and Li, Chunyi and Liao, Liang and Wang, Annan and Zhang, Erli and Sun, Wenxiu and Yan, Qiong and Min, Xiongkuo and Zhai, Guangtai and Lin, Weisi},
  journal={International Conference on Machine Learning (ICML)},
  year={2024},
  institution={Nanyang Technological University and Shanghai Jiao Tong University and Sensetime Research},
  note={Equal Contribution by Wu, Haoning and Zhang, Zicheng. Project Lead by Wu, Haoning. Corresponding Authors: Zhai, Guangtai and Lin, Weisi.}
}
```

## :heart: Acknowledgement

The code architecture is borrowed from [BasicSR](https://github.com/xinntao/BasicSR). Several implementations are taken from: [IQA-optimization](https://github.com/dingkeyan93/IQA-optimization), [Image-Quality-Assessment-Toolbox](https://github.com/RyanXingQL/Image-Quality-Assessment-Toolbox), [piq](https://github.com/photosynthesis-team/piq), [piqa](https://github.com/francois-rozet/piqa), [clean-fid](https://github.com/GaParmar/clean-fid)

We also thanks the following public repositories: [MUSIQ](https://github.com/google-research/google-research/tree/master/musiq), [DBCNN](https://github.com/zwx8981/DBCNN-PyTorch), [NIMA](https://github.com/kentsyx/Neural-IMage-Assessment), [HyperIQA](https://github.com/SSL92/hyperIQA), [CNNIQA](https://github.com/lidq92/CNNIQA), [WaDIQaM](https://github.com/lidq92/WaDIQaM), [PieAPP](https://github.com/prashnani/PerceptualImageError), [paq2piq](https://github.com/baidut/paq2piq), [MANIQA](https://github.com/IIGROUP/MANIQA) 

## :e-mail: Contact

If you have any questions, please email `chaofenghust@gmail.com`
