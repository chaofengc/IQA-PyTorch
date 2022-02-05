# Python Toolbox for Image Quality Assessment
An IQA toolbox with pure python and pytorch.


## Introduction

This is a image quality assessment toolbox with pure python and pytorch. We provide the following features:

- **Comprehensive.** Support many mainstream full reference (FR) and no reference (NR) metrics
- **Flexible.** Support training new DNN models with several public IQA datasets
- **Differentiable.** Most methods support pytorch backward
- **Convenient.** Quick inference and benchmark script
- **Accurate.** Results calibration of our implementation with original matlab scripts (if exist)

Please refer to [Awesome-Image-Quality-Assessment](https://github.com/chaofengc/Awesome-Image-Quality-Assessment) for a comprehensive summary of IQA methods, as well as download links for IQA datasets. Below are details of supported methods and datasets in this project. 

<details open>
<summary>Supported methods and datasets:</summary>

<table>
<tr><td>

| Method | Type | Backward | 
| --- | --- | --- | 
| LPIPS | FR | :white_check_mark: |  NR |
| DISTS | FR | :white_check_mark: |
| CKDN | DR<sup>[1](#fn1)</sup> | :white_check_mark: | 
| FSIM | FR | :white_check_mark: | 
| SSIM  | FR | :white_check_mark: | 
| MS-SSIM  | FR | :white_check_mark: | 
| CW-SSIM  | FR | :white_check_mark: | 
| PSNR  | FR | :white_check_mark: | 
| VIF | FR | :white_check_mark: | 
| GMSD  | FR | :white_check_mark: | 
| NLPD  | FR | :white_check_mark: | 
| VSI  | FR | :white_check_mark: | 
| MAD  | FR | :white_check_mark: | 

</td><td>

| Method | Type | Backward | 
| --- | --- | --- | 
| MUSIQ :hourglass_flowing_sand: | NR | | 
| DBCNN | NR | :white_check_mark: | 
| PaQ-2-PiQ :hourglass_flowing_sand: | NR | | 
| HyperIQA :hourglass_flowing_sand: | NR | | 
| NIMA :hourglass_flowing_sand: | NR | | 
| CNNIQA :hourglass_flowing_sand: | NR | | 
| NRQM(Ma) :hourglass_flowing_sand: | NR | | 
| HOSA :hourglass_flowing_sand: | NR | | 
| BRISQUE :hourglass_flowing_sand: | NR | | 
| NIQE :hourglass_flowing_sand: | NR | | 

</td><td>

| Dataset | Type |
| --- | --- |
| FLIVE(PaQ-2-PiQ) | NR | 
| SPAQ | NR/mobile | 
| AVA | NR/Aesthetic | 
| PIPAL | FR | 
| BAPPS | FR | 
| PieAPP | FR | 
| KADID-10k | FR | 
| KonIQ-10k(++) | NR | 
| LIVEChallenge | NR | 
| LIVEM | FR | 
| LIVE | FR | 
| TID2013 | FR | 
| TID2008 | FR | 
| CSIQ | FR | 

</td></tr> 
</table>

<a name="fn1">[1]</a> DR means distortion reference. Please refer to the paper for details. 

</details>

## Quick Start

### Dependencies and Installation
- Ubuntu >= 18.04
- Python >= 3.8
- Pytorch >= 1.8
- CUDA 11.0 (if use GPU)
- Other required packages in `requirements.txt`
```
# git clone this repository
git clone https://github.com/chaofengc/IQA-Toolbox-Python.git
cd IQA-Toolbox-Python
pip3 install -r requirements.txt
```

### Quick Inference

#### Test script 

Example test script with input directory and reference directory. Single image is also supported for `-i` and `-r` options. 
```
python inference_iqa.py -n LPIPS -i ./ResultsCalibra/dist_dir -r ./ResultsCalibra/ref_dir 
```

#### [**TODO**] Used as functions in your project

Metrics which allowed backward can be used for model optimization, such as image enhancement networks.

```
from pyiqa import LPIPS 

metric_func = LPIPS(net='alex', version='0.1').to(device)
# img_tensor_x/y: (N, 3, H, W)
# data format: RGB, 0 ~ 1
score = metric_func(img_tensor_x, img_tensor_y)
```

## Train 

### Dataset Preparation

- You only need to unzip downloaded datasets from official website without any extra operation. Download links are provided in [Awesome-Image-Quality-Assessment](https://github.com/chaofengc/Awesome-Image-Quality-Assessment).  
- We provide common interface to load these datasets with the prepared meta information files and train/val/test split files, which can be downloaded from [TODOdownload_link](). 

Details of the dataloader inferface and meta information files can be found in [Dataset Preparation](Dataset_Preparation.md) 

### Example Train Script 

Example to train DBCNN on LIVEChallenge dataset
```
# train for single experiment
python pyiqa/train.py -opt options/train/train_DBCNN.yml 

# train N splits for small datasets
python pyiqa/train_nsplits.py -opt options/train/train_DBCNN.yml 
```

## Benchmark Performances and Model Zoo

Please refer to the [results calibration](./ResultsCalibra/ResultsCalibra.md) to verify the correctness of imported codes and model weights, and the python implementations compared with original matlab scripts.

Here is an example script to get performance benchmark on different datasets:
```
python benchmark_results.py -opt options/benchmark_test.yml
```
And refer to [benchmark results](tests/benchmark_results.csv) for results.

### Performances of the retrained deep learning models

#### Small datasets, n-splits validation 

| Methods | CSIQ | TID2008 | TID2013 | LIVE | LIVEM | LIVEC | 
| --- | --- | --- | --- | --- | --- | --- | 
| DBCNN | | | | | | | 

#### Large dataset performance

<!-- | Methods | Dataset | Kon10k | LIVEC | SPAQ | AVA | Link(pth) |
| --- | --- | --- | --- | --- | --- | --- | -->

## Contribution

Any contributions to this repository are greatly appreciated. Please follow the [contribution instructions](Instruction.md) for contribution guidance.  

## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

<!-- ## Citation

```
TODO -->
<!-- ``` -->

## Acknowledgement

The code architecture is borrowed from [BasicSR](https://github.com/xinntao/BasicSR). Several implementations are taken from 

- [IQA-optimization](https://github.com/dingkeyan93/IQA-optimization)  
- [Image-Quality-Assessment-Toolbox](https://github.com/RyanXingQL/Image-Quality-Assessment-Toolbox) 
- [piq](https://github.com/photosynthesis-team/piq)
- [piqa](https://github.com/francois-rozet/piqa)

We also thanks the following works to make their codes public:
- **TODO**
- [DBCNN]() 
- [MUSIQ]() 

## Contact

If you have any questions, please email `chaofenghust@gmail.com`
