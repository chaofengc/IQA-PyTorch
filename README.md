# Python Toolbox for Image Quality Assessment
An IQA toolbox with pure python and pytorch.

**Please follow the [contribution instructions](Instruction.md) to make contributions to this repository.**

## [**TODO**] Introduction

This is a image quality assessment toolbox with pure python, supporting many mainstream full reference (FR) and no reference (NR) metrics. We also support training new DNN models with several public IQA datasets.

Please refer to [Awesome-Image-Quality-Assessment](https://github.com/chaofengc/Awesome-Image-Quality-Assessment) for a comprehensive summary as well as download links for IQA datasets. 
<details close>
<summary>Supported methods (FR):</summary>

- [x] LPIPS 
- [x] DISTS
- [x] FSIM 
- [x] SSIM 
- [x] PSNR 

</details>

<details close>
<summary>Supported methods (NR):</summary>

- [ ] MUSIQ

</details>

<details close>
<summary>Supported datasets:</summary>

- [x] LIVEChallenge 

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

#### [**TODO**] Use as function in your project
PyTorch backward is allowed for the following metrics: PSNR, LPIPS,  

```
from pyiqa import LPIPS 

metric_func = LPIPS(net='alex', version='0.1').to(device)
# img_tensor_x/y: (N, 3, H, W)
# data format: RGB, 0 ~ 1
score = metric_func(img_tensor_x, img_tensor_y)
```

### Train 

#### NR model

Example to train DBCNN on LIVEChallenge dataset
```
# train for single experiment
python pyiqa/train.py -opt options/train/DBCNN/train_DBCNN.yml 

# train N splits for small datasets
python pyiqa/train_nsplits.py -opt options/train/DBCNN/train_DBCNN.yml 
```

[**TODO**]
- [ ] Add more examples


## [**TODO**] Benchmark Performances and Model Zoo

**TODO** Please refer to the [results calibration](ResultsCalib.md) to verify the correctness of imported codes and model weights, and the python implementations compared with original matlab scripts.

### Performances of the retrained models

| Methods | Dataset | Kon10k | CLIVE | SPAQ | AVA | Link(pth) |
| --- | --- | --- | --- | --- | --- | --- |

## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

## Citation

```
TODO
```

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
