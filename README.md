# IQA-Toolbox-Python
An IQA toolbox with pure python.

**Please follow the [contribution instructions](Instruction.md) to make contributions to this repository.**

## [**TODO**] Introduction

This is a image quality assessment toolbox with pure python, supporting many mainstream full reference and no reference metrics.

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

#### Test single image
```
python inference_iqa.py --input_mode image --metric_name CKDN --image_path /path/to/image 
```

#### Test with directory
```
python inference_iqa.py --input_mode dir --metric_name CKDN --input_dir ./CalibraTest/dist_dir --ref_dir ./CalibraTest/ref_dir --save_file ./CalibraTest/ckdn_imported_result.txt 
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

### [**TODO**] Train 

## [**TODO**] Benchmark Performances and Model Zoo

Please refer to [Awesome-Image-Quality-Assessment](https://github.com/chaofengc/Awesome-Image-Quality-Assessment) for a comprehensive summary. 
<details open>
<summary>Supported methods (FR):</summary>

- [ ] LPIPS 
- [ ] DISTS

</details>

<details open>
<summary>Supported methods (NR):</summary>

- [x] MUSIQ

</details>

**TODO** Please refer to the [benchmark calibration]() to verify the correctness of imported codes and model weights, and the python implementations compared with original matlab scripts.

### Performances of the retrained models

| Methods | Dataset | Kon10k | CLIVE | SPAQ | AVA | Link(pth) |
| --- | --- | --- | --- | --- | --- | --- |

## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

## Citation

## Acknowledgement

The code architecture is borrowed from [BasicSR](https://github.com/xinntao/BasicSR). Several implementations are borrowed from [IQA-optimization](https://github.com/dingkeyan93/IQA-optimization) and [Image-Quality-Assessment-Toolbox](https://github.com/RyanXingQL/Image-Quality-Assessment-Toolbox). We also thanks the following works to make their codes public:
- **TODO**
- [DBCNN]() 
- [MUSIQ]() 

## Contact

If you have any questions, please email `chaofenghust@gmail.com`
