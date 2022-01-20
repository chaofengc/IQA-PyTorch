# Results Calibration

We random select 5 pairs of images from TID2013 for results calibration. Images are stored under `./dist_dir` and `./ref_dir`. Results of different metrics are saved under `./results_compare/`. We also record the problems encountered during our reproduction of matlab scripts in [MatlabReproduceNote](./MatlabReproduceNote.md)

| Method | I03.bmp | I04.bmp | I06.bmp | I08.bmp | I19.bmp |
| --- | --- | --- | --- | --- | --- |
| CKDN<sup>[1](#fn1)</sup>(org) | 0.2833 | 0.5766 | 0.6367 | 0.6579 | 0.5999 |
| CKDN(ours imported) | 0.2833 | 0.5766 | 0.6367 | 0.6579 | 0.5999 |
| LPIPS(org) | 0.7237 | 0.2572 | 0.0508 | 0.0521 | 0.4253 |
| LPIPS(ours imported) | 0.7237 | 0.2572 | 0.0508 | 0.0521 | 0.4253 |
| DISTS(org) | 0.4742 | 0.1424 | 0.0682 | 0.0287 | 0.3123 |
| DISTS(ours imported) | 0.4742 | 0.1424 | 0.0682 | 0.0287 | 0.3123 |
| SSIM<sup>[2](#fn2)</sup>(org) | 0.7326 | 0.9989 | 0.9995 | 0.9674 | 0.6790 |
| SSIM(ours imported) | 0.7328 | 0.9989 | 0.9995 | 0.9676 | 0.6791 |
| MS-SSIM<sup>[3](#fn3)</sup>(org) | 0.6981 | 0.9998 | 0.9999 | 0.9570 | 0.8547 |
| MS-SSIM(ours imported) | 0.6984 | 0.9998 | 0.9999 | 0.9571 | 0.8547 |
| PSNR<sup>[4](#fn4)</sup>(org) | 21.11 | 20.99 | 27.01 | 23.30 | 21.62 |
| PSNR(ours imported) | 21.11 | 20.99 | 27.01 | 23.30 | 21.62 |
| FSIM(org) | 0.6890 | 0.9702 | 0.9927 | 0.9575 | 0.8220 |
| FSIM(ours imported) | 0.6891 | 0.9702 | 0.9927 | 0.9575 | 0.8220 |
| VIF<sup>[5](#fn5)</sup>(org) | 0.0180 | 0.9960 | 0.9978 | 0.9111 | 0.1881 |
| VIF(ours imported) | 0.0180 | 0.9960 | 0.9978 | 0.9111 | 0.1881 |
| GMSD<sup>[6](#fn6)</sup>(org) | 0.2120 | 0.0002 | 0.0002 | 0.1317 | 0.1865 |
| GMSD(ours imported) | 0.2120 | 0.0002 | 0.0002 | 0.1317 | 0.1865 |

#### Notice
<a name="fn1">[1]</a> CKDN used degraded images as references in the original paper.   
<a name="fn2">[2]</a> The original SSIM matlab script downsample the image when larger than 256. We remove such constraint.   
<a name="fn3">[3]</a> We use Y-channel of YCBCR images as input of original MS-SSIM matlab script.  
<a name="fn4">[4]</a> The original PSNR code refers to scikit-learn package with RGB 3-channel calculation (from skimage.metrics import peak_signal_noise_ratio).  
<a name="fn5">[5]</a> We use Y-channel of YCBCR images as input of original VIF matlab script.  
<a name="fn6">[6]</a> We use Y-channel of YCBCR images as input of original GMSD matlab script.
