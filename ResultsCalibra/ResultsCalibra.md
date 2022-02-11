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
| CW-SSIM<sup>[9](#fn9)</sup>(org) | 0.2764 | 0.9998 | 1.0000 |  0.9067 | 0.8659 |
| CW-SSIM(ours imported) | 	0.2783 | 0.9998 | 1.0000 | 0.9064 | 0.8648 |
| PSNR<sup>[4](#fn4)</sup>(org) | 21.11 | 20.99 | 27.01 | 23.30 | 21.62 |
| PSNR(ours imported) | 21.11 | 20.99 | 27.01 | 23.30 | 21.62 |
| FSIM(org) | 0.6890 | 0.9702 | 0.9927 | 0.9575 | 0.8220 |
| FSIM(ours imported) | 0.6891 | 0.9702 | 0.9927 | 0.9575 | 0.8220 |
| VIF<sup>[5](#fn5)</sup>(org) | 0.0180 | 0.9960 | 0.9978 | 0.9111 | 0.1881 |
| VIF(ours imported) | 0.0180 | 0.9960 | 0.9978 | 0.9111 | 0.1881 |
| GMSD<sup>[6](#fn6)</sup>(org) | 0.2120 | 0.0002 | 0.0002 | 0.1317 | 0.1865 |
| GMSD(ours imported) | 0.2120 | 0.0002 | 0.0002 | 0.1317 | 0.1865 |
| NLPD<sup>[7](#fn7)</sup>(org) | 0.5127 | 0.0122 | 0.0097 |  0.2840 | 0.3948 |
| NLPD(ours imported) | 0.5132 | 0.0122 | 0.0098 | 0.2844 | 0.3958 |
| VSI<sup>[8](#fn8)</sup>(opt) | 0.9139 | 0.9620 | 0.9922 |  0.9571 | 0.9262 |
| VSI(ours imported) | 0.9244 | 0.9497 | 0.9877 | 0.9541 | 0.9348 |
| MAD<sup>[10](#fn10)</sup>(ours imported) | 188.17 | 0.0000 | 0.0000 |  89.15 | 174.02 |
| NIQE<sup>[11](#fn11)</sup>(org) | 15.80 | 3.78 | 3.18 |  3.17 | 8.76 |
| NIQE(ours imported) | 15.83 | 3.78 | 3.18 | 3.16 | 8.78 |
| BRISQUE<sup>[12](#fn12)</sup>(org) | 109.75 | -0.1951 | 0.9376 |  5.30 | 69.99 |
| BRISQUE(ours imported) | 109.75 | -0.1971 | 0.9905 | 5.44 | 64.59 |

#### Notice
<a name="fn1">[1]</a> CKDN used degraded images as references in the original paper.   
<a name="fn2">[2]</a> The original SSIM matlab script downsample the image when larger than 256. We remove such constraint.   
<a name="fn3">[3]</a> We use Y-channel of YCBCR images as input of original MS-SSIM matlab script.  
<a name="fn4">[4]</a> The original PSNR code refers to scikit-learn package with RGB 3-channel calculation (from skimage.metrics import peak_signal_noise_ratio).  
<a name="fn5">[5]</a> We use Y-channel of YCBCR images as input of original VIF matlab script.  
<a name="fn6">[6]</a> We use Y-channel of YCBCR images as input of original GMSD matlab script.  
<a name="fn7">[7]</a> We use Y-channel of YCBCR images as input of original NLPD matlab script, and try to mimic 'imfilter' and 'conv2' functions in matlab.  
<a name="fn8">[8]</a> Since official matlab code is not available, we use the implement of IQA-Optimization for comparation. The differences are described as follows. After modifying the above implementation, the results are basically the same.
1. we use interpolation to transform the image to 256*256 and then back to the image size after calculating VSMap in the SDSP function 
2. rgb2lab's function is slightly different
3. the range of ours is -127 to 128 when constructing SDMap, and the value of optimization is -128 to 127
4. different down-sampling operations  

<a name="fn9">[9]</a> We use Y-channel of YCBCR images as input of original CW-SSIM matlab script. The number of level is 4 and orientation is 8.
<a name="fn10">[10]</a> We use Y-channel of YCBCR images as input, and the original MAD matlab script is not available.
<a name="fn11">[11]</a> We use Y-channel of YCBCR images as input of original NIQE matlab script.
<a name="fn11">[12]</a> We use Y-channel of YCBCR images as input of original BRISQUE matlab script.
