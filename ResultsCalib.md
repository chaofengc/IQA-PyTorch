# Results Calibration

We random select 5 pairs of images from TID2013 for results calibration. Images are stored under `./ResultsCalibra/dist_dir` and `./ResultsCalibra/ref_dir`. Distorted images and reference images are renamed as the same for simplicity.

| Method | I03.bmp | I04.bmp | I06.bmp | I08.bmp | I19.bmp |
| --- | --- | --- | --- | --- | --- |
| CKDN<sup>1</sup>(org) | 0.2833 | 0.5766 | 0.6367 | 0.6579 | 0.5999 |
| CKDN(ours imported) | 0.2833 | 0.5766 | 0.6367 | 0.6579 | 0.5999 |
| LPIPS(org) | 0.7237 | 0.2572 | 0.0508 | 0.0521 | 0.4253 |
| LPIPS(ours imported) | 0.7237 | 0.2572 | 0.0508 | 0.0521 | 0.4253 |
| DISTS(org) | 0.4742 | 0.1424 | 0.0682 | 0.0287 | 0.3123 |
| DISTS(ours imported) | 0.4742 | 0.1424 | 0.0682 | 0.0287 | 0.3123 |
| SSIM<sup>2</sup>(org) | 0.7326 | 0.9989 | 0.9995 | 0.9674 | 0.6790 |
| SSIM(ours imported) | 0.7328 | 0.9989 | 0.9995 | 0.9676 | 0.6791 |
| MS-SSIM<sup>3</sup>(org) | 0.6981 | 0.9998 | 0.9999 | 0.9570 | 0.8547 |
| MS-SSIM(ours imported) | 0.6984 | 0.9998 | 0.9999 | 0.9571 | 0.8547 |
| PSNR<sup>4</sup>(org) | 21.11 | 20.99 | 27.01 | 23.30 | 21.62 |
| PSNR(ours imported) | 21.11 | 20.99 | 27.01 | 23.30 | 21.62 |
| FSIM(org) | 0.6890 | 0.9702 | 0.9927 | 0.9575 | 0.8220 |
| FSIM(ours imported) | 0.6891 | 0.9702 | 0.9927 | 0.9575 | 0.8220 |
| VIF<sup>5<sup>(org) | 0.0180 | 0.9960 | 0.9978 | 0.9111 | 0.1881 |
| VIF(ours imported) | 0.0180 | 0.9960 | 0.9978 | 0.9111 | 0.1881 |

#### Notice
1. CKDN used degraded images as references in the original paper.
2. The original SSIM matlab script downsample the image when larger than 256. We remove such constraint. 
3. We use Y-channel of YCBCR images as input of original MS-SSIM matlab script.
4. The original PSNR code refers to scikit-learn package with RGB 3-channel calculation (from skimage.metrics import peak_signal_noise_ratio).
5. We use Y-channel of YCBCR images as input of original VIF matlab script.
