# Benchmark Calibration

We random select 5 pairs of images from TID2013 for results calibration. Images are stored under `./CalibraTest/dist_dir` and `./CalibraTest/ref_dir`. Distorted images and reference images are renamed as the same for simplicity.

Note that we aims to calibrate the results here. Therefore, for simplicity, we keep dist and ref the same for all methods, including CKDN which used degraded images as references in the original paper.

| Method | I03.bmp | I04.bmp | I06.bmp | I08.bmp | I19.bmp |
| --- | --- | --- | --- | --- | --- |
| CKDN(org) | 0.2833 | 0.5766 | 0.6367 | 0.6579 | 0.5999 |
| CKDN(ours imported) | 0.2833 | 0.5766 | 0.6367 | 0.6579 | 0.5999 |
| LPIPS(org) | 0.7237 | 0.2572 | 0.0508 | 0.0521 | 0.4253 |
| LPIPS(ours imported) | 0.7237 | 0.2572 | 0.0508 | 0.0521 | 0.4253 |
| DISTS(org) | 0.4742 | 0.1424 | 0.0682 | 0.0287 | 0.3123 |
| DISTS(ours imported) | 0.4742 | 0.1424 | 0.0682 | 0.0287 | 0.3123 |
