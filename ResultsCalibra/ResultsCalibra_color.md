# Results Calibration color space yiq

We random select 5 pairs of images from TID2013 for results calibration. Images are stored under `./dist_dir` and `./ref_dir`. Results of different metrics are saved under `./results_compare/`. We also record the problems encountered during our reproduction of matlab scripts in [MatlabReproduceNote](./MatlabReproduceNote.md)

| Method                                              | I03.bmp | I04.bmp | I06.bmp | I08.bmp | I19.bmp |
| --------------------------------------------------- | ------- | ------- | ------- | ------- | ------- |

| NIQE(org)                                           | 15.7536 | 3.6549  | 3.2355  | 3.1840  | 8.6352  |
| NIQE(ours imported with round)                      | 15.6541 | 3.6541  | 3.2365  | 3.2030  | 9.1032  | !!
                                                        15.6527,  3.6542,  3.2342,  3.2075,  9.1106
| NIQE(ours imported)                                 | 15.3001 | 3.6416  | 3.1897  | 3.1844  | 8.8653  |

| BRISQUE(org)                                        | 94.6421 | -0.1076 | 0.9929  | 5.3583  | 72.2617 |
| BRISQUE(ours imported)                              | 94.6418 | -0.1043 | 1.0773  | 5.1486  | 66.8376 | !!
                                                        94.6457, -0.1087,  1.0812,  5.1437, 66.8386

| SSIM(org)                                           | 0.6993  | 0.9978  | 0.9989  | 0.9669  | 0.6519  |
| SSIM(ours imported)                                 | 0.6997  | 0.9978  | 0.9989  | 0.9671  | 0.6521  |

| CW-SSIM(org)                                        | 0.2763  | 0.9996  | 1.0000  | 0.9068  | 0.8658  |
| CW-SSIM(ours imported)                              | 0.2782  | 0.9995  | 1.0000  | 0.9065  | 0.8646  |

| MS-SSIM(org)                                        | 0.6733  | 0.9996  | 0.9998  | 0.9566  | 0.8462  |
| MS-SSIM(ours imported)                              | 0.6698  | 0.9993  | 0.9996  | 0.9567  | 0.8418  |
                                                        0.6699  | 0.9993  | 0.9996  | 0.9567  | 0.8418

| VIF(org)                                            | 0.0172  | 0.9891  | 0.9924  | 0.9103  | 0.1745  |
| VIF(ours imported)                                  | 0.0172  | 0.9891  | 0.9924  | 0.9103  | 0.1745  |

| GMSD(org)                                           | 0.2203  | 0.0005  | 0.0004  | 0.1346  | 0.2050  |
| GMSD(ours imported)                                 | 0.2203  | 0.0005  | 0.0004  | 0.1346  | 0.2050  |

| NLPD(org)                                           | 0.5616  | 0.0195  | 0.0159  | 0.3028  | 0.4326  |
| NLPD(ours imported)                                 | 0.5616  | 0.0139  | 0.0110  | 0.3033  | 0.4335  |



| FSIM(org)                                           | 0.6890  | 0.9702  | 0.9927  | 0.9575  | 0.8220  |
| FSIM(ours imported)                                 | 0.6891  | 0.9702  | 0.9927  | 0.9575  | 0.8220  |

| PSNR(org)                                           | 21.11   | 20.99   | 27.01   | 23.30   | 21.62   |
| PSNR(ours imported)                                 | 21.11   | 20.99   | 27.01   | 23.30   | 21.62   |

| MAD(ours imported)                                  |194.9324 | 0.0000  | 0.0000  | 91.6206 | 181.9651|
                                                        155.5415,  26.1180,  21.0799, 135.2712, 131.1859
