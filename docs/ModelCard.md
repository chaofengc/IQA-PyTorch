# Model Cards for IQA-PyTorch

## General FR/NR Methods

<table>
<tr><td>

| FR Method                | Backward           |
| ------------------------ | ------------------ |
| AHIQ                     | :white_check_mark: |
| PieAPP                   | :white_check_mark: |
| LPIPS                    | :white_check_mark: |
| DISTS                    | :white_check_mark: |
| WaDIQaM                  | :white_check_mark: |
| CKDN<sup>[1](#fn1)</sup> | :white_check_mark: |
| FSIM                     | :white_check_mark: |
| SSIM                     | :white_check_mark: |
| MS-SSIM                  | :white_check_mark: |
| CW-SSIM                  | :white_check_mark: |
| PSNR                     | :white_check_mark: |
| VIF                      | :white_check_mark: |
| GMSD                     | :white_check_mark: |
| NLPD                     | :white_check_mark: |
| VSI                      | :white_check_mark: |
| MAD                      | :white_check_mark: |

</td><td>

| NR Method                    | Backward                 |
| ---------------------------- | ------------------------ |
| FID                          | :heavy_multiplication_x: |
| CLIPIQA(+)                   | :white_check_mark:       |
| MANIQA                       | :white_check_mark:       |
| MUSIQ                        | :white_check_mark:       |
| DBCNN                        | :white_check_mark:       |
| PaQ-2-PiQ                    | :white_check_mark:       |
| HyperIQA                     | :white_check_mark:       |
| NIMA                         | :white_check_mark:       |
| WaDIQaM                      | :white_check_mark:       |
| CNNIQA                       | :white_check_mark:       |
| NRQM(Ma)<sup>[2](#fn2)</sup> | :heavy_multiplication_x: |
| PI(Perceptual Index)         | :heavy_multiplication_x: |
| BRISQUE                      | :white_check_mark:       |
| ILNIQE                       | :white_check_mark:       |
| NIQE                         | :white_check_mark:       |
</tr>
</table>

<a name="fn1">[1]</a> This method use distorted image as reference. Please refer to the paper for details.<br>
<a name="fn2">[2]</a> Currently, only naive random forest regression is implemented and **does not** support backward.

## IQA Methods for Specific Tasks

| Task           | Method  | Description                                                                                                                                                                 |
| -------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Underwater IQA | URanker | A ranking-based underwater image quality assessment (UIQA) method, AAAI2023, [Arxiv](https://arxiv.org/abs/2208.06857), [Github](https://github.com/RQ-Wu/UnderwaterRanker) |

## Outputs of Different Metrics 
**Note: `~` means that the corresponding numeric bound is typical value and not mathematically guaranteed**

| model    | lower better ? | min | max     | DATE | Link                                                                                                                                                      |
| -------- | -------------- | --- | ------- | ---- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| clipiqa  | False          | 0   | 1       | 2022 | https://arxiv.org/abs/2207.12396                                                                                                                          |
| maniqa   | False          | 0   |        | 2022 | https://arxiv.org/abs/2204.08958                                                                                                                          |
| hyperiqa | False          | 0   | 1       | 2020 | [pdf](https://openaccess.thecvf.com/content_CVPR_2020/papers/Su_Blindly_Assess_Image_Quality_in_the_Wild_Guided_by_a_CVPR_2020_paper.pdf)                 |
| cnniqa   | False          |   |       | 2014 | [pdf](https://openaccess.thecvf.com/content_cvpr_2014/papers/Kang_Convolutional_Neural_Networks_2014_CVPR_paper.pdf)                                      |
| tres     | False          |    | | 2022 | https://github.com/isalirezag/TReS                                                                                                                        |
| musiq    | False          |  ~0 | ~100 | 2021 | https://arxiv.org/abs/2108.05997                                                                                                                          |
| musiq-ava    | False          |  ~0  | ~10 | 2021 | https://arxiv.org/abs/2108.05997                                                                                                                          |
| musiq-koniq    | False          | ~0 | ~100 | 2021 | https://arxiv.org/abs/2108.05997                                                                                                                          |
| musiq    | False          |    | | 2021 | https://arxiv.org/abs/2108.05997                                                                                                                          |
| paq2piq  | False          |    | | 2020 | [pdf](https://openaccess.thecvf.com/content_CVPR_2020/papers/Ying_From_Patches_to_Pictures_PaQ-2-PiQ_Mapping_the_Perceptual_Space_of_CVPR_2020_paper.pdf) |
| dbcnn    | False          |    | | 2019 | https://arxiv.org/bas/1907.02665                                                                                                                          |
| brisque  | True           |    | | 2012 | [pdf](https://live.ece.utexas.edu/publications/2012/TIP%20BRISQUE.pdf)                                                                                    |
| pi       | True           |    | | 2018 | https://arxiv.org/abs/1809.07517                                                                                                                          |
| nima     | False          |   | | 2018 | https://arxiv.org/abs/1709.05424                                                                                                                          |
| nrqm     | False          |   | | 2016 | https://arxiv.org/abs/1612.05890                                                                                                                          |
| ilniqe   | True           | 0   | | 2015 | [pdf](http://www4.comp.polyu.edu.hk/~cslzhang/paper/IL-NIQE.pdf)                                                                                          |
| niqe     | True           | 0   | | 2012 | [pdf](https://live.ece.utexas.edu/publications/2013/mittal2013.pdf)                                                                                       |