# Model Cards for IQA-PyTorch

## General FR/NR Methods

List all model names with:
```
import pyiqa
print(pyiqa.list_models())
```

| FR Method                | Model names | Description
| ------------------------ | ------------------ | ------------ |
| TOPIQ |  `topiq_fr`, `topiq_fr-pipal` | Proposed in [this paper](https://arxiv.org/abs/2308.03060) | 
| AHIQ                     |  `ahiq` |
| PieAPP                   |  `pieapp` |
| LPIPS                    |  `lpips`, `lpips-vgg`, `stlpips`, `stlpips-vgg`  |
| DISTS                    |  `dists` |
| WaDIQaM                  |  | *No pretrain models* |
| CKDN<sup>[1](#fn1)</sup> |  `ckdn` |
| FSIM                     |  `fsim` |
| SSIM                     |  `ssim`, `ssimc` | Gray input (y channel), color input
| MS-SSIM                  |  `ms_ssim` |
| CW-SSIM                  |  `cw_ssim` |
| PSNR                     |  `psnr`, `psnry` | Color input, gray input (y channel)
| VIF                      |  `vif` |
| GMSD                     |  `gmsd` |
| NLPD                     |  `nlpd` |
| VSI                      |  `vsi` |
| MAD                      |  `mad` |

| NR Method                    | Model names | Description |
| ---------------------------- | ------------------------ | ------ |
| TOPIQ | `topiq_nr`, `topiq_nr-flive`, `topiq_nr-spaq` | [TOPIQ](https://arxiv.org/abs/2308.03060) with different datasets, `koniq` by default |
| TReS | `tres`, `tres-koniq`, `tres-flive` | TReS with different datasets, `koniq` by default |
| FID                          | `fid` | Statistic distance between two datasets |
| CLIPIQA(+)                   |  `clipiqa`, `clipiqa+`, `clipiqa+_vitL14_512`,`clipiqa+_rn50_512`  | CLIPIQA(+) with different backbone, RN50 by default |
| MANIQA                       | `maniqa`, `maniqa-kadid`, `maniqa-koniq`, `maniqa-pipal` |MUSIQ with different datasets, `koniq` by default |
| MUSIQ                        | `musiq`, `musiq-koniq`, `musiq-spaq`, `musiq-paq2piq`, `musiq-ava` | MUSIQ with different datasets, `koniq` by default |
| DBCNN                        | `dbcnn` |
| PaQ-2-PiQ                    | `paq2piq` |
| HyperIQA                     |  `hyperiqa` |
| NIMA                         |  `nima`, `nima-vgg16-ava` | Aesthetic metric trained with AVA dataset |
| WaDIQaM                      |  | *No pretrain models*
| CNNIQA                       |  `cnniqa` |
| NRQM(Ma)<sup>[2](#fn2)</sup> |  `nrqm` | No backward |
| PI(Perceptual Index)         |  `pi` | No backward |
| BRISQUE                      | `brisque` | No backward |
| ILNIQE                       | `ilniqe` | No backward |
| NIQE                         | `niqe` | No backward |
<!-- </tr>
</table> -->

<a name="fn1">[1]</a> This method use distorted image as reference. Please refer to the paper for details.<br>
<a name="fn2">[2]</a> Currently, only naive random forest regression is implemented and **does not** support backward.

## IQA Methods for Specific Tasks

| Task           | Method  | Description                                                                                                                                                                 |
| -------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Face IQA | `topiq_nr-face` | TOPIQ model trained with face IQA dataset (GFIQA) |
| Underwater IQA | `uranker` | A ranking-based underwater image quality assessment (UIQA) method, AAAI2023, [Arxiv](https://arxiv.org/abs/2208.06857), [Github](https://github.com/RQ-Wu/UnderwaterRanker) |

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