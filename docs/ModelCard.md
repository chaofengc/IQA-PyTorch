# Model Cards for IQA-PyTorch

## General FR/NR Methods

List all model names with:
```
import pyiqa
print(pyiqa.list_models())
```

| FR Method                | Model names                                                                                                                         | Description
| ------------------------ |-------------------------------------------------------------------------------------------------------------------------------------| ------------ |
| TOPIQ | `topiq_fr`, `topiq_fr-pipal`                                                                                                        | Proposed in [this paper](https://arxiv.org/abs/2308.03060) | 
| AHIQ                     | `ahiq`                                                                                                                              |
| PieAPP                   | `pieapp`                                                                                                                            |
| LPIPS                    | `lpips`, `lpips-vgg`, `stlpips`, `stlpips-vgg`, `lpips+`, `lpips-vgg+`                                                                                      |
| DISTS                    | `dists`                                                                                                                             |
| WaDIQaM                  | `wadiqam_fr`                                                                                                                                     |  |
| CKDN<sup>[1](#fn1)</sup> | `ckdn`                                                                                                                              |
| FSIM                     | `fsim`                                                                                                                              |
| SSIM                     | `ssim`, `ssimc`                                                                                                                     | Gray input (y channel), color input
| MS-SSIM                  | `ms_ssim`                                                                                                                           |
| CW-SSIM                  | `cw_ssim`                                                                                                                           |
| PSNR                     | `psnr`, `psnry`                                                                                                                     | Color input, gray input (y channel)
| VIF                      | `vif`                                                                                                                               |
| GMSD                     | `gmsd`                                                                                                                              |
| NLPD                     | `nlpd`                                                                                                                              |
| VSI                      | `vsi`                                                                                                                               |
| MAD                      | `mad`                                                                                                                               |

| NR Method                    | Model names | Description                                                                         |
| ---------------------------- | ------------------------ |-------------------------------------------------------------------------------------|
| Q-Align                  | `qalign` (with quality[default], aesthetic options)                                                                                                                            | Large vision-language models |
| LIQE | `liqe`, `liqe_mix` | CLIP based method  |
| ARNIQA                   | `arniqa`, `arniqa-live`, `arniqa-csiq`, `arniqa-tid`, `arniqa-kadid`, `arniqa-clive`, `arniqa-flive`, `arniqa-spaq` | [ARNIQA](https://arxiv.org/abs/2310.14918) with different datasets, `koniq` by default |
| TOPIQ | `topiq_nr`, `topiq_nr-flive`, `topiq_nr-spaq` | [TOPIQ](https://arxiv.org/abs/2308.03060) with different datasets, `koniq` by default |
| TReS | `tres`, `tres-flive` | TReS with different datasets, `koniq` by default                                    |
| FID                          | `fid` | Statistic distance between two datasets                                             |
| CLIPIQA(+)                   |  `clipiqa`, `clipiqa+`, `clipiqa+_vitL14_512`,`clipiqa+_rn50_512`  | CLIPIQA(+) with different backbone, RN50 by default                                 |
| MANIQA                       | `maniqa`, `maniqa-kadid`, `maniqa-pipal` | MUSIQ with different datasets, `koniq` by default                                   |
| MUSIQ                        | `musiq`, `musiq-spaq`, `musiq-paq2piq`, `musiq-ava` | MUSIQ with different datasets, `koniq` by default                                   |
| DBCNN                        | `dbcnn` |
| PaQ-2-PiQ                    | `paq2piq` |
| HyperIQA                     |  `hyperiqa` |
| NIMA                         |  `nima`, `nima-vgg16-ava` | Aesthetic metric trained with AVA dataset                                           |
| WaDIQaM                      | `wadiqam_nr` | |                                                                
| CNNIQA                       |  `cnniqa` |
| NRQM(Ma)<sup>[2](#fn2)</sup> |  `nrqm` | No backward                                                                         |
| PI(Perceptual Index)         |  `pi` | No backward                                                                         |
| BRISQUE                      | `brisque`, `brisque_matlab` | No backward                                                                         |
| ILNIQE                       | `ilniqe` | No backward                                                                         |
| NIQE                         | `niqe`, `niqe_matlab` | No backward                                                                         |
| PIQE                         | `piqe` | No backward                                                                         |
<!-- </tr>
</table> -->

<a name="fn1">[1]</a> This method use distorted image as reference. Please refer to the paper for details.<br>
<a name="fn2">[2]</a> Currently, only naive random forest regression is implemented and **does not** support backward.

## IQA Methods for Specific Tasks

| Task           | Method  | Description                                                                                                                                                                 |
| -------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Color IQA | `msswd` | Perceptual color difference metric MS-SWD, ECCV2024, [Arxiv](http://arxiv.org/abs/2407.10181), [Github](https://github.com/real-hjq/MS-SWD)
| Face IQA | `topiq_nr-face` | TOPIQ model trained with face IQA dataset (GFIQA) |
| Underwater IQA | `uranker` | A ranking-based underwater image quality assessment (UIQA) method, AAAI2023, [Arxiv](https://arxiv.org/abs/2208.06857), [Github](https://github.com/RQ-Wu/UnderwaterRanker) |

## Metric Output Score Range
**Note: `~` means that the corresponding numeric bound is typical value and not mathematically guaranteed**

You can now access the **rough** output range of each metric like this:
```
metric = pyiqa.create_metric('lpips')
print(metric.score_range)
```