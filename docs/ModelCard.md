# Model Cards for IQA-PyTorch

## General FR/NR Methods

List all model names with:
```
import pyiqa
print(pyiqa.list_models())
```

| FR Method                | Model names                                                            | Description                                                | Score Direction  |
| ------------------------ | ---------------------------------------------------------------------- | ---------------------------------------------------------- | ---------------- |
| TOPIQ                    | `topiq_fr`, `topiq_fr-pipal`                                           | Proposed in [this paper](https://arxiv.org/abs/2308.03060) | Higher is better |
| AHIQ                     | `ahiq`                                                                 |                                                            | Higher is better |
| PieAPP                   | `pieapp`                                                               |                                                            | Lower is better  |
| LPIPS                    | `lpips`, `lpips-vgg`, `stlpips`, `stlpips-vgg`, `lpips+`, `lpips-vgg+` |                                                            | Lower is better  |
| DISTS                    | `dists`                                                                |                                                            | Lower is better  |
| WaDIQaM                  | `wadiqam_fr`                                                           |                                                            | Higher is better |
| CKDN<sup>[1](#fn1)</sup> | `ckdn`                                                                 |                                                            | Higher is better |
| FSIM                     | `fsim`                                                                 |                                                            | Higher is better |
| SSIM                     | `ssim`, `ssimc`                                                        | Gray input (y channel), color input                        | Higher is better |
| MS-SSIM                  | `ms_ssim`                                                              |                                                            | Higher is better |
| CW-SSIM                  | `cw_ssim`                                                              |                                                            | Higher is better |
| PSNR                     | `psnr`, `psnry`                                                        | Color input, gray input (y channel)                        | Higher is better |
| VIF                      | `vif`                                                                  |                                                            | Higher is better |
| GMSD                     | `gmsd`                                                                 |                                                            | Lower is better  |
| NLPD                     | `nlpd`                                                                 |                                                            | Lower is better  |
| VSI                      | `vsi`                                                                  |                                                            | Higher is better |
| MAD                      | `mad`                                                                  |                                                            | Lower is better  |


| NR Method                    | Model names                                                                                                         | Description                                                                                  | Score Direction  |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------- |--------------------------------------------------------------------------------------------- | ---------------- |
| Q-Align                      | `qalign` (with quality[default], aesthetic options)                                                                 | Large vision-language models                                                                 | Higher is better |
| QualiCLIP(+)                 | `qualiclip`, `qualiclip+`, `qualiclip+-clive`, `qualiclip+-flive`, `qualiclip+-spaq`                                | [QualiCLIP(+)](https://arxiv.org/abs/2403.11176) with different datasets, `koniq` by default | Higher is better |
| LIQE                         | `liqe`, `liqe_mix`                                                                                                  | CLIP based method                                                                            | Higher is better |
| ARNIQA                       | `arniqa`, `arniqa-live`, `arniqa-csiq`, `arniqa-tid`, `arniqa-kadid`, `arniqa-clive`, `arniqa-flive`, `arniqa-spaq` | [ARNIQA](https://arxiv.org/abs/2310.14918) with different datasets, `koniq` by default       | Higher is better |
| TOPIQ                        | `topiq_nr`, `topiq_nr-flive`, `topiq_nr-spaq`                                                                       | [TOPIQ](https://arxiv.org/abs/2308.03060) with different datasets, `koniq` by default        | Higher is better |
| TReS                         | `tres`, `tres-flive`                                                                                                | TReS with different datasets, `koniq` by default                                             | Higher is better |
| FID                          | `fid`                                                                                                               | Statistic distance between two datasets                                                      | Lower is better  |
| CLIPIQA(+)                   | `clipiqa`, `clipiqa+`, `clipiqa+_vitL14_512`,`clipiqa+_rn50_512`                                                    | CLIPIQA(+) with different backbone, RN50 by default                                          | Higher is better |
| MANIQA                       | `maniqa`, `maniqa-kadid`, `maniqa-pipal`                                                                            | MUSIQ with different datasets, `koniq` by default                                            | Higher is better |
| MUSIQ                        | `musiq`, `musiq-spaq`, `musiq-paq2piq`, `musiq-ava`                                                                 | MUSIQ with different datasets, `koniq` by default                                            | Higher is better |
| DBCNN                        | `dbcnn`                                                                                                             |                                                                                              | Higher is better |
| PaQ-2-PiQ                    | `paq2piq`                                                                                                           |                                                                                              | Higher is better |
| HyperIQA                     | `hyperiqa`                                                                                                          |                                                                                              | Higher is better |
| NIMA                         | `nima`, `nima-vgg16-ava`                                                                                            | Aesthetic metric trained with AVA dataset                                                    | Higher is better |
| WaDIQaM                      | `wadiqam_nr`                                                                                                        |                                                                                              | Higher is better |
| CNNIQA                       | `cnniqa`                                                                                                            |                                                                                              | Higher is better |
| NRQM(Ma)<sup>[2](#fn2)</sup> | `nrqm`                                                                                                              | No backward                                                                                  | Higher is better |
| PI(Perceptual Index)         | `pi`                                                                                                                | No backward                                                                                  | Lower is better  |
| BRISQUE                      | `brisque`, `brisque_matlab`                                                                                         | No backward                                                                                  | Lower is better  |
| ILNIQE                       | `ilniqe`                                                                                                            | No backward                                                                                  | Lower is better  |
| NIQE                         | `niqe`, `niqe_matlab`                                                                                               | No backward                                                                                  | Lower is better  |
| PIQE                         | `piqe`                                                                                                              | No backward                                                                                  | Lower is better  |
<!-- </tr>
</table> -->

<a name="fn1">[1]</a> This method use distorted image as reference. Please refer to the paper for details.<br>
<a name="fn2">[2]</a> Currently, only naive random forest regression is implemented and **does not** support backward.

## IQA Methods for Specific Tasks

| Task           | Method          | Description                                                                                                                                                                 | Score Direction  |
| -------------- | --------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------- |
| Color IQA      | `msswd`         | Perceptual color difference metric MS-SWD, ECCV2024, [Arxiv](http://arxiv.org/abs/2407.10181), [Github](https://github.com/real-hjq/MS-SWD)                                 | Lower is better  |
| Face IQA       | `topiq_nr-face` | TOPIQ model trained with face IQA dataset (GFIQA)                                                                                                                           | Higher is better |
| Underwater IQA | `uranker`       | A ranking-based underwater image quality assessment (UIQA) method, AAAI2023, [Arxiv](https://arxiv.org/abs/2208.06857), [Github](https://github.com/RQ-Wu/UnderwaterRanker) | Higher is better |

## Metric Output Score Range
**Note: `~` means that the corresponding numeric bound is typical value and not mathematically guaranteed**

You can now access the **rough** output range of each metric like this:
```
metric = pyiqa.create_metric('lpips')
print(metric.score_range)
```