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

| Task | Method | Description |
| ---- | ---- | ---- |
| Underwater IQA | URanker | A ranking-based underwater image quality assessment (UIQA) method, AAAI2023, [Arxiv](https://arxiv.org/abs/2208.06857), [Github](https://github.com/RQ-Wu/UnderwaterRanker) | 