# Dataset preparation

This guide describes the dataset adapters used by pyiqa's training and
evaluation utilities. It does not distribute source image datasets; download
them from the dataset owners and follow their licenses and terms of use.

- [Dataset Preparation](#dataset-preparation)
  - [Supported Datasets](#supported-datasets)
  - [Resources](#resources)
  - [Interface of Dataloader](#interface-of-dataloader)
  - [Specific Datasets and Dataloader](#specific-datasets-and-dataloader)
  - [Test Dataloader](#test-dataloader)

## Supported Datasets

The following datasets can be loaded after downloading and arranging their
files as expected by the corresponding configuration (see the example
[benchmark options](../options/example_benchmark_data_opts.yml)):

| FR Dataset | Description | NR Dataset       | Description        |
| ---------- | ----------- | ---------------- | ------------------ |
| PIPAL      | *2AFC*      | FLIVE(PaQ-2-PiQ) | *Tech & Aesthetic* |
| BAPPS      | *2AFC*      | SPAQ             | *Mobile*           |
| PieAPP     | *2AFC*      | AVA              | *Aesthetic*        |
| KADID-10k  |             | KonIQ-10k(++)    |                    |
| LIVEM      |             | LIVEChallenge    |                    |
| LIVE       |             | [PIQ2023](https://github.com/DXOMARK-Research/PIQ2023)| Portrait dataset   |
| TID2013    |             | [GFIQA](http://database.mmsp-kn.de/gfiqa-20k-database.html)| Face IQA Dataset   |
| TID2008    |             |                  |                    |
| CSIQ       |             |                  |                    |

Please see more details at [Awesome Image Quality Assessment](https://github.com/chaofengc/Awesome-Image-Quality-Assessment)

## Resources

Here are some other resources to download the dataset:
- [**Our huggingface archive 🤗**](https://huggingface.co/datasets/chaofengc/IQA-Toolbox-Datasets/tree/main)
- [**Waterloo Bayesian IQA project**](http://ivc.uwaterloo.ca/research/bayesianIQA/). [ [IQA-Dataset](https://github.com/icbcbicc/IQA-Dataset) | [download links](http://ivc.uwaterloo.ca/database/IQADataset) ]

## Interface of Dataloader

General FR and NR dataset interfaces are implemented in
`pyiqa/data/general_fr_dataset.py` and `pyiqa/data/general_nr_dataset.py`. The
main options include:

- `opt` contains all dataset options, including
    - `dataroot_target`: target/distorted image directory.
    - `dataroot_ref` (optional): reference image directory for FR datasets.
    - `meta_info_file`: metadata file with relative image paths, MOS/DMOS labels,
      and any dataset-specific fields.
    - `augment` (optional): data augmentation settings, such as `hflip` or
      `random_crop`; paired FR images receive consistent geometric transforms.
    - `split_file` (optional): pickle file defining train/validation/test
      indices. When omitted, split metadata or the complete dataset is used.
    - `split_index` (optional): split name or index selected from the metadata
      or split file.
    - `dmos_max` (optional): convert DMOS to MOS using
      `mos = dmos_max - dmos` for datasets that require it.
    - `phase`: dataset phase, typically `train`, `val`, or `test`.

The above interface requires `meta_info_file` to provide dataset metadata and,
optionally, split labels. The delimiter must match the selected dataset
configuration; some metadata files use tabs despite having a `.csv` extension.
Typical columns are:

- NR datasets: image name, MOS/DMOS, optional standard deviation, optional
  split name.
- FR datasets: reference image name, distorted image name, MOS/DMOS, optional
  standard deviation, optional split name.

For example, an NR metadata row may look like:

```text
100.bmp    32.56107532210109    19.12472638223644    official_split
```

An FR metadata row may look like:

```text
I01.bmp    I01_01_1.bmp    5.51429    0.13013    official_split
```

The provided `train/val/test` splits follow these principles:

- For datasets which has official splits, we follow their splits.
- For official split which has no `val` part, e.g., AVA dataset, we random separate 5% from training data as validation.
- For small datasets which requires n-split results, we use `train:val=8:2`  ratio.
- All random seeds are set to `123` when needed.

Split names use the following conventions:

- The official split is saved in a column named `official_split`.
- [if necessary] Ten random splits are generated and stored using the format `ratio[split_ratio]_seed[seed number]_split[split index:02d]`. For example, for a split ratio of `train/val/test=8:0:2`, a seed number of 123, and the first split, the entry would be `ratio802_seed123_split01`.
- You can also use other custom split names, such as the `ILGnet_split` for the AVA dataset.

### Using separate split file

You may also use `split_file` to specify split membership. The pickle file
contains a mapping from a one-based split index to zero-based metadata row
indices. Empty lists represent unused phases:

```python
split_file = {
    1: {
        'train': [0, 1, 2],
        'val': [],
        'test': [3, 4],
    },
}
```

Example split files for common public datasets are generated by scripts in
the repository's [`scripts/`](../scripts/) directory.

## Specific Datasets and Dataloader

Some datasets use different label formats or directory layouts and therefore
have dedicated adapters:

- LIVE Challenge: related work often excludes the first seven samples.
- AVA: aesthetic-rating labels and splits.
- PieAPP: pairwise preference labels.
- BAPPS: pairwise perceptual judgments.

## Test Dataloader

Use `pytest tests/test_datasets_general.py` to check dataset loading. Dataset
tests may require local data and should be configured with the paths described
in `options/`.
