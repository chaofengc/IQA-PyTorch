# Dataset Preparation

- [Dataset Preparation](#dataset-preparation)
  - [Supported Datasets](#supported-datasets)
  - [Resources](#resources)
  - [Interface of Dataloader](#interface-of-dataloader)
  - [Specific Datasets and Dataloader](#specific-datasets-and-dataloader)
  - [Test Dataloader](#test-dataloader)

## Supported Datasets

The following datasets can be loaded with the current codes after downloaded (see example [scripts](../options/example_benchmark_data_opts.yml)):

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

We create general interfaces for FR and NR datasets in `pyiqa/data/general_fr_dataset.py` and `pyiqa/data/general_nr_dataset.py`. The main arguments are

- `opt` contains all dataset options, including
    - `dataroot_target`: path of target image folder.
    - `dataroot_ref [optional]`: path of reference image folder.
    - `meta_info_file`: file containing meta information of images, including relative image paths, mos labels and other labels.
    - `augment [optional]` data augmentation transform list
        - `hflip`: flip input images or pairs
        - `random_crop`: int or tuple, random crop input images or pairs
    - `split_file [optional]`: `train/val/test` split file `*.pkl`. If not specified, will use the split information in meta csv file or load the whole dataset.
    - `split_index [optional]`: `str` or `int`, which split to use, valid when `split_file` is specified or corresponding split information exits in meta csv file.
    - `dmos max`: some dataset use difference of mos. Set this to non-zero will change dmos to mos with `mos = dmos_max - dmos`.
    - `phase`: phase labels [train, val, test]

The above interface requires the `meta_info_file` to provide the dataset information and the train/val/test split. The `meta_info_file` are `.csv` files, and has the following general format
```
- For NR datasets: name, mos(mean), std, split_name
    ```
    100.bmp   	32.56107532210109   	19.12472638223644   train/val/test
    ```

- For FR datasets: ref_name, dist_name, mos(mean), std, split_name 
    ```
    I01.bmp        I01_01_1.bmp   5.51429        0.13013 train/val/test

    ```
```

Note that we generate `train/val/test` splits follow the principles below:

- For datasets which has official splits, we follow their splits.
- For official split which has no `val` part, e.g., AVA dataset, we random separate 5% from training data as validation.
- For small datasets which requires n-split results, we use `train:val=8:2`  ratio.
- All random seeds are set to `123` when needed.

According to these rules, the `split_name` is named as follows:

- The official split is saved in a column named `official_split`.
- [if necessary] Ten random splits are generated and stored using the format `ratio[split_ratio]_seed[seed number]_split[split index:02d]`. For example, for a split ratio of `train/val/test=8:0:2`, a seed number of 123, and the first split, the entry would be `ratio802_seed123_split01`.
- You can also use other custom split names, such as the `ILGnet_split` for the AVA dataset.

### Using separate split file

You may also use the `split_file` to specify the split information. The `split_file` are `.pkl` files which contains the `train/val/test` information with python dictionary in the following format:
```
{
    train_index: {
        train: [train_index_list]
        val: [val_index_list] # blank if no validation split
        test: [test_index_list] # blank if no test split
    }
}
```
The train_index starts from `1`. And the sample indexes correspond to the row index of `meta_info_file`, starting from `0`. We already generate the files for mainstream public datasets with scripts in folder [./scripts/](./scripts/).

## Specific Datasets and Dataloader

Some of the supported datasets have different label formats and file organizations, and we create specific dataloader for them:

- Live Challenge. The first 7 samples are usually removed in the related works.
- AVA. Different label formats.
- PieAPP. Different label formats.
- BAPPS. Different label formats.

## Test Dataloader

You may use `tests/test_datasets.py` to test whether a dataset can be correctly loaded.
