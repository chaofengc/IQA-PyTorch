# Dataset Preparation

- [Dataset Preparation](#dataset-preparation)
  - [Interface of Dataloader](#interface-of-dataloader)
  - [Specific Datasets and Dataloader](#specific-datasets-and-dataloader)
  - [Test Dataloader](#test-dataloader)

## Interface of Dataloader

We create general interfaces for FR and NR datasets in `pyiqa/data/general_fr_dataset.py` and `pyiqa/data/general_nr_dataset.py`. The main arguments are

- `opt` contains all dataset options, including  
    - `dataroot_target`: path of target image folder.   
    - `dataroot_ref [optional]`: path of reference image folder. 
    - `meta_info_file`: file containing meta information of images, including relative image paths, mos labels and other labels.
    - `augment [optional]` data augmentation transform list
        - `use_hflip/use_paired_hflip`: flip input images or pairs 
        - `random_crop/paired_random_crop`: int or tuple, random crop input images 
    - `split_file [optional]`: `train/val/test` split file `*.pkl`. If not specified, will load the whole dataset. 
    - `split_index [optional]`: default `1`, which split to use, only valid when `split_file` is specified.   
    - `phase`: phase labels [train, val, test]

The above interface requires two files to provide the dataset information, i.e., the `meta_info_file` and `split_file`. The `meta_info_file` are `.csv` files, and has the following general format
```
- For NR datasets: name, mos(mean), std
    ```
    100.bmp   	32.56107532210109   	19.12472638223644 
    ```
    
- For FR datasets: ref_name, dist_name, mos(mean), std
    ```
    I01.bmp        I01_01_1.bmp   5.51429        0.13013
    ```
```
The `split_file` are `.pkl` files which contains the `train/val/test` information with python dictionary in the following format:
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
