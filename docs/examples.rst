Examples
===================

Basic Usage
--------------
::

    import pyiqa
    import torch

    # list all available metrics
    print(pyiqa.list_models())

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # create metric with default setting
    iqa_metric = pyiqa.create_metric('lpips', device=device)
    # Note that gradient propagation is disabled by default. set as_loss=True to enable it as a loss function.
    iqa_loss = pyiqa.create_metric('lpips', device=device, as_loss=True)

    # create metric with custom setting
    iqa_metric = pyiqa.create_metric('psnr', test_y_channel=True, color_space='ycbcr').to(device)

    # check if lower better or higher better
    print(iqa_metric.lower_better)

    # example for iqa score inference
    # Tensor inputs, img_tensor_x/y: (N, 3, H, W), RGB, 0 ~ 1
    score_fr = iqa_metric(img_tensor_x, img_tensor_y)
    score_nr = iqa_metric(img_tensor_x)

    # img path as inputs.
    score_fr = iqa_metric('./ResultsCalibra/dist_dir/I03.bmp', './ResultsCalibra/ref_dir/I03.bmp')

    # For FID metric, use directory or precomputed statistics as inputs
    # refer to clean-fid for more details: https://github.com/GaParmar/clean-fid
    fid_metric = pyiqa.create_metric('fid')
    score = fid_metric('./ResultsCalibra/dist_dir/', './ResultsCalibra/ref_dir')
    score = fid_metric('./ResultsCalibra/dist_dir/', dataset_name="FFHQ", dataset_res=1024, dataset_split="trainval70k")