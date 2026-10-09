Examples
========

Basic Python API
----------------
::

    import pyiqa
    import torch

    # List configured metrics, including FR, NR, and task-specific metrics.
    print(pyiqa.list_models())

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Create a metric with its default configuration.
    lpips = pyiqa.create_metric('lpips', device=device)

    # Gradients are disabled by default. Enable them for a supported loss.
    lpips_loss = pyiqa.create_metric('lpips', device=device, as_loss=True)

    # Override options supported by a metric's architecture.
    psnr = pyiqa.create_metric(
        'psnr', device=device, test_y_channel=True, color_space='ycbcr'
    )

    # Read score interpretation metadata.
    print(lpips.lower_better)

Inputs and outputs
------------------

Tensor inputs use ``(N, C, H, W)`` layout, one or three channels, and values in
``[0, 1]``. Image paths are decoded as RGB. For full-reference (FR) metrics,
pass the distorted/target image first and the reference image second. No-
reference (NR) metrics require only the target image. Scores are returned as
PyTorch tensors.

::

    # FR: distorted image, then reference image.
    score_fr = lpips('./ResultsCalibra/dist_dir/I03.bmp',
                     './ResultsCalibra/ref_dir/I03.bmp')

    # NR: target image only.
    musiq = pyiqa.create_metric('musiq', device=device)
    score_nr = musiq('./ResultsCalibra/dist_dir/I03.bmp')

    # Tensor input follows the same ordering and value range.
    score_fr = lpips(distorted_tensor, reference_tensor)
    score_nr = musiq(target_tensor)

Distribution metrics
--------------------

FID and Inception Score operate on image collections rather than aligned
individual images. FID compares two directories or an image directory against
supported precomputed dataset statistics. Preprocessing modes and statistics
affect results; consult the metric card and the
`clean-fid documentation <https://github.com/GaParmar/clean-fid>`_ before
comparing results from different implementations.

::

    fid = pyiqa.create_metric('fid', device=device)
    score = fid('./generated_images', './reference_images')

Command-line interface
----------------------

Use ``pyiqa -ls`` to list available metrics. For a single NR image, run
``pyiqa musiq -t image.png``. For an FR metric, provide both paths, for example
``pyiqa lpips -t distorted.png -r reference.png``. See :doc:`gmad` for
candidate-pool comparisons of two NR metrics.