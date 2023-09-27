import pytest
import pyiqa
import torch
from pyiqa.utils import imread2tensor
import pandas as pd
import numpy as np
import os

# absolute and relative tolerance of the difference between our implementation and official results
ATOL = 1e-2
RTOL = 1e-2

# Currently, some metrics have larger differences with official results 
TOL_DICT = {
    'brisque': (1e-2, 8e-2),
    'niqe': (1e-2, 6e-2),
    'pi': (1e-2, 3e-2),
    'ilniqe': (1e-2, 4e-2),
    'musiq': (1e-2, 3e-2),
    'musiq-ava': (1e-2, 3e-2),
    'musiq-koniq': (1e-2, 3e-2),
    'musiq-paq2piq': (1e-2, 3e-2),
    'musiq-spaq': (1e-2, 3e-2),
    'laion_aes': (1e-2, 2e-2),
}

REF_IMG_DIR = './ResultsCalibra/ref_dir'
DIST_IMG_DIR = './ResultsCalibra/dist_dir'
OFFICIAL_RESULT_FILE = './ResultsCalibra/results_official.csv'
CALBR_SUMMARY_FILE = './ResultsCalibra/calibration_summary.csv'

def read_folder(path):
    img_batch = []
    for imgname in sorted(os.listdir(path)):
        imgpath = os.path.join(path, imgname)
        imgtensor = imread2tensor(imgpath)
        img_batch.append(imgtensor)
    return torch.stack(img_batch)


def metrics_with_official_results():
    official_results = pd.read_csv(OFFICIAL_RESULT_FILE).values.tolist()
    result_dict = {}
    for row in official_results:
        result_dict[row[0]] = np.array(row[1:])
    
    return result_dict


@pytest.fixture(scope='module')
def ref_img() -> torch.Tensor:
    return read_folder(REF_IMG_DIR)


@pytest.fixture(scope='module')
def dist_img() -> torch.Tensor:
    return read_folder(DIST_IMG_DIR)



# ==================================== Test metrics ====================================

@pytest.mark.calibration
@pytest.mark.parametrize(
        ("metric_name"),
        [(k) for k in metrics_with_official_results().keys()]
)
def test_match_official_with_given_cases(ref_img, dist_img, metric_name, device):
    official_result = metrics_with_official_results()[metric_name]
    metric = pyiqa.create_metric(metric_name, device=device)
    score = metric(dist_img, ref_img)

    # save results
    cal_sum = pd.read_csv(CALBR_SUMMARY_FILE, index_col='Method') 
    cal_sum.loc[metric_name] = [f'{item:.4f}' for item in official_result.tolist()]
    cal_sum.loc[metric_name + '(ours)'] = [f'{item:.4f}' for item in score.squeeze().cpu().numpy().tolist()]
    cal_sum = cal_sum.sort_values(by=['Method'], ascending=True)
    cal_sum.to_csv(CALBR_SUMMARY_FILE)

    if metric_name in TOL_DICT.keys():
        atol, rtol = TOL_DICT[metric_name]
    else:
        atol, rtol = ATOL, RTOL
    assert torch.allclose(score.squeeze(), torch.from_numpy(official_result).to(score), atol=atol, rtol=rtol), \
            f"Metric {metric_name} results mismatch with official results."


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
@pytest.mark.parametrize(
    ("metric_name"),
    [(k) for k in pyiqa.list_models() if k not in ['ahiq', 'fid', 'vsi', 'clipscore']]
)
def test_cpu_gpu_consistency(metric_name):
    """Test if the metric results are consistent between CPU and GPU.
    ahiq, fid, vsi are not tested because:
        1. ahiq uses random patch sampling;
        2. fid requires directory inputs;
        3. vsi will output NaN with random input.
    """
    x_cpu = torch.rand(1, 3, 224, 224)
    x_gpu = x_cpu.cuda()
    y_cpu = torch.rand(1, 3, 224, 224)
    y_gpu = y_cpu.cuda()

    metric_cpu = pyiqa.create_metric(metric_name, device='cpu')
    metric_gpu = pyiqa.create_metric(metric_name, device='cuda')
    metric_cpu.eval()
    metric_gpu.eval()

    if hasattr(metric_cpu.net, 'test_sample'):
        metric_cpu.net.test_sample = 1
    if hasattr(metric_gpu.net, 'test_sample'):
        metric_gpu.net.test_sample = 1

    score_cpu = metric_cpu(x_cpu, y_cpu)
    score_gpu = metric_gpu(x_gpu, y_gpu)

    assert torch.allclose(score_cpu, score_gpu.cpu(), atol=ATOL, rtol=RTOL), \
        f"Metric {metric_name} results mismatch between CPU and GPU."


@pytest.mark.parametrize(
    ("metric_name"),
    [(k) for k in pyiqa.list_models() if k not in ['pi', 'nrqm', 'fid', 'mad', 'vsi', 'clipscore', 'entropy']]
)
def test_gradient_backward(metric_name, device):
    """Test if the metric can be used in a gradient descent process.
    pi, nrqm and fid are not tested because they are not differentiable.
    mad and vsi give NaN with random input.
    """
    size = (2, 3, 224, 224)
    if 'swin' in metric_name:
        size = (2, 3, 384, 384)

    x = torch.randn(*size).to(device)
    y = torch.randn(*size).to(device)
    x.requires_grad_()

    metric = pyiqa.create_metric(metric_name, as_loss=True, device=device)
    metric.eval()
    if hasattr(metric.net, 'test_sample'):
        metric.net.test_sample = 1

    score = metric(x, y)
    if isinstance(score, tuple):
        score = score[0]
    score.sum().backward()

    assert torch.isnan(x.grad).sum() == 0, f"Metric {metric_name} cannot be used in a gradient descent process."

    if torch.cuda.is_available():
        del x
        del y
        torch.cuda.empty_cache()


@pytest.mark.parametrize(
    ("metric_name"),
    [(k) for k in pyiqa.list_models() if k not in ['fid', 'clipscore']]
)
def test_forward(metric_name, device):
    """Test if the metric can be used in a gradient descent process.
    pi, nrqm and fid are not tested because they are not differentiable.
    mad and vsi give NaN with random input.
    """
    size = (2, 3, 224, 224)
    if 'swin' in metric_name:
        size = (2, 3, 384, 384)

    x = torch.rand(*size).to(device)
    y = torch.rand(*size).to(device)

    metric = pyiqa.create_metric(metric_name, device=device)
    metric.eval()
    if hasattr(metric.net, 'test_sample'):
        metric.net.test_sample = 1
    
    score = metric(x, y)

    if torch.cuda.is_available():
        del x
        del y
        del score
        torch.cuda.empty_cache()