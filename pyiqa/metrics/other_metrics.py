from scipy import stats
import numpy as np
from pyiqa.utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
def calculate_2afc_score(d0, d1, gts, **kwargs):
    scores = (d0 < d1) * (1 - gts) + (d0 > d1) * gts + (d0 == d1) * 0.5
    return np.mean(scores)
