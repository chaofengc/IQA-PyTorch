from scipy import stats
from pyiqa.utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
def calculate_srcc(x, y):
    return stats.spearmanr(x, y)[0]


@METRIC_REGISTRY.register()
def calculate_plcc(x, y):
    return stats.pearsonr(x, y)[0]


@METRIC_REGISTRY.register()
def calculate_krcc(x, y):
    return stats.kendalltau(x, y)[0]
