from scipy import stats
from scipy.optimize import curve_fit
import numpy as np
from pyiqa.utils.registry import METRIC_REGISTRY


def fit_curve(x, y, curve_type='logistic_4params'):
    r'''Fit the scale of predict scores to MOS scores using logistic regression suggested by VQEG.

    The function with 4 params is more commonly used.
    The 5 params function takes from DBCNN:
        - https://github.com/zwx8981/DBCNN/blob/master/dbcnn/tools/verify_performance.m

    '''
    assert curve_type in [
        'logistic_4params', 'logistic_5params'], f'curve type should be in [logistic_4params, logistic_5params], but got {curve_type}.'

    betas_init_4params = [np.max(y), np.min(y), np.mean(x), np.std(x) / 4.]

    def logistic_4params(x, beta1, beta2, beta3, beta4):
        yhat = (beta1 - beta2) / (1 + np.exp(- (x - beta3) / beta4)) + beta2
        return yhat

    betas_init_5params = [10, 0, np.mean(y), 0.1, 0.1]

    def logistic_5params(x, beta1, beta2, beta3, beta4, beta5):
        logistic_part = 0.5 - 1. / (1 + np.exp(beta2 * (x - beta3)))
        yhat = beta1 * logistic_part + beta4 * x + beta5
        return yhat

    if curve_type == 'logistic_4params':
        logistic = logistic_4params
        betas_init = betas_init_4params
    elif curve_type == 'logistic_5params':
        logistic = logistic_5params
        betas_init = betas_init_5params

    betas, _ = curve_fit(logistic, x, y, p0=betas_init)
    yhat = logistic(x, *betas)
    return yhat


@METRIC_REGISTRY.register()
def calculate_rmse(x, y, fit_scale=None, eps=1e-8):
    if fit_scale is not None:
        x = fit_curve(x, y, fit_scale)
    return np.sqrt(np.mean((x - y) ** 2) + eps)


@METRIC_REGISTRY.register()
def calculate_plcc(x, y, fit_scale=None):
    if fit_scale is not None:
        x = fit_curve(x, y, fit_scale)
    return stats.pearsonr(x, y)[0]


@METRIC_REGISTRY.register()
def calculate_srcc(x, y):
    return stats.spearmanr(x, y)[0]


@METRIC_REGISTRY.register()
def calculate_krcc(x, y):
    return stats.kendalltau(x, y)[0]
