import numpy as np

from pyiqa.archs.fid_arch import frechet_distance


def _stats(seed=0, n=64, dim=16):
    feats = np.random.default_rng(seed).standard_normal((n, dim))
    return feats.mean(axis=0), np.cov(feats, rowvar=False)


def test_frechet_distance_of_identical_statistics_is_zero():
    # scipy 1.18 removed the `disp` argument of `scipy.linalg.sqrtm`; passing it
    # made every FID metric raise TypeError before computing anything.
    mu, sigma = _stats()
    assert abs(frechet_distance(mu, sigma, mu, sigma)) < 1e-6


def test_frechet_distance_of_a_mean_shift_is_the_squared_shift():
    # Same covariance, so the trace term vanishes and FID = ||mu1 - mu2||^2.
    mu, sigma = _stats()
    shift = np.full_like(mu, 0.5)
    expected = float(shift @ shift)
    assert np.isclose(
        frechet_distance(mu, sigma, mu + shift, sigma), expected, atol=1e-6
    )
