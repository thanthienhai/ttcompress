"""pytest scripts/offline/test_offline_helpers.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from common import cv_r2_w, design_matrix, fit_ridge, fit_wridge  # noqa: E402
from a4_a7_eval_stats import signflip_p, wild_ci  # noqa: E402
from ttcompress.attribution import cv_r2  # noqa: E402


def test_weighted_ridge_matches_pipeline_ridge():
    rng = np.random.default_rng(0)
    M = rng.random((64, 10)) < 0.4
    y = rng.random(64)
    coef = fit_ridge(design_matrix(M), y, 1.0)
    b0, w = fit_wridge(M.astype(float), y, np.ones(10))
    assert abs(coef[0] - b0) < 1e-10 and np.abs(coef[1:] - w).max() < 1e-10
    assert abs(cv_r2(M, y, 1.0) - cv_r2_w(lambda t: M.astype(float), y, np.ones(10))) < 1e-10


def test_signflip_and_wild():
    rng = np.random.default_rng(1)
    cl = np.repeat(np.arange(12), 5)
    d = rng.normal(0.3, 0.1, size=60)
    assert signflip_p(d, list(cl), 0.0, 'greater') < 0.01      # clearly > 0
    assert signflip_p(d, list(cl), 0.6, 'less') < 0.01         # clearly < 0.6
    lo, hi = wild_ci(d, list(cl))
    assert lo < d.mean() < hi
