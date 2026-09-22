import tempfile

import numpy as np
import pytest

from ttcompress.outcome_labels import (
    DocumentLabels,
    MaskOutcomeRecord,
    apply_mask,
    bootstrap_beta_ci,
    cv_mse,
    estimate_chunk_contributions,
    fit_ridge,
    generate_masks,
    load_document_labels,
    load_mask_outcomes,
    low_confidence_chunks,
    save_document_labels,
    save_mask_outcomes,
    select_alpha,
)


def test_generate_masks_shape_and_keep_count():
    C, K = 16, 20
    masks = generate_masks(C, K, ratios=(4, 8), seed=1)
    assert len(masks) == K
    for i, mask in enumerate(masks):
        assert len(mask) == C
        expected_ratio = (4, 8)[i % 2]
        assert sum(mask) == round(C / expected_ratio)


def test_generate_masks_keep_at_least_one():
    masks = generate_masks(C=3, K=4, ratios=(8,), seed=0)
    for mask in masks:
        assert sum(mask) >= 1


def test_apply_mask_preserves_order():
    chunks = ['a', 'b', 'c', 'd']
    mask = [False, True, False, True]
    assert apply_mask(chunks, mask) == 'b d'


def test_apply_mask_length_mismatch_raises():
    with pytest.raises(ValueError):
        apply_mask(['a', 'b'], [True])


def _synthetic_document(C=10, K=60, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    true_beta = rng.normal(0, 0.1, size=C + 1)
    true_beta[0] = 0.3  # intercept
    masks = generate_masks(C, K, seed=seed)
    X = np.hstack([np.ones((K, 1)), np.asarray(masks, dtype=float)])
    y = X @ true_beta + rng.normal(0, noise, size=K)
    return masks, y.tolist(), true_beta


def test_ridge_recovers_known_coefficients_noiseless():
    masks, outcomes, true_beta = _synthetic_document(C=8, K=80, noise=0.0, seed=42)
    beta = estimate_chunk_contributions(masks, outcomes, alpha=1e-6)
    np.testing.assert_allclose(beta, true_beta, atol=1e-3)


def test_ridge_underdetermined_raises():
    masks, outcomes, _ = _synthetic_document(C=20, K=5, seed=1)
    with pytest.raises(ValueError):
        estimate_chunk_contributions(masks, outcomes, alpha=1.0)


def test_cv_mse_runs_and_is_finite_with_enough_data():
    masks, outcomes, _ = _synthetic_document(C=6, K=40, noise=0.01, seed=2)
    mse = cv_mse(masks, outcomes, alpha=1.0, n_folds=5, seed=0)
    assert mse == mse  # not NaN
    assert mse >= 0


def test_select_alpha_picks_from_grid():
    docs = [_synthetic_document(C=6, K=40, noise=0.02, seed=s)[:2] for s in range(3)]
    grid = (0.01, 0.1, 1.0, 10.0)
    alpha = select_alpha(docs, alpha_grid=grid, n_folds=4, seed=0)
    assert alpha in grid


def test_bootstrap_ci_contains_point_estimate():
    masks, outcomes, _ = _synthetic_document(C=6, K=50, noise=0.02, seed=3)
    beta_point, lo, hi = bootstrap_beta_ci(masks, outcomes, alpha=1.0, n_boot=200, seed=0)
    assert (lo <= beta_point + 1e-9).all()
    assert (beta_point - 1e-9 <= hi).all()


def test_low_confidence_chunks_flags_wide_ci():
    lo = np.array([0.0, 0.1, -0.3])
    hi = np.array([0.0, 0.15, 0.4])
    flags = low_confidence_chunks(lo, hi, width_threshold=0.2)
    assert list(flags) == [False, True]


def test_mask_outcomes_roundtrip():
    record = MaskOutcomeRecord(
        doc_id='doc_0', chunks=['a', 'b', 'c'], positions=[0, 1, 2],
        question='Q?', reference_answer='A', masks=[[True, False, True], [False, True, False]],
        outcomes=[0.5, 0.2], metadata={'title': 'T'},
    )
    with tempfile.TemporaryDirectory() as d:
        path = save_mask_outcomes(record, d)
        loaded = load_mask_outcomes(path)
    assert loaded == record


def test_document_labels_roundtrip():
    labels = DocumentLabels(
        doc_id='doc_0', chunks=['a', 'b'], num_chunks=2, positions=[0, 1],
        beta=[0.3, 0.1, -0.05], ci_lo=[0.2, 0.0, -0.1], ci_hi=[0.4, 0.2, 0.0],
        low_confidence=[False, True], alpha=1.0, metadata={'title': 'T'},
    )
    with tempfile.TemporaryDirectory() as d:
        path = save_document_labels(labels, d)
        loaded = load_document_labels(path)
    assert loaded == labels
