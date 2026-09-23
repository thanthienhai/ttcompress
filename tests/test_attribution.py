import numpy as np

from ttcompress.attribution import (
    ChunkLabels, MaskOutcomes, cv_r2, ensemble_labels, fit_document, fit_position_prior, fit_ridge, design_matrix,
    generate_masks, gold_diagnostics, load_chunk_labels, masks_for_document, num_masks, position_r2, record_paths,
    save_record, select_alpha, zscore,
)
from tests.conftest import make_doc


def test_num_masks_bounds():
    assert num_masks(10) == 64
    assert num_masks(100) == 101
    assert num_masks(1000) == 256


def test_masks_never_full_or_empty_and_follow_keep_rates():
    masks = generate_masks(40, 400, [0.5, 0.25], seed=0)
    assert all(any(m) and not all(m) for m in masks)
    even = np.mean([np.mean(m) for m in masks[0::2]])
    odd = np.mean([np.mean(m) for m in masks[1::2]])
    assert abs(even - 0.5) < 0.05 and abs(odd - 0.25) < 0.05


def test_masks_are_identical_across_readers_for_a_document():
    a = masks_for_document('doc-1', 30, [0.5], 64, 1.0, 256)
    b = masks_for_document('doc-1', 30, [0.5], 64, 1.0, 256)
    c = masks_for_document('doc-2', 30, [0.5], 64, 1.0, 256)
    assert a == b and a != c


def _planted(C=20, K=200, seed=0, noise=0.01):
    rng = np.random.default_rng(seed)
    masks = generate_masks(C, K, [0.5], seed=seed)
    beta = np.zeros(C)
    beta[3], beta[11] = 0.6, 0.3
    y = 0.05 + design_matrix(masks)[:, 1:] @ beta + noise * rng.standard_normal(K)
    return masks, y, beta


def test_ridge_recovers_planted_contributions_even_with_k_below_c_plus_1():
    masks, y, beta = _planted(C=60, K=40)
    coef = fit_ridge(design_matrix(masks), y, alpha=0.3)[1:]
    assert set(np.argsort(-coef)[:2]) == {3, 11}


def test_cv_r2_high_for_additive_signal_and_alpha_selection_runs():
    masks, y, _ = _planted()
    assert cv_r2(masks, y, 1.0) > 0.9
    alpha, scores = select_alpha([(masks, y), (masks, np.zeros(len(y)))], grid=(0.1, 1.0, 10.0))
    assert alpha in scores and np.isfinite(scores[alpha])


def test_zscore_and_uninformative():
    z, ok = zscore([1.0, 2.0, 3.0])
    assert ok and abs(np.mean(z)) < 1e-9
    z, ok = zscore([0.2, 0.2])
    assert not ok and z == [0.0, 0.0]


def test_gold_diagnostics():
    d = gold_diagnostics([0.1, 0.9, 0.0, 0.5], gold=[1, 3])
    assert d['gold_recall_at_g'] == 1.0 and d['gold_mrr'] == 1.0
    d = gold_diagnostics([0.9, 0.1, 0.0, 0.5], gold=[1])
    assert d['gold_recall_at_g'] == 0.0 and d['gold_mrr'] == 1 / 3


def _record(doc_id, y, masks, reader='r1'):
    doc = make_doc([f'c{i}' for i in range(len(masks[0]))], gold=(3, 11), doc_id=doc_id)
    return MaskOutcomes(doc=doc.to_dict(), reader=reader, masks=masks, f1=list(map(float, y)),
                        logprob=list(map(float, -1 + y)), full_f1=1.0)


def test_fit_document_roundtrip_and_ensemble(tmp_path):
    masks, y, _ = _planted()
    lab_a = fit_document(_record('d', y, masks, 'r1'), 'f1', alpha=1.0)
    lab_b = fit_document(_record('d', y[::-1].copy(), masks, 'r2'), 'f1', alpha=1.0)
    assert lab_a.informative and lab_a.diagnostics['gold_recall_at_g'] == 1.0
    path = save_record(lab_a, str(tmp_path))
    (tmp_path / 'summary.json').write_text('{}')
    assert record_paths(str(tmp_path)) == [path]
    assert load_chunk_labels(path).z == lab_a.z
    ens = ensemble_labels([lab_a, lab_b])
    assert ens.reader == 'ensemble(r1+r2)' and len(ens.z) == len(lab_a.z)
    assert fit_document(_record('d', y, masks), 'logprob', alpha=1.0).target == 'logprob'


def test_uninformative_document_is_flagged():
    masks, y, _ = _planted()
    lab = fit_document(_record('d', np.zeros_like(y), masks), 'f1', alpha=1.0)
    assert not lab.informative


def test_pooled_position_prior_detects_a_position_effect():
    labels = []
    for k in range(30):
        z = [2.0 if i == 0 else -0.2 for i in range(10)]
        labels.append(ChunkLabels(doc={'gold_chunks': [0]}, reader='r', target='f1', beta=z, intercept=0.0, z=z,
                                  informative=True, alpha=1.0, cv_r2=1.0, full_f1=1.0))
    prior = fit_position_prior(labels)
    assert prior[0] > 1.5 and position_r2(labels, prior) > 0.99
