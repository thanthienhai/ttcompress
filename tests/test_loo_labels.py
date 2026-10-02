"""LOO labeling baseline (--mask-scheme loo / --estimator loo) and the equal-cost ridge baseline (--max-masks)."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from ttcompress.attribution import (
    MaskOutcomes, fit_document_loo, generate_masks, is_loo_masks, load_chunk_labels, loo_masks, save_record,
    subsample_masks, zscore,
)
from tests.conftest import TINY_CAUSAL_LM, make_doc

ENV = {**os.environ, 'HF_HUB_OFFLINE': '1', 'PYTHONIOENCODING': 'utf-8'}


def test_loo_masks_shape_and_content():
    m = loo_masks(4)
    assert m == [[False, True, True, True], [True, False, True, True],
                 [True, True, False, True], [True, True, True, False]]
    assert is_loo_masks(m)
    assert loo_masks(1) == [[False]] and is_loo_masks([[False]])   # the one allowed empty context
    assert not is_loo_masks(m[::-1]) and not is_loo_masks(m[:3]) and not is_loo_masks([])
    assert not is_loo_masks(generate_masks(4, 4, [0.5], seed=0))
    with pytest.raises(ValueError):
        loo_masks(0)


def _loo_record(f1, full_f1=1.0, gold=(1,), doc_id='d', logprob=None, full_logprob=None):
    doc = make_doc([f'c{i}' for i in range(len(f1))], gold=gold, doc_id=doc_id)
    return MaskOutcomes(doc=doc.to_dict(), reader='r', masks=loo_masks(len(f1)), f1=list(f1), full_f1=full_f1,
                        logprob=logprob, full_logprob=full_logprob)


def test_fit_document_loo_exact_deltas_and_z():
    lab = fit_document_loo(_loo_record([1.0, 0.2, 0.9, 1.0], full_f1=1.0))
    assert np.allclose(lab.beta, [0.0, 0.8, 0.1, 0.0])
    assert lab.intercept == 1.0 and lab.estimator == 'loo' and lab.informative
    assert np.allclose(lab.z, zscore([0.0, 0.8, 0.1, 0.0])[0])
    assert np.isnan(lab.alpha) and np.isnan(lab.cv_r2)
    assert lab.diagnostics['gold_recall_at_g'] == 1.0 and lab.diagnostics['gold_mrr'] == 1.0
    lp = fit_document_loo(_loo_record([1.0, 0.2], logprob=[-1.0, -3.0], full_logprob=-0.5), 'logprob')
    assert np.allclose(lp.beta, [0.5, 2.5]) and lp.intercept == -0.5 and lp.target == 'logprob'


def test_fit_document_loo_uninformative_and_rejects_non_loo_masks():
    lab = fit_document_loo(_loo_record([0.5, 0.5, 0.5], full_f1=1.0))   # every drop costs the same
    assert not lab.informative and lab.z == [0.0, 0.0, 0.0]
    single = fit_document_loo(_loo_record([0.0], full_f1=1.0, gold=(0,)))            # C = 1: one delta, constant
    assert single.beta == [1.0] and not single.informative
    rec = _loo_record([1.0, 0.2, 0.9])
    rec.masks = generate_masks(3, 3, [0.5], seed=0)
    with pytest.raises(ValueError, match='leave-one-out'):
        fit_document_loo(rec)
    with pytest.raises(ValueError, match='logprob'):
        fit_document_loo(_loo_record([1.0, 0.2]), 'logprob')


def _random_record(doc_id, K=64, C=6, seed=0):
    doc = make_doc([f'c{j}' for j in range(C)], gold=(1,), doc_id=doc_id)
    masks = generate_masks(C, K, [0.5], seed=seed)
    f1 = [float(k) for k in range(K)]       # f1 == index: makes the kept subset readable
    return MaskOutcomes(doc=doc.to_dict(), reader='r', masks=masks, f1=f1, logprob=[-x for x in f1], full_f1=1.0)


def test_subsample_is_deterministic_ordered_and_per_document():
    a, b = _random_record('doc-a'), _random_record('doc-b')
    sa = subsample_masks(a, 11, seed=0)
    assert len(sa.masks) == len(sa.f1) == len(sa.logprob) == 11
    idx = [int(x) for x in sa.f1]
    assert idx == sorted(idx) and len(set(idx)) == 11                       # original order, no repeats
    assert sa.masks == [a.masks[i] for i in idx] and sa.logprob == [-float(i) for i in idx]
    assert subsample_masks(a, 11, seed=0) == sa                              # deterministic
    assert subsample_masks(a, 11, seed=1).f1 != sa.f1                        # seed matters
    # independent of other documents: same subset whether b was subsampled first or not
    subsample_masks(b, 11, seed=0)
    assert subsample_masks(a, 11, seed=0) == sa
    assert [int(x) for x in subsample_masks(b, 11, seed=0).f1] != idx         # keyed by doc_id
    assert subsample_masks(a, 64, seed=0) is a and subsample_masks(a, 100, seed=0) is a
    assert len(a.masks) == 64                                                # input untouched


def _measure(tmp_path, *extra, shard=1):
    """Shard 1 of 2 with n=1 is empty: writes measure_config.json without loading the reader."""
    return subprocess.run([sys.executable, 'generate_labels.py', 'measure', '--source', 'vimqa', '--split', 'dev',
                           '--n', '1', '--shard', str(shard), '--num-shards', '2', '--reader-model', TINY_CAUSAL_LM,
                           '--device', 'cpu', '--backend', 'hf', '--k-min', '4', '--k-max', '4',
                           '--max-model-len', '512', '--out-root', str(tmp_path), *extra],
                          capture_output=True, env=ENV, text=True, encoding='utf-8')


def test_measure_config_records_mask_scheme_only_when_not_random(tmp_path):
    out = tmp_path / 'r' / TINY_CAUSAL_LM.replace('/', '--') / 'vimqa_dev'
    assert _measure(tmp_path / 'r').returncode == 0
    assert 'mask_scheme' not in json.loads((out / 'measure_config.json').read_text(encoding='utf-8'))
    # a LOO run cannot write into a random-mask label dir
    bad = _measure(tmp_path / 'r', '--mask-scheme', 'loo')
    assert bad.returncode != 0 and 'use another --out-root' in bad.stderr
    out_loo = tmp_path / 'l' / TINY_CAUSAL_LM.replace('/', '--') / 'vimqa_dev'
    assert _measure(tmp_path / 'l', '--mask-scheme', 'loo').returncode == 0
    assert json.loads((out_loo / 'measure_config.json').read_text(encoding='utf-8'))['mask_scheme'] == 'loo'


def test_measure_loo_writes_leave_one_out_records(tmp_path):
    res = _measure(tmp_path, '--mask-scheme', 'loo', shard=0)
    assert res.returncode == 0, res.stderr
    out = tmp_path / TINY_CAUSAL_LM.replace('/', '--') / 'vimqa_dev'
    (path,) = [p for p in out.glob('*.json') if p.name != 'measure_config.json']
    rec = json.loads(path.read_text(encoding='utf-8'))
    assert is_loo_masks(rec['masks']) and len(rec['masks']) == len(rec['doc']['chunks']) == len(rec['f1'])


def _raw_dir(path, records, scheme=None):
    path.mkdir(parents=True)
    for r in records:
        save_record(r, str(path))
    if scheme is not None:
        (path / 'measure_config.json').write_text(json.dumps({'mask_scheme': scheme} if scheme != 'random' else {}),
                                                  encoding='utf-8')
    return path


def _fit(*args):
    return subprocess.run([sys.executable, 'generate_labels.py', 'fit', *map(str, args)],
                          capture_output=True, text=True, encoding='utf-8', env=ENV)


def test_fit_cli_loo_and_mismatch_refusals(tmp_path):
    loo = _raw_dir(tmp_path / 'loo', [_loo_record([1.0, 0.2, 0.9], doc_id=f'd{i}') for i in range(3)], 'loo')
    rnd = _raw_dir(tmp_path / 'rnd', [_random_record(f'd{i}', seed=i) for i in range(3)], 'random')
    out = tmp_path / 'fit_loo'
    res = _fit('--raw-dir', loo, '--estimator', 'loo', '--out-dir', out)
    assert res.returncode == 0, res.stderr
    labs = [load_chunk_labels(str(p)) for p in out.glob('*.json') if p.name != 'summary.json']
    assert len(labs) == 3 and all(lab.estimator == 'loo' and np.allclose(lab.beta, [0.0, 0.8, 0.1]) for lab in labs)
    summary = json.loads((out / 'summary.json').read_text(encoding='utf-8'))
    assert summary['estimator'] == 'loo' and summary['alpha'] is None and summary['n_informative'] == 3
    # estimator / mask scheme mismatches, and alpha options that mean nothing for LOO
    for args, msg in [(('--raw-dir', rnd, '--estimator', 'loo'), '--mask-scheme loo'),
                      (('--raw-dir', tmp_path / 'nocfg_missing', '--estimator', 'loo'), '--mask-scheme loo'),
                      (('--raw-dir', loo, '--alpha', '1.0'), '--estimator loo'),
                      (('--raw-dir', loo, '--estimator', 'loo', '--alpha', '1.0'), 'no alpha'),
                      (('--raw-dir', rnd, '--alpha-from', loo, '--max-masks', '11'), 'random masks')]:
        bad = _fit(*args, '--out-dir', tmp_path / 'x')
        assert bad.returncode != 0 and msg in bad.stderr, (args, bad.stderr)


def test_fit_cli_max_masks_subsamples_train_and_dev(tmp_path):
    rnd = _raw_dir(tmp_path / 'rnd', [_random_record(f'd{i}', seed=i) for i in range(3)], 'random')
    dev = _raw_dir(tmp_path / 'dev', [_random_record(f'v{i}', seed=10 + i) for i in range(3)])  # no config: random
    full, cut = tmp_path / 'full', tmp_path / 'cut'
    assert _fit('--raw-dir', rnd, '--alpha', '1.0', '--out-dir', full).returncode == 0
    res = _fit('--raw-dir', rnd, '--alpha-from', dev, '--alpha-grid', '0.1,1,10', '--max-masks', '11',
               '--mask-seed', '3', '--out-dir', cut)
    assert res.returncode == 0, res.stderr
    s_full = json.loads((full / 'summary.json').read_text(encoding='utf-8'))
    s_cut = json.loads((cut / 'summary.json').read_text(encoding='utf-8'))
    assert not {'estimator', 'max_masks', 'mask_seed'} & set(s_full)          # default path: keys unchanged
    assert s_cut['estimator'] == 'ridge' and s_cut['max_masks'] == 11 and s_cut['mask_seed'] == 3
    assert s_cut['mean_masks_used'] == 11.0 and s_cut['n_alpha_docs'] == 3
    # the cut fit is exactly the in-process fit on the subsampled record
    from ttcompress.attribution import fit_document, load_mask_outcomes
    rec = load_mask_outcomes(str(rnd / 'd0.json'))
    want = fit_document(subsample_masks(rec, 11, 3), 'f1', s_cut['alpha'])
    assert np.allclose(load_chunk_labels(str(cut / 'd0.json')).beta, want.beta)
    assert not np.allclose(load_chunk_labels(str(full / 'd0.json')).beta, want.beta)
