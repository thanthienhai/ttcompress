"""A2 (sparse vs dense refit at equal mask counts) and A3 (K curve), primary reader, F1 target.

Beta is refit with the pipeline's ridge (ttcompress.attribution.fit_ridge) and the source's selected alpha.
Docs: informative ones (F1 varies across masks). "Answer chunk ranked first" = the top-beta chunk contains a
gold answer string (evaluate.py / pruner_training definition), over docs with >= 1 answer chunk; a fit whose
beta is constant (no outcome variation in the subsample) counts as a miss and is tallied as degenerate.

A2: (i) all masks, (ii) sparse masks (keep <= 2 chunks; n_s of them), (iii) n_s masks drawn from the
non-sparse masks (5 draws, averaged), (iv) n_s masks drawn from all masks (5 draws).
A3: K' in {8,16,32,48,all}, 5 random draws without replacement: Spearman with the full-K beta, answer-first,
gold recall@|g|, 5-fold CV R^2 inside the subsample, and R^2 on the held-out (unused) masks.

    python scripts/offline/a2_a3_refit.py [--max-docs-uit 1000]
"""
from __future__ import annotations

import argparse
import os
import time
from multiprocessing import Pool

import numpy as np
from scipy.stats import rankdata

from common import (LABEL_SPLIT, SOURCES, beta_of, fit_ridge, design_matrix, gold_diagnostics,
                    iter_records, raw_path, save, slim, source_alpha)
from ttcompress.attribution import cv_r2
from ttcompress.sources import hash_seed

N_REP = 5
KS = (8, 16, 32, 48)
A2_SOURCES = ('vimqa', 'hotpotqa', '2wiki')


def _spearman(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float('nan')
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def _label_metrics(beta, d):
    degenerate = float(np.std(beta) < 1e-12)
    ans = d['ans']
    first = float('nan') if not ans else (0.0 if degenerate else float(int(np.argsort(-beta, kind='stable')[0]) in ans))
    rec = gold_diagnostics(beta.tolist(), d['gold'])['gold_recall_at_g'] if d['gold'] else float('nan')
    if degenerate and d['gold']:
        rec = 0.0
    return {'first': first, 'recall': rec, 'degenerate': degenerate}


def _heldout_r2(M, y, idx, alpha):
    rest = np.setdiff1d(np.arange(len(y)), idx)
    if len(rest) < 2 or np.ptp(y[rest]) < 1e-9:
        return float('nan')
    coef = fit_ridge(design_matrix(M[idx]), y[idx], alpha)
    pred = design_matrix(M[rest]) @ coef
    return float(1 - np.sum((pred - y[rest]) ** 2) / np.sum((y[rest] - y[rest].mean()) ** 2))


def doc_task(args):
    d, alpha, do_a2 = args
    M = np.asarray(d['masks'], dtype=bool)
    y = np.asarray(d['f1'], dtype=float)
    if np.ptp(y) < 1e-9:
        return None
    K = len(y)
    rng = np.random.default_rng(hash_seed('offline:' + d['doc_id']))
    full = beta_of(M, y, alpha)
    out = {'K': K, 'full': _label_metrics(full, d)}
    if do_a2:
        sparse = np.where(M.sum(axis=1) <= 2)[0]
        dense = np.where(M.sum(axis=1) > 2)[0]
        n_s = len(sparse)
        out['n_s'] = n_s
        if n_s >= 2 and len(dense) >= n_s:
            out['sparse'] = _label_metrics(beta_of(M[sparse], y[sparse], alpha), d)
            for name, pool_idx in (('dense_matched', dense), ('any_matched', np.arange(K))):
                reps = [_label_metrics(beta_of(M[i], y[i], alpha), d)
                        for i in (rng.choice(pool_idx, n_s, replace=False) for _ in range(N_REP))]
                out[name] = {k: float(np.nanmean([r[k] for r in reps])) if not all(np.isnan(r[k]) for r in reps)
                             else float('nan') for k in reps[0]}
    kc = {}
    for k in KS + ('all',):
        if k == 'all':
            m = _label_metrics(full, d)
            m.update(spearman=1.0, cv_r2=cv_r2(M, y, alpha), heldout_r2=float('nan'))
            kc['all'] = m
            continue
        if k >= K:
            continue
        reps = []
        for _ in range(N_REP):
            idx = rng.choice(K, k, replace=False)
            b = beta_of(M[idx], y[idx], alpha)
            m = _label_metrics(b, d)
            m['spearman'] = _spearman(b, full)
            m['cv_r2'] = cv_r2(M[idx], y[idx], alpha) if np.ptp(y[idx]) > 1e-9 else float('nan')
            m['heldout_r2'] = _heldout_r2(M, y, idx, alpha)
            reps.append(m)
        kc[str(k)] = {key: (float(np.nanmean(v)) if not np.all(np.isnan(v)) else float('nan'))
                      for key in reps[0] for v in [np.array([r[key] for r in reps])]}
    out['kcurve'] = kc
    return out


def _agg(rows, key):
    res = {}
    sub = [r[key] for r in rows if key in r]
    if not sub:
        return None
    for m in sub[0]:
        v = np.array([s[m] for s in sub], dtype=float)
        v = v[~np.isnan(v)]
        res[m] = float(v.mean()) if len(v) else None
        if m in ('cv_r2', 'heldout_r2', 'spearman'):
            res[m + '_median'] = float(np.median(v)) if len(v) else None
        res['n_' + m] = int(len(v))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-docs-uit', type=int, default=1000)
    ap.add_argument('--max-docs', type=int, default=3000)
    ap.add_argument('--workers', type=int, default=min(16, max(1, (os.cpu_count() or 2) - 4)))
    args = ap.parse_args()
    t0 = time.time()
    out = {}
    with Pool(args.workers) as pool:
        for s in SOURCES:
            alpha = source_alpha(s)
            cap = args.max_docs_uit if s == 'uit_viquad' else args.max_docs
            docs = [slim(r) for r in iter_records(raw_path(s, LABEL_SPLIT[s]), max_docs=cap, seed=0)]
            rows = [r for r in pool.map(doc_task, [(d, alpha, s in A2_SOURCES) for d in docs], chunksize=8) if r]
            res = {'split': LABEL_SPLIT[s], 'alpha': alpha, 'n_docs': len(docs), 'n_informative': len(rows),
                   'mean_K': float(np.mean([r['K'] for r in rows]))}
            if s in A2_SOURCES:
                a2 = [r for r in rows if 'sparse' in r]
                res['a2'] = {'n_docs': len(a2), 'mean_n_s': float(np.mean([r['n_s'] for r in a2])),
                             'n_docs_skipped_n_s_lt_2': sum(1 for r in rows if 'sparse' not in r),
                             **{k: _agg(a2, k) for k in ('full', 'sparse', 'dense_matched', 'any_matched')}}
            res['kcurve'] = {k: _agg([r['kcurve'] for r in rows], k) for k in [str(k) for k in KS] + ['all']}
            out[s] = res
            save('a2_a3_refit.json', out)  # incremental: a hung pool shutdown must not lose results
            print(s, round(time.time() - t0), 's', res.get('a2', {}).get('full'), res.get('a2', {}).get('sparse'),
                  res.get('a2', {}).get('dense_matched'), flush=True)
    print('saved', flush=True)


if __name__ == '__main__':
    main()
