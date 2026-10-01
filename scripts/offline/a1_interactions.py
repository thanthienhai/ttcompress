"""A1: is 1 - R^2 of the additive surrogate noise or misspecification?

(a) determinism: identical masks within a document -> do the reader outcomes differ? (all readers)
(b) 5-fold CV R^2: additive ridge vs + gold-pair interaction(s) vs + top-2-beta-pair interaction (placebo,
    pair picked inside each training fold) vs + all pairwise interactions (separate, CV-chosen penalty).
    Penalties are chosen by mean CV MSE on a tuning set (the dev split where it exists, otherwise the docs
    themselves -- xquad_vi / 2wiki have only test labels).

    python scripts/offline/a1_interactions.py [--max-docs 1000]
"""
from __future__ import annotations

import argparse
import itertools
import os
import time
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

from common import (DATA, LABEL_SPLIT, PRIMARY, SOURCES, beta_of, cv_mse_w, cv_r2_w, fit_wridge, iter_records,
                    raw_path, save, slim, source_alpha)

MAIN_GRID = (0.1, 0.3, 1.0, 3.0, 10.0)
INT_GRID = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)


# ---------------------------------------------------------------------------
# (a) determinism
# ---------------------------------------------------------------------------

def determinism(reader: str) -> dict:
    out = {}
    raw_dir = os.path.join(DATA, 'labels', 'raw', reader)
    for fn in sorted(os.listdir(raw_dir)):
        if not fn.endswith('.jsonl'):
            continue
        st = defaultdict(float)
        for rec in iter_records(os.path.join(raw_dir, fn)):
            groups = defaultdict(list)
            for k, m in enumerate(rec['masks']):
                groups[tuple(m)].append(k)
            dup = [g for g in groups.values() if len(g) > 1]
            st['docs'] += 1
            st['docs_with_dup'] += bool(dup)
            for g in dup:
                st['dup_groups'] += 1
                f = [rec['f1'][k] for k in g]
                st['f1_differs'] += (max(f) - min(f)) > 1e-9
                if rec.get('logprob'):
                    lp = [rec['logprob'][k] for k in g]
                    st['lp_groups'] += 1
                    st['lp_max_absdiff_sum'] += max(lp) - min(lp)
                    st['lp_differs_1e-3'] += (max(lp) - min(lp)) > 1e-3
        out[fn[:-6]] = dict(st)
    return out


# ---------------------------------------------------------------------------
# (b) interaction surrogates
# ---------------------------------------------------------------------------

def _pairs_design(M, pairs):
    if not pairs:
        return M
    return np.hstack([M, np.stack([M[:, i] * M[:, j] for i, j in pairs], axis=1)])


def models_for(d, alpha_main):
    """name -> (build(train_idx) -> X, n_interaction_cols). Built once per doc."""
    M = np.asarray(d['masks'], dtype=float)
    y = np.asarray(d['f1'], dtype=float)
    C = M.shape[1]
    gold = sorted(set(d['gold']))
    gold_pairs = list(itertools.combinations(gold, 2))
    all_pairs = list(itertools.combinations(range(C), 2))
    X_gold = _pairs_design(M, gold_pairs)
    X_all = _pairs_design(M, all_pairs)

    def top2(train_idx):
        b = beta_of(M[train_idx].astype(bool), y[train_idx], alpha_main)
        i, j = sorted(np.argsort(-b, kind='stable')[:2])
        return _pairs_design(M, [(int(i), int(j))])

    models = {'additive': (lambda t: M, 0), 'top2_pair': (top2, 1), 'all_pairs': (lambda t: X_all, len(all_pairs))}
    if gold_pairs:
        models['gold_pairs'] = (lambda t: X_gold, len(gold_pairs))
    return M, y, C, gold_pairs, models


def _pen(C, n_int, a_main, a_int):
    return np.concatenate([np.full(C, a_main), np.full(n_int, a_int)])


def tune_doc(args):
    d, alpha_main = args
    M, y, C, _, models = models_for(d, alpha_main)
    if np.ptp(y) < 1e-9:
        return None
    res = {'additive': {a: cv_mse_w(lambda t: M, y, np.full(C, a)) for a in MAIN_GRID}}
    for name in ('top2_pair', 'all_pairs', 'gold_pairs'):
        if name in models:
            build, n_int = models[name]
            res[name] = {a: cv_mse_w(build, y, _pen(C, n_int, alpha_main, a)) for a in INT_GRID}
    return res


def eval_doc(args):
    d, alpha_main, alpha_int = args
    M, y, C, gold_pairs, models = models_for(d, alpha_main)
    if np.ptp(y) < 1e-9:
        return None
    out = {'doc_id': d['doc_id'], 'C': C, 'K': len(y), 'n_gold': len(set(d['gold']))}
    for name, (build, n_int) in models.items():
        pen = np.full(C, alpha_main) if name == 'additive' else _pen(C, n_int, alpha_main, alpha_int[name])
        out['r2_' + name] = cv_r2_w(build, y, pen)
    if gold_pairs:
        build, n_int = models['gold_pairs']
        X = build(None)
        _, w = fit_wridge(X, y, _pen(C, n_int, alpha_main, alpha_int['gold_pairs']))
        out['gold_int_coef'] = float(np.mean(w[C:]))
        out['gold_main_coef'] = float(np.mean(w[sorted(set(d['gold']))]))
    return out


def run_source(source, max_docs, pool):
    split = LABEL_SPLIT[source]
    alpha_main = source_alpha(source)
    docs = [slim(r) for r in iter_records(raw_path(source, split), max_docs=max_docs, seed=0)]
    tune_split = 'dev' if os.path.exists(raw_path(source, 'dev')) else split
    tune_docs = docs if tune_split == split else [slim(r) for r in iter_records(raw_path(source, 'dev'))]
    tuned = [t for t in pool.map(tune_doc, [(d, alpha_main) for d in tune_docs], chunksize=8) if t]
    chosen, grids = {}, {}
    for name in ('additive', 'top2_pair', 'all_pairs', 'gold_pairs'):
        rows = [t[name] for t in tuned if name in t]
        if not rows:
            continue
        mean = {a: float(np.mean([r[a] for r in rows])) for a in rows[0]}
        grids[name] = mean
        chosen[name] = min(mean, key=mean.get)
    alpha_int = {k: v for k, v in chosen.items() if k != 'additive'}
    rows = [r for r in pool.map(eval_doc, [(d, alpha_main, alpha_int) for d in docs], chunksize=8) if r]

    def m(key, sub=rows):
        v = [r[key] for r in sub if key in r and not np.isnan(r[key])]
        return (float(np.mean(v)) if v else None), len(v)

    summ = {'split': split, 'tune_split': tune_split, 'n_docs': len(docs), 'n_informative': len(rows),
            'alpha_main_pipeline': alpha_main, 'alpha_main_cv_on_tune': chosen.get('additive'),
            'alpha_int': alpha_int, 'cv_mse_grid': grids}
    for name in ('additive', 'gold_pairs', 'top2_pair', 'all_pairs'):
        summ['r2_' + name] = m('r2_' + name)
    # paired gains on the docs that have each model
    for name in ('gold_pairs', 'top2_pair', 'all_pairs'):
        sub = [r for r in rows if 'r2_' + name in r and not np.isnan(r['r2_' + name])]
        if sub:
            g = np.array([r['r2_' + name] - r['r2_additive'] for r in sub])
            summ['gain_' + name] = {'mean': float(g.mean()), 'se': float(g.std(ddof=1) / np.sqrt(len(g))),
                                    'share_positive': float((g > 0).mean()), 'n': len(g)}
    co = [r for r in rows if 'gold_int_coef' in r]
    if co:
        ic = np.array([r['gold_int_coef'] for r in co])
        mc = np.array([r['gold_main_coef'] for r in co])
        summ['gold_int_coef'] = {'mean': float(ic.mean()), 'median': float(np.median(ic)),
                                 'share_positive': float((ic > 0).mean()), 'mean_gold_main_coef': float(mc.mean()),
                                 'n': len(co)}
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-docs', type=int, default=1000, help='train docs per source (test sources: all 100)')
    ap.add_argument('--workers', type=int, default=min(16, max(1, (os.cpu_count() or 2) - 4)))
    ap.add_argument('--skip-determinism', action='store_true')
    args = ap.parse_args()
    t0 = time.time()
    out = {'max_docs': args.max_docs}
    if not args.skip_determinism:
        out['determinism'] = {r: determinism(r) for r in sorted(os.listdir(os.path.join(DATA, 'labels', 'raw')))}
        print('determinism done', round(time.time() - t0), 's', flush=True)
    with Pool(args.workers) as pool:
        out['surrogates'] = {}
        for s in SOURCES:
            out['surrogates'][s] = run_source(s, args.max_docs, pool)
            print(s, {k: v for k, v in out['surrogates'][s].items() if k.startswith(('r2_', 'gain_'))},
                  round(time.time() - t0), 's', flush=True)
    print(save('a1_interactions.json', out))


if __name__ == '__main__':
    main()
