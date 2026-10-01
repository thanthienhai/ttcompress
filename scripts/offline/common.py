"""Shared helpers for the CPU-only offline reviewer analyses (no reader calls).

Data: results/hf_main_eval (HF dataset thanthienhai/ttcompress-main-eval). Ridge fitting, CV folds,
gold diagnostics, bootstrap and Holm come from the ttcompress package; this module only adds a ridge with
a per-column penalty (needed for interaction terms) solved in the dual, which reduces to
ttcompress.attribution.fit_ridge when every penalty equals alpha.
"""
from __future__ import annotations

import json
import os
import random
import sys
from typing import Dict, Iterator, List, Optional, Sequence

# one BLAS thread per process: dozens of pool workers x default OpenBLAS threads exhaust the Windows page file
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ttcompress.attribution import _folds, design_matrix, fit_ridge, gold_diagnostics  # noqa: E402,F401
from ttcompress.sources import contains_answer  # noqa: E402

DATA = os.path.join(ROOT, 'results', 'hf_main_eval')
OUT = os.path.join(ROOT, 'results', 'offline_reviewer')
PRIMARY = 'Qwen--Qwen3-8B'
SOURCES = ('uit_viquad', 'xquad_vi', 'vimqa', 'hotpotqa', '2wiki')
MULTI = ('vimqa', 'hotpotqa', '2wiki')
# labels were measured on train/dev for three sources; xquad_vi and 2wiki only have the 100 oracle test docs
LABEL_SPLIT = {'uit_viquad': 'train', 'vimqa': 'train', 'hotpotqa': 'train', 'xquad_vi': 'test', '2wiki': 'test'}


def raw_path(source: str, split: str, reader: str = PRIMARY) -> str:
    return os.path.join(DATA, 'labels', 'raw', reader, f'{source}_{split}.jsonl')


def source_alpha(source: str, reader: str = PRIMARY, target: str = 'f1') -> float:
    """The alpha the pipeline selected for this source (train summary; test-only sources reuse their summary)."""
    for split in ('train', 'test'):
        p = os.path.join(DATA, 'labels', 'fit', reader, target, f'{source}_{split}.summary.json')
        if os.path.exists(p):
            with open(p, encoding='utf-8') as f:
                return float(json.load(f)['alpha'])
    return 1.0


def count_lines(path: str) -> int:
    with open(path, 'rb') as f:
        return sum(1 for _ in f)


def iter_records(path: str, max_docs: Optional[int] = None, seed: int = 0) -> Iterator[Dict]:
    """Stream raw label records; with max_docs, a seeded uniform subsample of lines."""
    keep = None
    if max_docs is not None:
        n = count_lines(path)
        if max_docs < n:
            keep = set(random.Random(seed).sample(range(n), max_docs))
    with open(path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if keep is not None and i not in keep:
                continue
            yield json.loads(line)


def answer_chunks(doc: Dict) -> List[int]:
    """Chunks containing a gold answer string (the evaluate.py / pruner_training definition)."""
    return [i for i, c in enumerate(doc['chunks']) if contains_answer(c, doc['answers'])]


def slim(rec: Dict) -> Dict:
    """Drop chunk texts after computing what is needed (keeps worker payloads small)."""
    d = rec['doc']
    return {'doc_id': d['doc_id'], 'source': d['source'], 'hop': d['hop'], 'C': len(d['chunks']),
            # F1 is rounded: identical answers can differ by float jitter (~1e-16), which would make a doc look
            # "informative" and blow up R^2 denominators
            'gold': list(d['gold_chunks']), 'ans': answer_chunks(d), 'masks': rec['masks'],
            'f1': [round(float(v), 9) for v in rec['f1']],
            'logprob': rec.get('logprob')}


# ---------------------------------------------------------------------------
# ridge with per-column penalties (dual form; intercept unpenalized via centering)
# ---------------------------------------------------------------------------

def fit_wridge(X: np.ndarray, y: np.ndarray, pen: np.ndarray):
    """min ||y - b0 - X w||^2 + sum_j pen_j w_j^2. Returns (b0, w). Dual solve: O(n^3), n = #masks."""
    s = 1.0 / np.sqrt(pen)
    Z = X * s
    zm, ym = Z.mean(axis=0), y.mean()
    Zc, yc = Z - zm, y - ym
    a = np.linalg.solve(Zc @ Zc.T + np.eye(len(y)), yc)
    wz = Zc.T @ a
    w = wz * s
    return ym - (X.mean(axis=0) @ w), w


def cv_r2_w(build, y: np.ndarray, pen: np.ndarray, n_folds: int = 5, seed: int = 0) -> float:
    """5-fold CV R^2 with the same folds as ttcompress.attribution.cv_r2. `build(train_idx)` returns the full
    design matrix [K, p] (it may depend on the training fold, e.g. a placebo pair picked in-fold)."""
    n = len(y)
    if np.ptp(y) < 1e-9 or n < 4:
        return float('nan')
    preds = np.full(n, np.nan)
    for test_idx in _folds(n, n_folds, seed):
        train_idx = np.setdiff1d(np.arange(n), test_idx)
        X = build(train_idx)
        b0, w = fit_wridge(X[train_idx], y[train_idx], pen)
        preds[test_idx] = b0 + X[test_idx] @ w
    return float(1 - np.sum((preds - y) ** 2) / np.sum((y - y.mean()) ** 2))


def cv_mse_w(build, y, pen, n_folds=5, seed=0) -> float:
    n = len(y)
    preds = np.full(n, np.nan)
    for test_idx in _folds(n, n_folds, seed):
        train_idx = np.setdiff1d(np.arange(n), test_idx)
        X = build(train_idx)
        b0, w = fit_wridge(X[train_idx], y[train_idx], pen)
        preds[test_idx] = b0 + X[test_idx] @ w
    return float(np.mean((preds - y) ** 2))


def beta_of(masks, y, alpha: float) -> np.ndarray:
    """Per-chunk beta exactly as the pipeline fits it (ttcompress.attribution.fit_ridge)."""
    return fit_ridge(design_matrix(masks), np.asarray(y, dtype=float), alpha)[1:]


def top1_is_answer(beta: np.ndarray, ans: Sequence[int]) -> float:
    if not ans:
        return float('nan')
    return float(int(np.argsort(-beta, kind='stable')[0]) in set(ans))


def save(name: str, obj) -> str:
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name)
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=float)
    return p
