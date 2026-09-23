"""Stage A -- utility attribution by random chunk ablation + ridge surrogate.

For one document with C chunks and one reader:

    y_k = u(reader(question, context kept by mask m_k))      k = 1..K
    y_k ~= beta_0 + sum_c beta_c * m_k[c]                     (ridge, intercept unpenalized)

u is the downstream token F1 (the method) or the teacher-forced answer
log-prob (the AT2/ContextCite target, ablation only). beta_c is the chunk's
utility attribution; within-document z-scored beta is the pseudo-label the
pruner is distilled on (Stage B, pruner_training.py).

Masks are i.i.d. Bernoulli(p) per chunk, p cycling over `keep_rates`
(ContextCite uses p=0.5; mixing in sparser rates keeps the surrogate honest
near the 1/4-1/8 budgets the pruner is deployed at). Masks are seeded by the
document id, so every reader sees the exact same masks for a document --
which is what makes per-reader labels comparable and ensemblable.

Unlike the previous pipeline, K < C+1 is allowed: with alpha > 0 the ridge
system is well-posed, and label quality is MEASURED instead of assumed
(held-out R^2 of the surrogate, gold-chunk recall of the attribution).
"""
from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .sources import hash_seed

# ---------------------------------------------------------------------------
# masks
# ---------------------------------------------------------------------------

def num_masks(C: int, k_min: int = 64, k_per_chunk: float = 1.0, k_max: int = 256) -> int:
    """K for a document with C chunks: at least k_min, about k_per_chunk
    masks per ridge parameter, capped at k_max (cost control)."""
    return int(min(k_max, max(k_min, math.ceil(k_per_chunk * (C + 1)))))


def generate_masks(C: int, K: int, keep_rates: Sequence[float], seed: int) -> List[List[bool]]:
    """K Bernoulli masks over C chunks; every mask keeps >= 1 chunk and
    drops >= 1 chunk (when C > 1), so no mask is the full or empty context."""
    if C <= 0:
        raise ValueError(f"C must be > 0; got {C}")
    rng = random.Random(seed)
    masks = []
    for k in range(K):
        p = keep_rates[k % len(keep_rates)]
        m = [rng.random() < p for _ in range(C)]
        if not any(m):
            m[rng.randrange(C)] = True
        if C > 1 and all(m):
            m[rng.randrange(C)] = False
        masks.append(m)
    return masks


def masks_for_document(doc_id: str, C: int, keep_rates: Sequence[float], k_min: int, k_per_chunk: float,
                       k_max: int) -> List[List[bool]]:
    return generate_masks(C, num_masks(C, k_min, k_per_chunk, k_max), keep_rates, seed=hash_seed(f'masks:{doc_id}'))


# ---------------------------------------------------------------------------
# ridge surrogate
# ---------------------------------------------------------------------------

def design_matrix(masks) -> np.ndarray:
    m = np.asarray(masks, dtype=float)
    return np.hstack([np.ones((m.shape[0], 1)), m])


def fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    penalty = alpha * np.eye(X.shape[1])
    penalty[0, 0] = 0.0
    return np.linalg.lstsq(X.T @ X + penalty, X.T @ y, rcond=None)[0]


def _folds(n: int, n_folds: int, seed: int) -> List[np.ndarray]:
    idx = np.random.default_rng(seed).permutation(n)
    return np.array_split(idx, min(n_folds, n))


def cv_predictions(masks, outcomes, alpha: float, n_folds: int = 5, seed: int = 0) -> np.ndarray:
    """Out-of-fold predictions of the surrogate for every mask."""
    X, y = design_matrix(masks), np.asarray(outcomes, dtype=float)
    preds = np.full(len(y), np.nan)
    if len(y) < 4:
        return preds
    for test_idx in _folds(len(y), n_folds, seed):
        train_idx = np.setdiff1d(np.arange(len(y)), test_idx)
        preds[test_idx] = X[test_idx] @ fit_ridge(X[train_idx], y[train_idx], alpha)
    return preds


def cv_mse(masks, outcomes, alpha: float, n_folds: int = 5, seed: int = 0) -> float:
    preds = cv_predictions(masks, outcomes, alpha, n_folds, seed)
    y = np.asarray(outcomes, dtype=float)
    ok = ~np.isnan(preds)
    return float(np.mean((preds[ok] - y[ok]) ** 2)) if ok.any() else float('nan')


def cv_r2(masks, outcomes, alpha: float, n_folds: int = 5, seed: int = 0) -> float:
    """Held-out R^2 of the linear surrogate: how much of the reader's
    outcome variation across masks an additive-per-chunk model explains.
    Low on multi-hop is expected (interactions) and is itself a finding."""
    y = np.asarray(outcomes, dtype=float)
    if np.var(y) == 0:
        return float('nan')
    preds = cv_predictions(masks, outcomes, alpha, n_folds, seed)
    ok = ~np.isnan(preds)
    return float(1 - np.sum((preds[ok] - y[ok]) ** 2) / np.sum((y[ok] - y[ok].mean()) ** 2))


def select_alpha(documents: Sequence[Tuple[Sequence, Sequence]], grid: Sequence[float] = (0.1, 0.3, 1.0, 3.0, 10.0),
                 n_folds: int = 5, seed: int = 0) -> Tuple[float, Dict[float, float]]:
    """One global alpha: argmin of mean per-document CV MSE over dev docs
    (documents whose outcomes never vary carry no information and are skipped)."""
    scores = {}
    informative = [(m, o) for m, o in documents if np.var(o) > 0]
    for a in grid:
        vals = [cv_mse(m, o, a, n_folds, seed) for m, o in informative]
        vals = [v for v in vals if not np.isnan(v)]
        scores[a] = float(np.mean(vals)) if vals else float('inf')
    return min(grid, key=lambda a: scores[a]), scores


def bootstrap_beta_ci(masks, outcomes, alpha: float, n_boot: int = 200, seed: int = 0):
    X, y = design_matrix(masks), np.asarray(outcomes, dtype=float)
    rng = np.random.default_rng(seed)
    boots = np.stack([fit_ridge(X[idx], y[idx], alpha) for idx in rng.integers(0, len(y), size=(n_boot, len(y)))])
    return np.quantile(boots, 0.025, axis=0), np.quantile(boots, 0.975, axis=0)


# ---------------------------------------------------------------------------
# label post-processing and diagnostics
# ---------------------------------------------------------------------------

def zscore(beta_chunks: Sequence[float]) -> Tuple[List[float], bool]:
    """Within-document z-score; (zeros, False) when beta is constant."""
    b = np.asarray(beta_chunks, dtype=float)
    sd = b.std()
    if sd < 1e-12:
        return [0.0] * len(b), False
    return ((b - b.mean()) / sd).tolist(), True


def gold_diagnostics(beta_chunks: Sequence[float], gold: Sequence[int]) -> Dict[str, float]:
    """Does the attribution find the evidence? recall of gold chunks among
    the top-|gold| chunks by beta, and the reciprocal rank of the best gold
    chunk. On single-hop, recall@|gold| ~= 1 means beta has collapsed onto
    the answer-span label (METHOD_SPEC.md risk #2)."""
    if not gold:
        return {'gold_recall_at_g': float('nan'), 'gold_mrr': float('nan')}
    order = list(np.argsort(-np.asarray(beta_chunks, dtype=float), kind='stable'))
    top = set(order[:len(gold)])
    ranks = [order.index(g) + 1 for g in gold]
    return {'gold_recall_at_g': sum(g in top for g in gold) / len(gold), 'gold_mrr': 1.0 / min(ranks)}


POSITION_BINS = 10


def position_bin(i: int, C: int, bins: int = POSITION_BINS) -> int:
    return min(bins - 1, int(bins * i / max(1, C)))


def fit_position_prior(labels: Sequence['ChunkLabels'], bins: int = POSITION_BINS) -> List[float]:
    """Mean z-scored label per relative-position bin, pooled over documents.
    A pooled (not per-document) fit: within one document the label is a
    spike on the evidence, and a per-document regression on a U-shaped
    position score always returns ~0 regardless of how much position matters."""
    sums, counts = np.zeros(bins), np.zeros(bins)
    for lab in labels:
        if not lab.informative:
            continue
        for i, z in enumerate(lab.z):
            b = position_bin(i, len(lab.z), bins)
            sums[b] += z
            counts[b] += 1
    return (sums / np.maximum(counts, 1)).tolist()


def position_r2(labels: Sequence['ChunkLabels'], prior: Sequence[float]) -> float:
    """Share of z-label variance the pooled position prior explains."""
    ys, preds = [], []
    for lab in labels:
        if lab.informative:
            ys.extend(lab.z)
            preds.extend(prior[position_bin(i, len(lab.z), len(prior))] for i in range(len(lab.z)))
    ys, preds = np.asarray(ys), np.asarray(preds)
    if len(ys) == 0 or ys.var() == 0:
        return float('nan')
    return float(1 - ((ys - preds) ** 2).sum() / ((ys - ys.mean()) ** 2).sum())


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------

@dataclass
class MaskOutcomes:
    """Expensive reader output for one (document, reader); fit is CPU-only on top."""
    doc: Dict                        # QADocument.to_dict()
    reader: str
    masks: List[List[bool]]
    f1: List[float]
    logprob: Optional[List[float]] = None
    full_f1: float = float('nan')
    full_logprob: Optional[float] = None
    full_answer: str = ''
    seconds: float = 0.0


@dataclass
class ChunkLabels:
    doc: Dict
    reader: str                       # a reader tag, or 'ensemble(a+b+...)'
    target: str                       # 'f1' | 'logprob'
    beta: List[float]                 # per chunk (intercept stored separately)
    intercept: float
    z: List[float]                    # within-document z-score of beta
    informative: bool                 # False when outcomes never varied -> no training signal
    alpha: float
    cv_r2: float
    full_f1: float
    diagnostics: Dict[str, float] = field(default_factory=dict)
    ci_lo: Optional[List[float]] = None
    ci_hi: Optional[List[float]] = None


def _write(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(asdict(obj), f, ensure_ascii=False)
    os.replace(tmp, path)  # atomic: a killed shard never leaves a half-written record behind


def save_record(obj, out_dir: str) -> str:
    path = os.path.join(out_dir, f"{obj.doc['doc_id']}.json")
    _write(obj, path)
    return path


def record_paths(directory: str) -> List[str]:
    """Per-document record files in a measure/fit dir (summary/config files excluded)."""
    import glob
    skip = {'summary.json', 'measure_config.json'}
    return sorted(p for p in glob.glob(os.path.join(directory, '*.json')) if os.path.basename(p) not in skip)


def load_mask_outcomes(path: str) -> MaskOutcomes:
    with open(path, encoding='utf-8') as f:
        return MaskOutcomes(**json.load(f))


def load_chunk_labels(path: str) -> ChunkLabels:
    with open(path, encoding='utf-8') as f:
        return ChunkLabels(**json.load(f))


def fit_document(rec: MaskOutcomes, target: str, alpha: float, n_boot: int = 0) -> ChunkLabels:
    outcomes = rec.f1 if target == 'f1' else rec.logprob
    if outcomes is None:
        raise ValueError(f"{rec.doc['doc_id']}: no {target!r} outcomes were measured")
    X = design_matrix(rec.masks)
    y = np.asarray(outcomes, dtype=float)
    informative = bool(np.var(y) > 0)
    coef = fit_ridge(X, y, alpha)
    beta = coef[1:].tolist()
    z, varied = zscore(beta)
    labels = ChunkLabels(
        doc=rec.doc, reader=rec.reader, target=target, beta=beta, intercept=float(coef[0]), z=z,
        informative=informative and varied, alpha=alpha,
        cv_r2=cv_r2(rec.masks, outcomes, alpha) if informative else float('nan'),
        full_f1=rec.full_f1, diagnostics=gold_diagnostics(beta, rec.doc['gold_chunks']),
    )
    if n_boot:
        lo, hi = bootstrap_beta_ci(rec.masks, outcomes, alpha, n_boot)
        labels.ci_lo, labels.ci_hi = lo[1:].tolist(), hi[1:].tolist()
    return labels


def ensemble_labels(per_reader: Sequence[ChunkLabels]) -> ChunkLabels:
    """Reader-agnostic label: mean of the readers' within-document z-scores
    (only readers for which the document was informative contribute)."""
    ok = [lab for lab in per_reader if lab.informative]
    base = per_reader[0]
    C = len(base.beta)
    if any(len(lab.beta) != C for lab in per_reader):
        raise ValueError(f"{base.doc['doc_id']}: readers disagree on the chunk count")
    if ok:
        mean_z = np.mean([lab.z for lab in ok], axis=0)
        z, varied = zscore(mean_z)
    else:
        mean_z, z, varied = np.zeros(C), [0.0] * C, False
    return ChunkLabels(
        doc=base.doc, reader='ensemble(' + '+'.join(lab.reader for lab in per_reader) + ')', target=base.target,
        beta=mean_z.tolist(), intercept=0.0, z=z, informative=bool(ok) and varied, alpha=base.alpha,
        cv_r2=float(np.nanmean([lab.cv_r2 for lab in ok])) if ok else float('nan'),
        full_f1=float(np.nanmean([lab.full_f1 for lab in per_reader])),
        diagnostics={**gold_diagnostics(mean_z.tolist(), base.doc['gold_chunks']), 'n_readers_informative': len(ok)},
    )
