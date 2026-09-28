"""Answer scoring, selection scoring, and cluster-bootstrap statistics.

token F1 / normalization are SQuAD-style and keep Vietnamese tone marks
(syllable-level overlap), unchanged from the previous pipeline so numbers
stay comparable. Every answer metric takes the max over gold aliases.

Statistics: arms are always evaluated on the SAME documents, so comparisons
use a PAIRED cluster bootstrap of the per-document difference -- overlapping
marginal CIs say nothing about a paired difference
(RUN_REPORT_2026-09-23.md review).
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def normalize_answer(text: str) -> List[str]:
    """Lowercase, strip punctuation and English articles, split on whitespace."""
    text = unicodedata.normalize('NFC', text or '').lower()
    text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
    return [t for t in text.split() if t not in ('a', 'an', 'the')]


def _f1(pred_tokens: List[str], ref_tokens: List[str]) -> float:
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)
    common = Counter(pred_tokens) & Counter(ref_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(pred_tokens), overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def token_f1(prediction: str, answers: Sequence[str]) -> float:
    p = normalize_answer(prediction)
    return max((_f1(p, normalize_answer(a)) for a in answers), default=0.0)


def exact_match(prediction: str, answers: Sequence[str]) -> float:
    p = normalize_answer(prediction)
    return float(any(p == normalize_answer(a) for a in answers))


def answer_recall(prediction: str, answers: Sequence[str]) -> float:
    """Fraction of the gold answer's tokens present in the prediction --
    robust to a verbose reader that answers in a full sentence."""
    p = Counter(normalize_answer(prediction))
    best = 0.0
    for a in answers:
        r = normalize_answer(a)
        if r:
            best = max(best, sum((Counter(r) & p).values()) / len(r))
    return best


def gold_chunk_recall(kept: Sequence[int], gold: Sequence[int]) -> float:
    """Fraction of gold (needle / supporting) chunks the selection kept."""
    if not gold:
        return float('nan')
    kept_set = set(kept)
    return sum(g in kept_set for g in gold) / len(gold)


# ---------------------------------------------------------------------------
# ranking metrics (reader-free; used for pruner model selection on dev labels)
# ---------------------------------------------------------------------------

def ndcg_at_k(scores: Sequence[float], gains: Sequence[float], k: int) -> float:
    gains = np.clip(np.asarray(gains, dtype=float), 0.0, None)
    order = np.argsort(-np.asarray(scores, dtype=float), kind='stable')[:k]
    ideal = np.sort(gains)[::-1][:k]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    idcg = float((ideal * discounts[:len(ideal)]).sum())
    if idcg <= 0:
        return float('nan')
    return float((gains[order] * discounts[:len(order)]).sum() / idcg)


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    from scipy.stats import spearmanr
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float('nan')
    return float(spearmanr(a, b).correlation)


# ---------------------------------------------------------------------------
# cluster bootstrap
# ---------------------------------------------------------------------------

def _cluster_members(clusters: Sequence) -> List[np.ndarray]:
    groups: Dict[object, List[int]] = {}
    for i, c in enumerate(clusters):
        groups.setdefault(c, []).append(i)
    return [np.asarray(v, dtype=int) for v in groups.values()]


def _boot_means(values: np.ndarray, clusters: Optional[Sequence], n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if clusters is None:
        idx = rng.integers(0, len(values), size=(n_boot, len(values)))
        return values[idx].mean(axis=1)
    members = _cluster_members(clusters)
    k = len(members)
    sums = np.array([values[m].sum() for m in members])
    sizes = np.array([len(m) for m in members], dtype=float)
    draws = rng.integers(0, k, size=(n_boot, k))
    return sums[draws].sum(axis=1) / sizes[draws].sum(axis=1)


def bootstrap_mean_ci(values: Sequence[float], clusters: Optional[Sequence] = None, n_boot: int = 10000,
                      ci: float = 0.95, seed: int = 42) -> Tuple[float, Optional[float], Optional[float]]:
    """(mean, lo, hi); NaN values are dropped (with their cluster labels)."""
    vals = np.asarray(values, dtype=float)
    keep = ~np.isnan(vals)
    vals = vals[keep]
    cl = None if clusters is None else [c for c, k in zip(clusters, keep) if k]
    if len(vals) < 2:
        return (float(vals.mean()) if len(vals) else float('nan')), None, None
    boot = _boot_means(vals, cl, n_boot, seed)
    lo, hi = np.quantile(boot, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(vals.mean()), float(lo), float(hi)


def paired_bootstrap_diff(a: Sequence[float], b: Sequence[float], clusters: Optional[Sequence] = None,
                          n_boot: int = 10000, ci: float = 0.95, seed: int = 42) -> Dict[str, Optional[float]]:
    """mean(a - b) with a paired cluster-bootstrap CI and a two-sided
    bootstrap p-value (share of resamples on the other side of 0, x2; +1
    smoothed so it is never exactly 0 -- the resolution is 2 / (n_boot + 1))."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    mean, lo, hi = bootstrap_mean_ci(diff, clusters, n_boot, ci, seed)
    keep = ~np.isnan(diff)
    if keep.sum() < 2:
        return {'diff': mean, 'ci95': [lo, hi], 'p': None, 'n': int(keep.sum())}
    cl = None if clusters is None else [c for c, k in zip(clusters, keep) if k]
    boot = _boot_means(diff[keep], cl, n_boot, seed)
    p = 2 * (min((boot <= 0).sum(), (boot >= 0).sum()) + 1) / (len(boot) + 1)
    return {'diff': mean, 'ci95': [lo, hi], 'p': float(min(1.0, p)), 'n': int(keep.sum())}


def directional_p(boot: np.ndarray, kind: str, margin: float = 0.0) -> float:
    """One-sided bootstrap p-value (+1 smoothed) of a pre-registered claim
    about a difference d, from its bootstrap distribution `boot` (NaNs dropped):
      superiority      H0: d <= 0
      noninferiority   H0: d <= -margin
      equivalence      TOST, H0: |d| >= margin (the larger of the two one-sided p's)."""
    boot = boot[~np.isnan(boot)]
    n = len(boot) + 1
    if kind == 'superiority':
        return float(((boot <= 0).sum() + 1) / n)
    if kind == 'noninferiority':
        return float(((boot <= -margin).sum() + 1) / n)
    if kind == 'equivalence':
        return float((max((boot <= -margin).sum(), (boot >= margin).sum()) + 1) / n)
    raise ValueError(f"unknown test kind {kind!r}")


def paired_bootstrap_test(a: Sequence[float], b: Sequence[float], clusters: Optional[Sequence], kind: str,
                          margin: float = 0.0, n_boot: int = 10000, seed: int = 42) -> Dict[str, Optional[float]]:
    """mean(a - b), its paired cluster-bootstrap 95% CI and the one-sided p of `kind` (directional_p)."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    keep = ~np.isnan(diff)
    n = int(keep.sum())
    if n < 2:
        return {'diff': float(np.nanmean(diff)) if n else float('nan'), 'ci95': [None, None], 'p': None, 'n': n}
    cl = None if clusters is None else [c for c, k in zip(clusters, keep) if k]
    boot = _boot_means(diff[keep], cl, n_boot, seed)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {'diff': float(diff[keep].mean()), 'ci95': [float(lo), float(hi)], 'p': directional_p(boot, kind, margin),
            'n': n}


def cluster_bootstrap_column_means(arr: np.ndarray, clusters: Sequence, n_boot: int, seed: int = 42) -> np.ndarray:
    """[n_boot, n_columns] column means of `arr` [n_rows, n_columns] under a
    cluster bootstrap of its rows -- for statistics that are functions of
    several paired means (upgrade retention and its differences)."""
    members = _cluster_members(clusters)
    sums = np.stack([arr[m].sum(axis=0) for m in members])           # [k, cols]
    sizes = np.array([len(m) for m in members], dtype=float)         # [k]
    draws = np.random.default_rng(seed).integers(0, len(members), size=(n_boot, len(members)))
    return sums[draws].sum(axis=1) / sizes[draws].sum(axis=1)[:, None]


def holm_adjust(pvalues: Sequence[Optional[float]]) -> List[Optional[float]]:
    """Holm-Bonferroni adjusted p-values (family-wise error); None stays None."""
    idx = [i for i, p in enumerate(pvalues) if p is not None]
    m = len(idx)
    out: List[Optional[float]] = [None] * len(pvalues)
    running = 0.0
    for rank, i in enumerate(sorted(idx, key=lambda i: pvalues[i])):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        out[i] = running
    return out


def bh_adjust(pvalues: Sequence[Optional[float]]) -> List[Optional[float]]:
    """Benjamini-Hochberg q-values (false discovery rate); None stays None."""
    idx = [i for i, p in enumerate(pvalues) if p is not None]
    m = len(idx)
    out: List[Optional[float]] = [None] * len(pvalues)
    running = 1.0
    for rank, i in reversed(list(enumerate(sorted(idx, key=lambda i: pvalues[i]), start=1))):
        running = min(running, pvalues[i] * m / rank)
        out[i] = running
    return out


def upgrade_retention_array(weak_compressed, strong_compressed, weak_full, strong_full) -> np.ndarray:
    """Vectorized upgrade_retention (NaN where the full-context gap is ~0)."""
    gap = np.asarray(strong_full, dtype=float) - np.asarray(weak_full, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = (np.asarray(strong_compressed, dtype=float) - np.asarray(weak_compressed, dtype=float)) / gap
    return np.where(np.abs(gap) < 1e-9, np.nan, out)


def upgrade_retention(weak_compressed: float, strong_compressed: float, weak_full: float, strong_full: float) -> float:
    """Share of the full-context reader upgrade (weak -> strong) that
    survives compression (arXiv 2606.21807's measurement): 1.0 = the
    compressor preserves the whole gap, 0 = it erases it, <0 = it flips it."""
    gap = strong_full - weak_full
    if abs(gap) < 1e-9:
        return float('nan')
    return (strong_compressed - weak_compressed) / gap
