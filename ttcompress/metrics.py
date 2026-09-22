"""Scoring + the document-clustered bootstrap CI.

token_f1/needle_recall copied verbatim from vncompress/vncompress/evaluation.py
so numbers are comparable to the existing WAVE4 results. The cluster bootstrap
is copied verbatim from vncompress/vncompress/evaluation.py
(_cluster_member_indices/_bootstrap_diff_means) + scripts/analyze_sweep.py
(bootstrap_mean_ci) — this is "the document-clustered bootstrap already fixed
in WAVE4_REPORT.md §9" PCS_METHOD_SPEC.md §7 says to reuse, not rewrite.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def _normalize_answer(text: str) -> List[str]:
    """Lowercase, strip punctuation, split on whitespace (SQuAD-style),
    keeping tone marks intact -- syllable-level overlap for Vietnamese."""
    text = unicodedata.normalize('NFC', text).lower()
    text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
    return text.split()


def compute_token_f1(predictions: List[str], references: List[str]) -> float:
    """Mean SQuAD-style token-overlap F1."""
    if not predictions:
        return 0.0
    scores = []
    for pred, ref in zip(predictions, references):
        p_tokens, r_tokens = _normalize_answer(pred), _normalize_answer(ref)
        if not p_tokens or not r_tokens:
            scores.append(float(p_tokens == r_tokens))
            continue
        common = Counter(p_tokens) & Counter(r_tokens)
        overlap = sum(common.values())
        if overlap == 0:
            scores.append(0.0)
            continue
        precision, recall = overlap / len(p_tokens), overlap / len(r_tokens)
        scores.append(2 * precision * recall / (precision + recall))
    return float(np.mean(scores))


def token_f1_one(prediction: str, reference: str) -> float:
    return compute_token_f1([prediction], [reference])


def compute_needle_recall(predictions: List[str], references: List[str]) -> float:
    """Mean needle-retrieval recall: fraction of the reference's syllables
    recovered in the prediction."""
    if not predictions:
        return 0.0
    scores = []
    for pred, ref in zip(predictions, references):
        p_tokens, r_tokens = _normalize_answer(pred), _normalize_answer(ref)
        if not r_tokens:
            scores.append(float(not p_tokens))
            continue
        common = Counter(p_tokens) & Counter(r_tokens)
        scores.append(sum(common.values()) / len(r_tokens))
    return float(np.mean(scores))


def needle_recall_one(prediction: str, reference: str) -> float:
    return compute_needle_recall([prediction], [reference])


def _cluster_member_indices(clusters: Sequence) -> List[np.ndarray]:
    """Group row positions by cluster label (e.g. haystack_id), preserving
    first-seen order — rows sharing a label are resampled as one block."""
    groups: Dict[object, List[int]] = {}
    for i, c in enumerate(clusters):
        groups.setdefault(c, []).append(i)
    return [np.asarray(v, dtype=int) for v in groups.values()]


def _bootstrap_means(values: np.ndarray, n_boot: int, rng: np.random.Generator,
                      clusters: Optional[Sequence] = None) -> np.ndarray:
    """Bootstrap distribution of mean(values). iid over rows when `clusters`
    is None; otherwise a CLUSTER bootstrap (Cameron, Gelbach & Miller 2008)
    that resamples whole clusters with replacement."""
    if clusters is None:
        idx = rng.integers(0, len(values), size=(n_boot, len(values)))
        return values[idx].mean(axis=1)
    members = _cluster_member_indices(clusters)
    k = len(members)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sel = np.concatenate([members[j] for j in rng.integers(0, k, size=k)])
        means[i] = values[sel].mean()
    return means


def bootstrap_mean_ci(
    values: Sequence[Optional[float]], n_boot: int = 10000, ci: float = 0.95,
    seed: int = 42, clusters: Optional[Sequence] = None,
) -> Tuple[float, Optional[float], Optional[float]]:
    """(mean, ci_low, ci_high). Pass `clusters` (a label per value, e.g.
    haystack_id) to resample whole source documents instead of individual
    samples -- see PCS_METHOD_SPEC.md §7."""
    kept = [(v, None if clusters is None else clusters[i])
            for i, v in enumerate(values) if v is not None]
    vals = np.array([k[0] for k in kept], dtype=float)
    if len(vals) < 2:
        return (float(vals.mean()) if len(vals) else 0.0), None, None
    cl = [k[1] for k in kept] if clusters is not None else None
    rng = np.random.default_rng(seed)
    boot = _bootstrap_means(vals, n_boot, rng, cl)
    lo, hi = np.quantile(boot, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(vals.mean()), float(lo), float(hi)
