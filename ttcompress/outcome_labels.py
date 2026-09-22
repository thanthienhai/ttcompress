"""Stage A — Outcome-Supervised Relevance: generate (mask, token_f1) labels
by actually measuring the reader's outcome under random chunk-keep masks,
then estimate each chunk's marginal contribution via ridge regression.

OUTCOME_SUPERVISED_RELEVANCE_SPEC.md §2. Unlike ttcompress/pcs.py (bolt-on,
0 GPU-hour), this stage is deliberately expensive: it calls the reader
(ttcompress.reader.generate_answer) once per mask per document.

Chunk = one paragraph of the constructed needle+haystack document (§2.2),
matching how the existing internal needle_in_haystack task defines a unit
-- see build_document.py for the construction itself. A chunk's regression
target here (beta_c) is later broadcast to every TOKEN inside that chunk
when training the classifier (Stage C): E6 itself scores per token (resolved
in PCS_METHOD_SPEC.md §4 / ttcompress/pcs.py), chunk is only the unit these
labels are estimated at.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# §2.3 — mask generation
# ---------------------------------------------------------------------------

def generate_masks(C: int, K: int, ratios: Sequence[float] = (4, 8), seed: int = 0) -> List[List[bool]]:
    """K random binary keep-masks over C chunks. Keep-count concentrated at
    round(C/ratio) for each ratio in `ratios` (K split evenly across them) --
    matches the 4x/8x operating points PCS is evaluated at
    (PCS_METHOD_SPEC.md §6), per spec §2.3 "tập trung xác suất giữ quanh 1/4
    hoặc 1/8". Each mask keeps a uniformly random subset of chunks of that
    size; order is restored by apply_mask, not encoded in the mask itself."""
    if C <= 0:
        raise ValueError(f"C must be > 0; got {C}")
    rng = random.Random(seed)
    masks = []
    for i in range(K):
        ratio = ratios[i % len(ratios)]
        keep_k = min(C, max(1, round(C / ratio)))
        kept = set(rng.sample(range(C), keep_k))
        masks.append([c in kept for c in range(C)])
    return masks


def apply_mask(chunks: Sequence[str], mask: Sequence[bool]) -> str:
    """Join kept chunks in original order -- the same order-preserving rule
    TruncationCompressor/PCS use at token level (ttcompress/compression.py),
    applied here at chunk granularity per spec §2.3."""
    if len(chunks) != len(mask):
        raise ValueError(f"chunks ({len(chunks)}) and mask ({len(mask)}) length mismatch")
    return ' '.join(c for c, keep in zip(chunks, mask) if keep)


# ---------------------------------------------------------------------------
# §2.4 — outcome measurement (calls the reader; the expensive part, §7)
# ---------------------------------------------------------------------------

def measure_outcomes(
    chunks: Sequence[str], masks: Sequence[Sequence[bool]], query: str, reference_answer: str,
    generate_fn: Callable[[str, str], Optional[str]], score_fn: Callable[[str, str], float],
) -> List[float]:
    """token_f1 (or any score_fn) of the reader's answer for each mask, in
    the same order as `masks`. `generate_fn(context_text, query) -> answer`
    and `score_fn(answer, reference) -> float` are injected so this stays
    untied to a specific reader/model -- callers pass
    `lambda ctx, q: reader.generate_answer(model, tokenizer, tokenizer.encode(ctx, ...), q, ...)`
    and `metrics.token_f1_one`."""
    outcomes = []
    for mask in masks:
        context = apply_mask(chunks, mask)
        answer = generate_fn(context, query)
        outcomes.append(score_fn(answer or '', reference_answer))
    return outcomes


# ---------------------------------------------------------------------------
# §2.5 — ridge regression: token_f1(m_k) ~= beta_0 + sum_c beta_c * m_k[c]
# ---------------------------------------------------------------------------

def _design_matrix(masks: np.ndarray) -> np.ndarray:
    """(K, C) 0/1 mask array -> (K, C+1) with an intercept column at index 0."""
    k = masks.shape[0]
    return np.hstack([np.ones((k, 1)), masks.astype(float)])


def fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Closed-form ridge regression; the intercept (column 0) is not
    penalized, standard practice so `alpha` only shrinks the per-chunk
    coefficients, not the document's baseline token_f1."""
    n_features = X.shape[1]
    penalty = alpha * np.eye(n_features)
    penalty[0, 0] = 0.0
    return np.linalg.solve(X.T @ X + penalty, X.T @ y)


def estimate_chunk_contributions(masks: Sequence[Sequence[bool]], outcomes: Sequence[float], alpha: float) -> np.ndarray:
    """beta, shape (C+1,): beta[0] is the intercept, beta[1:] are the
    per-chunk contributions beta_c from PCS_METHOD_SPEC.md's sibling doc §2.5."""
    X = _design_matrix(np.asarray(masks))
    y = np.asarray(outcomes, dtype=float)
    if X.shape[0] < X.shape[1]:
        raise ValueError(
            f"{X.shape[0]} masks < {X.shape[1]} parameters (C+1 chunks+intercept) -- "
            f"underdetermined even with ridge; increase K for this document."
        )
    return fit_ridge(X, y, alpha)


def cv_mse(masks: Sequence[Sequence[bool]], outcomes: Sequence[float], alpha: float, n_folds: int = 5, seed: int = 0) -> float:
    """K-fold CV mean squared error for one document's (masks, outcomes) at
    a given alpha -- the per-document building block §2.5's "chọn alpha
    bằng CV trên dev split" averages over (see select_alpha)."""
    X = _design_matrix(np.asarray(masks))
    y = np.asarray(outcomes, dtype=float)
    n = len(y)
    folds_n = min(n_folds, n)
    if folds_n < 2:
        return float('nan')
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, folds_n)
    sq_errors = []
    for i in range(len(folds)):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(len(folds)) if j != i])
        if len(train_idx) < X.shape[1]:
            continue  # not enough rows to fit this fold's ridge -- skip it
        beta = fit_ridge(X[train_idx], y[train_idx], alpha)
        preds = X[test_idx] @ beta
        sq_errors.extend(((preds - y[test_idx]) ** 2).tolist())
    return float(np.mean(sq_errors)) if sq_errors else float('nan')


def select_alpha(
    dev_documents: Sequence[Tuple[Sequence[Sequence[bool]], Sequence[float]]],
    alpha_grid: Sequence[float] = (0.01, 0.1, 1.0, 10.0, 100.0), n_folds: int = 5, seed: int = 0,
) -> float:
    """The single global alpha (§2.5: "chọn alpha bằng CV trên dev split,"
    not per-document) -- mean per-document CV MSE across every Dev-split
    document, argmin over `alpha_grid`."""
    best_alpha, best_score = alpha_grid[0], float('inf')
    for alpha in alpha_grid:
        scores = [cv_mse(masks, outcomes, alpha, n_folds, seed) for masks, outcomes in dev_documents]
        scores = [s for s in scores if not np.isnan(s)]
        mean_score = float(np.mean(scores)) if scores else float('inf')
        if mean_score < best_score:
            best_alpha, best_score = alpha, mean_score
    return best_alpha


# ---------------------------------------------------------------------------
# §2.6 — bootstrap stability check
# ---------------------------------------------------------------------------

def bootstrap_beta_ci(
    masks: Sequence[Sequence[bool]], outcomes: Sequence[float], alpha: float,
    n_boot: int = 1000, ci: float = 0.95, seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap resample masks (rows) with replacement, refit ridge each
    time. Returns (beta_point, ci_lo, ci_hi), each shape (C+1,)."""
    X = _design_matrix(np.asarray(masks))
    y = np.asarray(outcomes, dtype=float)
    n = len(y)
    beta_point = fit_ridge(X, y, alpha)
    rng = np.random.default_rng(seed)
    boot_betas = np.empty((n_boot, X.shape[1]))
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_betas[b] = fit_ridge(X[idx], y[idx], alpha)
    lo = np.quantile(boot_betas, (1 - ci) / 2, axis=0)
    hi = np.quantile(boot_betas, 1 - (1 - ci) / 2, axis=0)
    return beta_point, lo, hi


def low_confidence_chunks(ci_lo: np.ndarray, ci_hi: np.ndarray, width_threshold: float = 0.5) -> np.ndarray:
    """Per-chunk boolean (excludes the intercept at index 0): CI width wider
    than `width_threshold` (token_f1 in [0,1]; a 0.5-wide CI spans half the
    possible range, i.e. this chunk's contribution isn't pinned down --
    the default is a documented judgment call, not from the spec, which
    names the flag but not a numeric cutoff)."""
    width = ci_hi[1:] - ci_lo[1:]
    return width > width_threshold


# ---------------------------------------------------------------------------
# Per-document label record -- what gets written to
# outcome_labels_raw/{doc_id}.jsonl (§2.4) and read back for Stage B/C.
# ---------------------------------------------------------------------------

@dataclass
class MaskOutcomeRecord:
    """The expensive-to-produce intermediate: one document's masks +
    measured outcomes, before ridge fitting. Saved separately from the final
    DocumentLabels so re-running Stage A's ridge/CV/bootstrap math (cheap,
    CPU-only) never requires re-calling the reader (expensive, GPU-only) --
    the same measure/fit split as this project's train.py vs evaluate.py."""
    doc_id: str
    chunks: List[str]
    positions: List[int]           # 0..num_chunks-1
    question: str
    reference_answer: str
    masks: List[List[bool]]
    outcomes: List[float]
    metadata: Dict = field(default_factory=dict)


def save_mask_outcomes(record: MaskOutcomeRecord, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{record.doc_id}.jsonl")
    with open(path, 'w', encoding='utf-8') as f:
        f.write(json.dumps({
            'doc_id': record.doc_id, 'chunks': record.chunks, 'positions': record.positions,
            'question': record.question, 'reference_answer': record.reference_answer,
            'masks': record.masks, 'outcomes': record.outcomes, 'metadata': record.metadata,
        }, ensure_ascii=False) + '\n')
    return path


def load_mask_outcomes(path: str) -> MaskOutcomeRecord:
    with open(path, 'r', encoding='utf-8') as f:
        d = json.loads(f.readline())
    return MaskOutcomeRecord(**d)


@dataclass
class DocumentLabels:
    doc_id: str
    chunks: List[str]              # the constructed document's chunk texts, in order --
                                    # carried here (not re-derived) so Stage B/C never need
                                    # to re-run UIT-ViQuAD sampling/document construction
                                    # with matching seeds to recover them.
    num_chunks: int
    positions: List[int]          # chunk index i_c in the constructed document, 0..num_chunks-1
    beta: List[float]             # beta[0] = intercept, beta[1:] = per-chunk contributions
    ci_lo: List[float]
    ci_hi: List[float]
    low_confidence: List[bool]    # per chunk (len == num_chunks, aligned with positions/beta[1:])
    alpha: float
    metadata: Dict = field(default_factory=dict)


def save_document_labels(labels: DocumentLabels, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{labels.doc_id}.jsonl")
    with open(path, 'w', encoding='utf-8') as f:
        f.write(json.dumps({
            'doc_id': labels.doc_id, 'chunks': labels.chunks, 'num_chunks': labels.num_chunks,
            'positions': labels.positions, 'beta': labels.beta, 'ci_lo': labels.ci_lo, 'ci_hi': labels.ci_hi,
            'low_confidence': labels.low_confidence, 'alpha': labels.alpha, 'metadata': labels.metadata,
        }, ensure_ascii=False) + '\n')
    return path


def load_document_labels(path: str) -> DocumentLabels:
    with open(path, 'r', encoding='utf-8') as f:
        d = json.loads(f.readline())
    return DocumentLabels(**d)
