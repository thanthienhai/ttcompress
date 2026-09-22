"""Stage B — separate the position-signal component out of each chunk's
estimated contribution: beta_c = gamma_0 + gamma_1 * g(i_c, L) + residual_c.

OUTCOME_SUPERVISED_RELEVANCE_SPEC.md §3. `g(i_c, L)` is the exact same U-shaped
position score PCS uses (PCS_METHOD_SPEC.md §2), reused from ttcompress.pcs --
not redefined here, so a chunk that ridge regression already explains via
"it's near an edge" doesn't also get credit for it in the residual signal fed
to Stage C's classifier.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .outcome_labels import DocumentLabels
from .pcs import g


def normalize_per_document(values: Sequence[float]) -> List[float]:
    """z-score normalize one document's label array (beta_c or residual_c)
    to mean 0 / std 1 -- OUTCOME_SUPERVISED_RELEVANCE_SPEC.md §4 says Stage C
    trains on "label chuẩn hoá per-document" without naming a method; z-score
    is the standard choice for a regression target pooled across many
    documents of different raw-label scale, and is a documented judgment call
    here, not read off the spec. A document with zero variance (all chunks
    equally weighted) normalizes to all-zero rather than dividing by zero."""
    arr = np.asarray(values, dtype=float)
    std = arr.std()
    if std < 1e-12:
        return np.zeros_like(arr).tolist()
    return ((arr - arr.mean()) / std).tolist()


def position_scores_for_chunks(positions: Sequence[int], num_chunks: int) -> List[float]:
    """g(i_c, L) for each chunk position i_c in a document of `num_chunks`
    chunks -- L is num_chunks here (chunk-level unit, §2.1), not token count."""
    return [g(i, num_chunks) for i in positions]


def decompose(beta_c: Sequence[float], positions: Sequence[int], num_chunks: int) -> Tuple[float, float, np.ndarray]:
    """Ordinary least squares: beta_c ~= gamma_0 + gamma_1 * g(i_c, L).
    Returns (gamma_0, gamma_1, residual_c) where residual_c = beta_c - fitted.
    Two free parameters over len(beta_c) points -- no regularization needed
    (unlike Stage A's per-chunk ridge, which has as many parameters as data
    points; this is a simple 2-parameter fit over many chunks)."""
    beta = np.asarray(beta_c, dtype=float)
    pos_scores = np.asarray(position_scores_for_chunks(positions, num_chunks), dtype=float)
    X = np.column_stack([np.ones_like(pos_scores), pos_scores])
    gamma, *_ = np.linalg.lstsq(X, beta, rcond=None)
    gamma_0, gamma_1 = float(gamma[0]), float(gamma[1])
    fitted = X @ gamma
    residual = beta - fitted
    return gamma_0, gamma_1, residual


@dataclass
class TrainingLabels:
    """Stage C's actual input: one document's chunks + the two normalized
    label variants spec §3 says to keep in parallel (§4: "train song song 2
    classifier")."""
    doc_id: str
    chunks: List[str]
    positions: List[int]
    raw_label: List[float]        # normalize_per_document(beta_c) -> relevance_v2's target
    residual_label: List[float]   # normalize_per_document(residual_c) -> relevance_v3's target
    gamma_0: float
    gamma_1: float
    low_confidence: List[bool]
    metadata: Dict = field(default_factory=dict)


def decompose_document(labels: DocumentLabels) -> TrainingLabels:
    """DocumentLabels (Stage A output) -> TrainingLabels (Stage B output,
    Stage C input) for one document."""
    beta_c = labels.beta[1:]  # drop the intercept, per-chunk contributions only
    gamma_0, gamma_1, residual_c = decompose(beta_c, labels.positions, labels.num_chunks)
    return TrainingLabels(
        doc_id=labels.doc_id, chunks=labels.chunks, positions=labels.positions,
        raw_label=normalize_per_document(beta_c), residual_label=normalize_per_document(residual_c.tolist()),
        gamma_0=gamma_0, gamma_1=gamma_1, low_confidence=labels.low_confidence, metadata=labels.metadata,
    )


def save_training_labels(labels: TrainingLabels, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{labels.doc_id}.jsonl")
    with open(path, 'w', encoding='utf-8') as f:
        f.write(json.dumps({
            'doc_id': labels.doc_id, 'chunks': labels.chunks, 'positions': labels.positions,
            'raw_label': labels.raw_label, 'residual_label': labels.residual_label,
            'gamma_0': labels.gamma_0, 'gamma_1': labels.gamma_1,
            'low_confidence': labels.low_confidence, 'metadata': labels.metadata,
        }, ensure_ascii=False) + '\n')
    return path


def load_training_labels(path: str) -> TrainingLabels:
    with open(path, 'r', encoding='utf-8') as f:
        d = json.loads(f.readline())
    return TrainingLabels(**d)
