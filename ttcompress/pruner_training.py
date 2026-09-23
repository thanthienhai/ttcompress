"""Stage B -- distilling attribution labels into the pruner.

Label sources (what the pruner is supervised on), chosen per run:
  beta      within-document z-scored ridge coefficients from ONE reader
  ensemble  mean z-score over several readers (reader-agnostic label, RQ3)
  span      binary gold-chunk label (needle / supporting facts): the
            answer-span-supervised pruner the utility label must beat (RQ2)

Losses:
  listnet   cross-entropy between softmax(target / tau) and softmax(scores)
            over the document's chunks -- what selection actually needs is
            the ranking inside one document
  mse       regression on the z-score (keeps score scale meaningful; used
            as an auxiliary term, weight w_mse)
  bce       binary cross-entropy on the span label
Default: listnet + 0.5 * mse for beta/ensemble, bce for span.

Optional position adjustment: subtract a pooled relative-position prior
(attribution.fit_position_prior) from the z-labels before training, so the
pruner is not taught the reader's lost-in-the-middle bias (ablation).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .attribution import ChunkLabels, load_chunk_labels, position_bin, record_paths
from .metrics import ndcg_at_k, spearman


@dataclass
class TrainExample:
    doc_id: str
    source: str
    question: str
    chunks: List[str]
    target: List[float]        # z-label (beta/ensemble) or 0/1 (span)
    gold: List[int]
    beta_z: Optional[List[float]] = None   # kept for dev metrics even when training on span


def load_label_dirs(dirs: Sequence[str]) -> List[ChunkLabels]:
    labels = []
    for d in dirs:
        paths = record_paths(d)
        if not paths:
            raise SystemExit(f"no label files in {d}")
        labels.extend(load_chunk_labels(p) for p in paths)
    return labels


def make_examples(labels: Sequence[ChunkLabels], label_source: str,
                  position_prior: Optional[Sequence[float]] = None) -> List[TrainExample]:
    """Uninformative documents (outcome never varied) are dropped for
    beta/ensemble -- a constant label carries no ranking signal -- but kept
    for span, whose label does not depend on the reader."""
    out = []
    for lab in labels:
        doc = lab.doc
        C = len(doc['chunks'])
        if label_source == 'span':
            target = [1.0 if i in set(doc['gold_chunks']) else 0.0 for i in range(C)]
        else:
            if not lab.informative:
                continue
            target = list(lab.z)
            if position_prior is not None:
                target = [z - position_prior[position_bin(i, C, len(position_prior))] for i, z in enumerate(target)]
        out.append(TrainExample(doc['doc_id'], doc['source'], doc['question'], doc['chunks'], target,
                                list(doc['gold_chunks']), list(lab.z) if lab.informative else None))
    return out


def listnet_loss(scores: torch.Tensor, target: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    return -(F.softmax(target / tau, dim=-1) * F.log_softmax(scores, dim=-1)).sum()


def example_loss(scores: torch.Tensor, ex: TrainExample, label_source: str, tau: float, w_mse: float) -> torch.Tensor:
    target = torch.tensor(ex.target, dtype=scores.dtype, device=scores.device)
    if label_source == 'span':
        pos = target.sum().clamp_min(1.0)
        pos_weight = ((len(target) - pos) / pos).clamp_min(1.0)  # few gold chunks among many
        return F.binary_cross_entropy_with_logits(scores, target, pos_weight=pos_weight)
    loss = listnet_loss(scores, target, tau)
    if w_mse:
        loss = loss + w_mse * F.mse_loss(scores, target)
    return loss


def ranking_metrics(scores: Sequence[float], ex: TrainExample, keep_fraction: float = 0.25) -> Dict[str, float]:
    """Reader-free dev metrics: gold recall inside the top ~keep_fraction of
    chunks, and agreement with the (single-reader or ensemble) beta label."""
    order = np.argsort(-np.asarray(scores, dtype=float), kind='stable')
    k = max(1, round(len(scores) * keep_fraction))
    top = set(order[:k].tolist())
    m = {'gold_recall@25%': sum(g in top for g in ex.gold) / len(ex.gold) if ex.gold else float('nan')}
    if ex.beta_z is not None:
        m['ndcg@3_beta'] = ndcg_at_k(scores, ex.beta_z, 3)
        m['spearman_beta'] = spearman(scores, ex.beta_z)
    return m


def mean_metrics(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    keys = sorted({k for r in rows for k in r})
    return {k: float(np.nanmean([r.get(k, np.nan) for r in rows])) for k in keys}
