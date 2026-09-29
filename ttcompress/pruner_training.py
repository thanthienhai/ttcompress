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

Reductions: listnet SUMS over the C chunks (a cross-entropy between two
distributions, Cao et al. 2007); mse is the MEAN over chunks (the paper
states it this way). With tau=1 both terms have the same minimiser (scores
= z), so the mix does not move the ranking target, only the emphasis. The
gradient-norm ratio w_mse*mse : listnet is ~1 for smooth (Gaussian) targets
at any C, but real labels are spiky (one needle / two supporting paragraphs)
and there it falls with C at initialisation: measured ~1.3 at C=2, ~0.5-0.7
at C=10, ~0.2-0.3 at C=40 -- long single-hop haystacks lean on the listnet
(top-of-ranking) term more than 10-paragraph multi-hop documents do.
Averaging listnet over chunks would instead shrink its gradient by 1/C.
Both components are logged per epoch (train_pruner.py, train_log.json).

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
    """Uninformative documents (outcome never varied) are dropped: a constant
    label carries no ranking signal. Also for span, whose label does not need
    the reader: the RQ2 control must train on exactly pruner_beta_primary's
    documents (it reads the same fit dirs), so only the label type differs."""
    out = []
    for lab in labels:
        if not lab.informative:
            continue
        doc = lab.doc
        C = len(doc['chunks'])
        if label_source == 'span':
            target = [1.0 if i in set(doc['gold_chunks']) else 0.0 for i in range(C)]
        else:
            target = list(lab.z)
            if position_prior is not None:
                target = [z - position_prior[position_bin(i, C, len(position_prior))] for i, z in enumerate(target)]
        out.append(TrainExample(doc['doc_id'], doc['source'], doc['question'], doc['chunks'], target,
                                list(doc['gold_chunks']), list(lab.z)))
    return out


def listnet_loss(scores: torch.Tensor, target: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    # Cross-entropy between the two chunk distributions: a sum over chunks (Cao et al. 2007), next to a
    # mean-reduced MSE. Not an imbalance: the ListNet value grows like log C only through the label's
    # entropy, which has no gradient; per chunk, both gradients are ~ z_i / C at init (0.5·MSE / ListNet
    # gradient norm ≈ 1 for C = 2..40), so w_mse weighs the two terms alike for any C. A mean over chunks
    # here would shrink ListNet's gradient C-fold on long documents.
    return -(F.softmax(target / tau, dim=-1) * F.log_softmax(scores, dim=-1)).sum()


def example_loss(scores: torch.Tensor, ex: TrainExample, label_source: str, tau: float, w_mse: float,
                 parts: Optional[Dict[str, float]] = None) -> torch.Tensor:
    """`parts`, if given, receives the unweighted component values (for logging)."""
    target = torch.tensor(ex.target, dtype=scores.dtype, device=scores.device)
    if label_source == 'span':
        pos = target.sum().clamp_min(1.0)
        pos_weight = ((len(target) - pos) / pos).clamp_min(1.0)  # few gold chunks among many
        loss = F.binary_cross_entropy_with_logits(scores, target, pos_weight=pos_weight)
        if parts is not None:
            parts['bce'] = loss.item()
        return loss
    loss = listnet_loss(scores, target, tau)
    if parts is not None:
        parts['listnet'] = loss.item()
    if w_mse:
        mse = F.mse_loss(scores, target)
        if parts is not None:
            parts['mse'] = mse.item()
        loss = loss + w_mse * mse
    return loss


def accumulation_group_size(pos: int, n: int, docs_per_step: int) -> int:
    """Documents in the gradient-accumulation group that position `pos` (of n)
    belongs to: docs_per_step, except for a shorter final group. Dividing each
    document's loss by this (not by docs_per_step) makes every optimizer step
    average over its own documents, so the last step is not under-weighted."""
    start = (pos // docs_per_step) * docs_per_step
    return min(docs_per_step, n - start)


def ranking_metrics(scores: Sequence[float], ex: TrainExample, keep_fraction: float = 0.25) -> Dict[str, float]:
    """Reader-free dev metrics: gold recall inside the top ~keep_fraction of
    chunks, and agreement with the (single-reader or ensemble) beta label."""
    if not np.all(np.isfinite(np.asarray(scores, dtype=float))):
        # a NaN model must not look like `lead` (argsort keeps document order) and win model selection
        nan = float('nan')
        return {'gold_recall@25%': nan, **({'ndcg@3_beta': nan, 'spearman_beta': nan} if ex.beta_z is not None else {})}
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
