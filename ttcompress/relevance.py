"""relevance(i) = p_i = sigmoid(f_theta(x_i)) — PCS_METHOD_SPEC.md §3.

Two providers, both bolt-on at inference time (torch.no_grad, no gradients
ever run here):

  - E6RelevanceProvider: the original wave-2 E6/E11 token CLASSIFIER
    (2-class keep/drop, softmax). Bolt-on only per PCS_METHOD_SPEC.md hard
    constraint #1 -- this file never trains it.
  - RegressionRelevanceProvider: the OUTCOME_SUPERVISED_RELEVANCE_SPEC.md
    Stage C checkpoints (relevance_v2/relevance_v3) -- SAME base encoder
    architecture, but a 1-unit REGRESSION head (AutoModelForTokenClassification
    with num_labels=1), trained by train_relevance.py (that script is where
    the actual gradient training happens, not this file). Score is
    sigmoid(raw logit), which is the literal p_i = sigmoid(f_theta(x_i))
    formula from PCS_METHOD_SPEC.md §3 -- softmax over a single logit would
    always be 1.0 and is not applicable here.

Status as of this build: no trained checkpoint exists yet for either
provider (E6 classifier: scripts/train_encoder_compressor.py in vncompress,
unrun; regression: train_relevance.py in this repo, unrun -- needs Stage A/B
labels first). Both are real, runnable code paths that have NOT been
exercised end-to-end with a real checkpoint. SyntheticRelevanceProvider below
exists only to smoke-test the rest of the pipeline (train.py/evaluate.py
wiring, the lambda sweep, the bootstrap CI) without a GPU or a checkpoint; it
must never be used to produce a reported number.
"""
from __future__ import annotations

import random
from typing import List, Optional

import torch
import torch.nn.functional as F

from .compression import pool_char_to_token, token_spans


class _BaseEncoderRelevanceProvider:
    """Shared char-offset -> encoder-window -> char-score -> token-score
    pooling pipeline (adapted from
    vncompress/vncompress/encoder_compression.py::EncoderClassifierCompressor).
    Subclasses only implement `_score_logits`, which turns one window's raw
    `[S, num_labels]` logits into a `[S]` per-token score in [0, 1] --
    everything else (windowing, offset mapping, char pooling onto the
    generation tokenizer's tokens) is identical between the classifier and
    the regressor.
    """

    def __init__(
        self, generation_tokenizer, encoder_path: str,
        device: str = 'cuda', max_encoder_len: int = 256, stride: int = 128,
    ):
        self.tokenizer = generation_tokenizer
        self.encoder_path = encoder_path
        self.device = device
        self.max_encoder_len = max_encoder_len
        self.stride = max(1, min(stride, max_encoder_len - 1))
        self._encoder = None
        self._enc_tok = None

    def _ensure_encoder(self):
        if self._encoder is not None:
            return
        from transformers import AutoModelForTokenClassification, AutoTokenizer
        self._enc_tok = AutoTokenizer.from_pretrained(self.encoder_path, use_fast=True)
        self._encoder = AutoModelForTokenClassification.from_pretrained(self.encoder_path)
        if self.device == 'cuda' and torch.cuda.is_available():
            self._encoder = self._encoder.to('cuda')
        self._encoder.eval()

    def _encoder_device(self):
        try:
            return next(self._encoder.parameters()).device
        except (StopIteration, AttributeError):
            return torch.device('cpu')

    def _encode_offsets(self, text: str):
        try:
            enc = self._enc_tok(text, return_offsets_mapping=True, add_special_tokens=False, truncation=False)
            return list(enc['input_ids']), list(enc['offset_mapping'])
        except (TypeError, NotImplementedError, ValueError, KeyError):
            ids = self._enc_tok.encode(text, add_special_tokens=False)
            offsets, cursor = [], 0
            for tid in ids:
                piece = self._enc_tok.decode([tid])
                found = text.find(piece, cursor) if piece else -1
                if found < 0:
                    offsets.append((cursor, cursor))
                else:
                    offsets.append((found, found + len(piece)))
                    cursor = found + len(piece)
            return ids, offsets

    def _score_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """logits: [S, num_labels] for one encoder window -> [S] score in
        [0, 1]. Overridden per subclass (softmax-index vs sigmoid)."""
        raise NotImplementedError

    @torch.no_grad()
    def _score_per_char(self, text: str) -> torch.Tensor:
        ids, offsets = self._encode_offsets(text)
        char_sum = torch.zeros(len(text))
        char_cnt = torch.zeros(len(text))
        if not ids:
            return torch.full((len(text),), float('nan'))

        device = self._encoder_device()
        n_tok = len(ids)
        begin = 0
        while begin < n_tok:
            end = min(begin + self.max_encoder_len, n_tok)
            tensor = torch.tensor([ids[begin:end]], device=device)
            logits = self._encoder(tensor).logits[0]
            scores = self._score_logits(logits.float()).cpu()
            for k, (s, e) in enumerate(offsets[begin:end]):
                if e > s and k < scores.numel():
                    char_sum[s:e] += float(scores[k])
                    char_cnt[s:e] += 1.0
            if end >= n_tok:
                break
            begin += self.stride

        char_scores = torch.full((len(text),), float('nan'))
        covered = char_cnt > 0
        char_scores[covered] = char_sum[covered] / char_cnt[covered]
        return char_scores

    def relevance(self, input_ids: List[int]) -> List[float]:
        self._ensure_encoder()
        n = len(input_ids)
        text, spans = token_spans(self.tokenizer, input_ids)
        char_scores = self._score_per_char(text)
        token_scores = pool_char_to_token(char_scores, spans, n, fill_neutral=True)
        return token_scores.tolist()


class E6RelevanceProvider(_BaseEncoderRelevanceProvider):
    """Per-token P(keep) from a trained LLMLingua-2-style 2-class token
    classifier (wave-2 E6/E11 in vncompress). Query-agnostic: `relevance()`
    takes only token ids, never a query."""

    def __init__(
        self, generation_tokenizer, encoder_path: str,
        device: str = 'cuda', keep_label: int = 1,
        max_encoder_len: int = 256, stride: int = 128,
    ):
        super().__init__(generation_tokenizer, encoder_path, device, max_encoder_len, stride)
        self.keep_label = keep_label

    def _score_logits(self, logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        label = min(self.keep_label, probs.shape[-1] - 1)
        return probs[:, label]


class RegressionRelevanceProvider(_BaseEncoderRelevanceProvider):
    """Per-token sigmoid(raw regression logit) from an
    OUTCOME_SUPERVISED_RELEVANCE_SPEC.md Stage C checkpoint
    (num_labels=1, trained by train_relevance.py with MSE loss against
    normalized outcome-supervised labels -- see that script and
    ttcompress/relevance_training.py). Same query-agnostic contract as
    E6RelevanceProvider."""

    def _score_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logits[:, 0])


class SyntheticRelevanceProvider:
    """Deterministic pseudo-relevance for local smoke-testing only (no GPU,
    no checkpoint). NEVER use for a reported lambda-sweep or bench number —
    it carries no semantic signal. See module docstring."""

    def __init__(self, seed: int = 42):
        self._seed = seed

    def relevance(self, input_ids: List[int]) -> List[float]:
        rng = random.Random(self._seed ^ hash(tuple(input_ids[:8])) & 0xFFFF)
        return [rng.random() for _ in input_ids]
