"""Position-Calibrated Selection — the method under test.

    score(i) = (1 - lam) * relevance(i) + lam * g(i, L),   lam in [0, 1]

lam=0  -> score(i) = relevance(i)          -> the existing 'encoder' arm
lam=1  -> score(i) = g(i, L)               -> exactly reproduces
                                               TruncationCompressor (unit-
                                               tested in tests/, see
                                               PCS_METHOD_SPEC.md §2)
0<lam<1 -> the new, unexplored region.

Confirms PCS_METHOD_SPEC.md §4: E6 (see relevance.py, copied from
vncompress/vncompress/encoder_compression.py) pools its keep-probability
onto GENERATION-TOKENIZER token spans (`_pool_char_to_token`), i.e. one
relevance score per token in `input_ids` — the same unit `g(i, L)` is
defined over. No token->chunk conversion needed.
"""
from __future__ import annotations

from typing import List, Optional, Protocol, Sequence

from .compression import BaseCompressor, CompressionConfig, CompressionResult, TruncationCompressor


def g(i: int, L: int) -> float:
    """Position score: 1 at both ends, 0 in the middle. U-shaped, symmetric.
    PCS_METHOD_SPEC.md §2. L=1 is a degenerate single-token sequence — every
    position is "the edge", so g=1."""
    if L <= 1:
        return 1.0
    d = min(i, L - 1 - i)
    return 1.0 - 2.0 * d / (L - 1)


def position_scores(n: int) -> List[float]:
    return [g(i, n) for i in range(n)]


def mix_scores(relevance: Sequence[float], n: int, lam: float) -> List[float]:
    pos = position_scores(n)
    return [(1.0 - lam) * relevance[i] + lam * pos[i] for i in range(n)]


class RelevanceProvider(Protocol):
    """relevance(input_ids) -> one score per token, in [0, 1]. Query-agnostic
    by construction (PCS_METHOD_SPEC.md §5): implementations must not accept
    or look at a query anywhere in the call chain."""

    def relevance(self, input_ids: List[int]) -> List[float]: ...


class PCSCompressor(BaseCompressor):
    """The score(i) family above, as a BaseCompressor arm.

    lam == 1.0 bypasses scoring entirely and delegates to
    TruncationCompressor — PCS_METHOD_SPEC.md §2 is explicit that a
    floating-point score_to_rank tie-break can diverge from
    TruncationCompressor's tail-leaning tie rule near the head/tail boundary,
    so lambda=1 must call the real thing rather than risk a near-miss.

    Every other lambda (including 0) goes through
    BaseCompressor.select_with_boundary — the "existing top-K selection
    rule" the checklist asks to reuse, and the same rule the old 'encoder'
    arm already used, so lambda=0 truly changes nothing.
    """

    def __init__(
        self, tokenizer, model=None, config: Optional[CompressionConfig] = None,
        lam: float = 0.0, relevance_provider: Optional[RelevanceProvider] = None,
    ):
        super().__init__(tokenizer, model, config)
        if not 0.0 <= lam <= 1.0:
            raise ValueError(f"lam must be in [0, 1]; got {lam}")
        if lam < 1.0 and relevance_provider is None:
            raise ValueError("relevance_provider is required for lam < 1.0")
        self.lam = lam
        self.relevance_provider = relevance_provider

    def get_name(self) -> str:
        return f"PCS[lam={self.lam:.2f}]"

    def compress(self, input_ids: List[int], **kwargs) -> CompressionResult:
        self.validate_input(input_ids)
        n = len(input_ids)

        if self.lam == 1.0:
            result = TruncationCompressor(self.tokenizer, self.model, self.config).compress(input_ids)
            result.method_name = self.get_name()
            result.metadata['lam'] = 1.0
            result.metadata['delegated_to'] = 'TruncationCompressor'
            return result

        import time
        start = time.time()
        relevance = self.relevance_provider.relevance(input_ids)
        if len(relevance) != n:
            raise ValueError(f"relevance_provider returned {len(relevance)} scores for {n} tokens")
        scores = mix_scores(relevance, n, self.lam)
        keep_indices = self.select_with_boundary(scores, n)
        compressed_ids = [input_ids[i] for i in keep_indices]
        return self._build_result(
            compressed_ids, n, (time.time() - start) * 1000,
            metadata={'lam': self.lam},
        )
