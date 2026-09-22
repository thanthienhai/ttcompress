"""Compressor base classes + the two fixed anchor/reference arms.

Copied (near-verbatim, trimmed to what PCS needs) from
vncompress/vncompress/compression.py — TruncationCompressor and
SnapKVCompressor especially must match byte-for-byte behavior, since PCS's
lambda=1 anchor is defined as "reproduces TruncationCompressor exactly"
(PCS_METHOD_SPEC.md §2) and h2o/snapkv are run as-is, not reimplemented.
"""
from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


@dataclass
class CompressionResult:
    compressed_ids: List[int]
    compressed_text: str
    original_length: int
    compressed_length: int
    compression_ratio: float
    token_savings_pct: float
    method_name: str
    processing_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompressionConfig:
    target_ratio: float = 4.0
    keep_boundary_tokens: int = 2
    min_compressed_length: int = 1


class BaseCompressor(ABC):
    """Shared budget math + result building. See vncompress.compression for
    the full docstring; trimmed here to the parts PCS/truncation/snapkv use."""

    def __init__(self, tokenizer, model=None, config: Optional[CompressionConfig] = None):
        self.tokenizer = tokenizer
        self.model = model
        self.config = config or CompressionConfig()

    @abstractmethod
    def compress(self, input_ids: List[int], **kwargs) -> CompressionResult: ...

    @abstractmethod
    def get_name(self) -> str: ...

    def _compute_compression_ratio(self, original_length: int, compressed_length: int) -> Tuple[float, float]:
        ratio = original_length / max(compressed_length, 1)
        savings = ((original_length - compressed_length) / max(original_length, 1)) * 100
        return ratio, savings

    def _build_result(self, compressed_ids, original_length, processing_time_ms, metadata=None) -> CompressionResult:
        comp_len = len(compressed_ids)
        ratio, savings = self._compute_compression_ratio(original_length, comp_len)
        try:
            compressed_text = self.tokenizer.decode(compressed_ids, skip_special_tokens=True)
        except Exception:
            compressed_text = '[decode error]'
        return CompressionResult(
            compressed_ids=compressed_ids, compressed_text=compressed_text,
            compression_ratio=ratio, token_savings_pct=savings,
            original_length=original_length, compressed_length=comp_len,
            method_name=self.get_name(), processing_time_ms=processing_time_ms,
            metadata=metadata or {},
        )

    def target_length(self, n: int) -> int:
        ratio = self.config.target_ratio
        if ratio <= 0:
            raise ValueError(f"target_ratio must be > 0 (ratio = original/compressed); got {ratio}")
        return max(int(n / ratio), self.config.min_compressed_length)

    def select_with_boundary(self, scores: Sequence[float], n: int) -> List[int]:
        """Indices to keep: `keep_boundary_tokens` on each side + top-scoring
        middle, in original order. This is the "existing top-K selection
        rule" PCS reuses for every lambda except the lambda=1 anchor (see
        pcs.PCSCompressor) — identical to the rule the old 'encoder' arm used,
        so lambda=0 stays "unchanged" per PCS_METHOD_SPEC.md §0."""
        k = max(0, min(self.config.keep_boundary_tokens, n // 2))
        target_len = min(self.target_length(n), n)
        mid_start, mid_end = k, max(k, n - k)
        mid_budget = max(0, min(target_len - 2 * k, mid_end - mid_start))
        chosen: List[int] = []
        if mid_budget > 0:
            mid = list(range(mid_start, mid_end))
            chosen = sorted(sorted(mid, key=lambda i: scores[i], reverse=True)[:mid_budget])
        return sorted(set(range(k)) | set(chosen) | set(range(mid_end, n)))

    def validate_input(self, input_ids: List[int]) -> bool:
        if not input_ids:
            raise ValueError("Empty input sequence")
        return len(input_ids) >= self.config.min_compressed_length


class TruncationCompressor(BaseCompressor):
    """Head+tail truncation — the query-agnostic floor PCS's lambda=1 anchor
    must reproduce exactly. See vncompress.compression.TruncationCompressor
    for the full rationale; logic copied verbatim (ratio convention:
    original/compressed — do not invert, see PCS_METHOD_SPEC.md §2 warning).
    """

    def compress(self, input_ids: List[int], **kwargs) -> CompressionResult:
        start = time.time()
        self.validate_input(input_ids)
        n = len(input_ids)
        ratio = self.config.target_ratio
        if ratio <= 0:
            raise ValueError(f"target_ratio must be > 0 (ratio = original/compressed); got {ratio}")
        keep_k = math.ceil(n / ratio)
        if keep_k >= n:
            return self._build_result(
                list(input_ids), n, (time.time() - start) * 1000,
                metadata={'head_k': n, 'tail_k': 0, 'kept_all': True},
            )
        head_k = keep_k // 2
        tail_k = keep_k - head_k
        compressed_ids = list(input_ids[:head_k]) + list(input_ids[n - tail_k:])
        return self._build_result(
            compressed_ids, n, (time.time() - start) * 1000,
            metadata={'head_k': head_k, 'tail_k': tail_k, 'kept_all': False},
        )

    def get_name(self) -> str:
        return "Truncation"


def _normalize(scores: torch.Tensor) -> torch.Tensor:
    if scores.numel() == 0:
        return scores
    lo, hi = scores.min(), scores.max()
    if hi > lo:
        return (scores - lo) / (hi - lo)
    return torch.full_like(scores, 0.5)


def token_spans(tokenizer, input_ids: Sequence[int]) -> Tuple[str, List[Tuple[int, int]]]:
    """Rebuild the decoded text and each token's character span in it."""
    pieces, spans, cursor = [], [], 0
    for tid in input_ids:
        piece = tokenizer.decode([tid], clean_up_tokenization_spaces=False)
        pieces.append(piece)
        spans.append((cursor, cursor + len(piece)))
        cursor += len(piece)
    return ''.join(pieces), spans


def pool_char_to_token(char_scores: torch.Tensor, spans: List[Tuple[int, int]], n: int, fill_neutral: bool) -> torch.Tensor:
    out = torch.zeros(n)
    for i, (s, e) in enumerate(spans):
        if e > s:
            window = char_scores[s:e]
            valid = window[~torch.isnan(window)]
            if valid.numel():
                out[i] = valid.mean()
    if fill_neutral:
        unmapped = out == 0
        if unmapped.any() and (~unmapped).any():
            out[unmapped] = out[~unmapped].median()
    return out


class SnapKVCompressor(BaseCompressor):
    """SnapKV / H2O / StreamingLLM attention-based selection (Li et al. 2024,
    arxiv:2404.14469; Zhang et al. 2023). Needs a real causal LM with
    output_attentions=True — this is the expensive arm in PCS_METHOD_SPEC.md
    §9/§10 and is NOT exercised on this dev machine; copied verbatim from
    vncompress so its numbers, when run, match the original registration."""

    def __init__(
        self, tokenizer, model=None, config: Optional[CompressionConfig] = None, device: str = 'cuda',
        window_size: int = 32, kernel_size: int = 5, max_capacity_prompt: int = 512,
        pooling: str = 'maxpool', mode: str = 'snapkv',
    ):
        super().__init__(tokenizer, model, config)
        self.device = device
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.max_capacity_prompt = max_capacity_prompt
        self.pooling = pooling
        self.mode = mode

    def get_name(self) -> str:
        return f"SnapKV-{self.mode}"

    def _compute_attention_importance(self, input_ids: torch.Tensor, attention_mask=None) -> torch.Tensor:
        """Attention-weight importance per key position, from the LAST
        `window_size` query positions only, averaged over the last
        len(layers)//4 layers -- exactly what SnapKV/H2O (Li et al. 2024;
        Zhang et al. 2023) call the "observation window".

        Split into two forward passes so only the observation window's
        attention weights ever get materialized, not every query
        position's: output_attentions=True hands back attention for EVERY
        layer at once (HF can't return a subset of layers), and at PCS's
        real context lengths that's infeasible -- LongBench's ~15,000-token
        documents would need 36 layers x 32 heads x 15,000^2 x 2 bytes =~
        518GB just for the attention tensors, a guaranteed CUDA OOM on any
        real GPU (this was unexercised code -- see the class docstring --
        until it was actually run and hit this). The math is unchanged:
        running the observation window through with a KV cache already
        built from the preceding tokens gives IDENTICAL attention weights
        to a single full pass (verified in
        tests/test_snapkv_windowed_attention.py against a real attention
        reference implementation) -- only what gets computed/retained is
        smaller.
        """
        if self.model is None:
            raise RuntimeError("SnapKV requires a model for attention computation")
        n = input_ids.shape[1]
        window_start = max(0, n - self.window_size)

        with torch.no_grad():
            if window_start > 0:
                prefill = self.model(
                    input_ids=input_ids[:, :window_start],
                    attention_mask=None if attention_mask is None else attention_mask[:, :window_start],
                    use_cache=True, output_attentions=False,
                )
                outputs = self.model(
                    input_ids=input_ids[:, window_start:],
                    attention_mask=None if attention_mask is None else attention_mask[:, :n],
                    past_key_values=prefill.past_key_values, use_cache=True, output_attentions=True,
                    cache_position=torch.arange(window_start, n, device=input_ids.device),
                )
            else:
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True, use_cache=True)

        all_attentions = outputs.attentions
        if all_attentions is None:
            raise RuntimeError("Model did not return attention weights. Set output_attentions=True.")

        attn_stack = torch.stack(all_attentions, dim=0)  # [L, B, H, W, S] -- W = window_size (or n if n <= window_size)
        num_layers_to_use = max(1, len(all_attentions) // 4)
        recent_attns = attn_stack[-num_layers_to_use:]

        importance = recent_attns.mean(dim=0).sum(dim=2).mean(dim=0)  # [H, S]
        if self.pooling in ('maxpool', 'avgpool') and self.kernel_size > 1:
            pool = F.max_pool1d if self.pooling == 'maxpool' else F.avg_pool1d
            imp = pool(importance.unsqueeze(0), kernel_size=self.kernel_size, stride=1, padding=self.kernel_size // 2)
            importance = imp.squeeze(0)
        for h in range(importance.shape[0]):
            importance[h] = _normalize(importance[h])
        return importance  # [H, S]

    def _h2o_importance(self, input_ids, attention_mask=None) -> torch.Tensor:
        return self._compute_attention_importance(input_ids, attention_mask).mean(dim=0).unsqueeze(0)

    def compress(self, input_ids: List[int], **kwargs) -> CompressionResult:
        start = time.time()
        n = len(input_ids)
        if not self.validate_input(input_ids):
            return self._build_result(list(input_ids), n, (time.time() - start) * 1000)

        input_tensor = torch.tensor([input_ids]).to(self.device)
        importance = (self._h2o_importance(input_tensor) if self.mode == 'h2o'
                      else self._compute_attention_importance(input_tensor))
        num_heads = importance.shape[0]

        budget = min(self.max_capacity_prompt, max(int(n / self.config.target_ratio), self.config.min_compressed_length))
        kv_mask = torch.zeros(num_heads, n, dtype=torch.bool)
        k = self.config.keep_boundary_tokens
        for head in range(num_heads):
            head_imp = importance[head]
            mid_imp = head_imp[k:n - k] if n > 2 * k else head_imp
            mid_budget = max(0, budget - 2 * k)
            if 0 < mid_budget < len(mid_imp):
                _, top_indices = torch.topk(mid_imp, mid_budget)
                for idx in top_indices:
                    kv_mask[head, k + idx.item()] = True
            elif len(mid_imp) > 0:
                kv_mask[head, k:n - k] = True
            for i in range(min(k, n)):
                kv_mask[head, i] = True
            for i in range(max(0, n - k), n):
                kv_mask[head, i] = True

        keep_indices = kv_mask.any(dim=0).nonzero(as_tuple=True)[0].tolist()
        compressed_ids = [input_ids[i] for i in keep_indices]

        return self._build_result(
            compressed_ids, n, (time.time() - start) * 1000,
            metadata={'mode': self.mode, 'num_heads': num_heads},
        )
