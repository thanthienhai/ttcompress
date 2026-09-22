"""Arm registry — PCS_METHOD_SPEC.md §6 scope: encoder / truncation /
encoder_pcs / h2o / snapkv on needle_in_haystack only. Nothing else."""
from __future__ import annotations

from typing import Optional

from .compression import BaseCompressor, CompressionConfig, SnapKVCompressor, TruncationCompressor
from .pcs import PCSCompressor, RelevanceProvider

ARMS = ('encoder', 'truncation', 'encoder_pcs', 'h2o', 'snapkv')

# h2o/snapkv are needle_in_haystack @ 8x only (PCS_METHOD_SPEC.md §6: "chỉ
# 8x để giữ ngân sách compute -- output_attentions=True trên reader 8B là
# O(S^2)/layer").
ARM_RATIOS = {
    'encoder': (4.0, 8.0),
    'truncation': (4.0, 8.0),
    'encoder_pcs': (4.0, 8.0),
    'h2o': (8.0,),
    'snapkv': (8.0,),
}


def make_compressor(
    arm: str, ratio: float, tokenizer, model=None,
    relevance_provider: Optional[RelevanceProvider] = None, lam: Optional[float] = None,
) -> BaseCompressor:
    config = CompressionConfig(target_ratio=ratio)
    if arm == 'truncation':
        return TruncationCompressor(tokenizer, model, config)
    if arm == 'encoder':
        return PCSCompressor(tokenizer, model, config, lam=0.0, relevance_provider=relevance_provider)
    if arm == 'encoder_pcs':
        if lam is None:
            raise ValueError("encoder_pcs requires --lam")
        return PCSCompressor(tokenizer, model, config, lam=lam, relevance_provider=relevance_provider)
    if arm == 'h2o':
        return SnapKVCompressor(tokenizer, model, config, mode='h2o')
    if arm == 'snapkv':
        return SnapKVCompressor(tokenizer, model, config, mode='snapkv')
    raise ValueError(f"unknown arm {arm!r}; choose from {ARMS}")
