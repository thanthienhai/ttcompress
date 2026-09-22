"""Mandatory unit test from PCS_METHOD_SPEC.md §2/§11:

    score(i, lambda=1) must select EXACTLY the same token set as
    TruncationCompressor, for every n/ratio tested -- not "close", identical.

This is checked before any other PCS number is trusted.
"""
import pytest

from ttcompress.compression import CompressionConfig, TruncationCompressor
from ttcompress.pcs import PCSCompressor, g

N_VALUES = [1, 2, 3, 4, 5, 6, 7, 10, 11, 15, 21, 50, 101, 199]
RATIOS = [1.5, 2.0, 3.0, 4.0, 8.0, 16.0]


@pytest.mark.parametrize('n', N_VALUES)
@pytest.mark.parametrize('ratio', RATIOS)
def test_lambda1_matches_truncation_exactly(tokenizer, n, ratio):
    input_ids = list(range(n))
    config = CompressionConfig(target_ratio=ratio)

    truncation = TruncationCompressor(tokenizer, config=config).compress(input_ids)
    pcs = PCSCompressor(tokenizer, config=config, lam=1.0).compress(input_ids)

    assert pcs.compressed_ids == truncation.compressed_ids
    assert pcs.compressed_length == truncation.compressed_length


def test_lambda1_needs_no_relevance_provider(tokenizer):
    # lambda=1 bypasses relevance entirely -- must not require one.
    PCSCompressor(tokenizer, lam=1.0).compress(list(range(20)))


def test_lambda_below_1_requires_relevance_provider(tokenizer):
    with pytest.raises(ValueError):
        PCSCompressor(tokenizer, lam=0.5, relevance_provider=None)


def test_g_bounds_and_symmetry():
    L = 21
    assert g(0, L) == 1.0
    assert g(L - 1, L) == 1.0
    for i in range(L):
        assert g(i, L) == pytest.approx(g(L - 1 - i, L))
        assert 0.0 <= g(i, L) <= 1.0
    mid = L // 2
    assert g(mid, L) < g(0, L)


def test_g_single_token():
    assert g(0, 1) == 1.0
