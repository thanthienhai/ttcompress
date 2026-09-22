"""SnapKVCompressor._compute_attention_importance used to call the model
once with output_attentions=True over the FULL context, which makes HF
return attention weights for every layer AND every query position at once --
at PCS's real context lengths (LongBench ~15,000 tokens) that is ~518GB
(36 layers x 32 heads x 15000^2 x 2 bytes), a guaranteed CUDA OOM (see
compression.py's docstring on the fix). The fix splits this into a prefill
pass (no attention retained) + a small windowed pass (KV-cache continuation,
attention retained only for the last `window_size` query positions).

This file proves that split is mathematically IDENTICAL to the original
single pass, using a real (tiny) multi-layer self-attention stand-in with
correct causal masking and KV-cache continuation -- not a mocked return
value, so the comparison is a genuine numerical check, not a tautology.
"""
from types import SimpleNamespace

import pytest
import torch

from ttcompress.compression import CompressionConfig, SnapKVCompressor


class _TinyAttentionLM:
    """A real (if tiny) causal self-attention stack: actual softmax(QK^T)V
    per layer, real causal masking, real KV-cache continuation across calls.
    Deterministic given `seed`, so a one-pass call and a two-pass
    (prefill + windowed) call over the SAME tokens can be compared exactly."""

    def __init__(self, num_layers=3, num_heads=2, head_dim=4, vocab_size=64, seed=0):
        # float64: isolates the refactor's LOGIC from float32 non-associativity
        # noise (splitting one batched matmul into two doesn't give
        # bit-identical float32 results even when mathematically equivalent,
        # and that noise compounds through several nonlinear layers -- at
        # float64 it drops to ~1e-10, confirming the split is exact).
        g = torch.Generator().manual_seed(seed)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        d = num_heads * head_dim
        self.embed = torch.randn(vocab_size, d, generator=g, dtype=torch.float64)
        self.Wq = [torch.randn(d, d, generator=g, dtype=torch.float64) for _ in range(num_layers)]
        self.Wk = [torch.randn(d, d, generator=g, dtype=torch.float64) for _ in range(num_layers)]
        self.Wv = [torch.randn(d, d, generator=g, dtype=torch.float64) for _ in range(num_layers)]

    def __call__(self, input_ids, attention_mask=None, past_key_values=None,
                 use_cache=True, output_attentions=False, cache_position=None):
        _, t = input_ids.shape
        h = self.embed[input_ids[0]]  # [T, D]
        past = past_key_values or [(None, None)] * self.num_layers
        new_past = []
        attentions = [] if output_attentions else None
        for layer in range(self.num_layers):
            q = (h @ self.Wq[layer]).view(t, self.num_heads, self.head_dim).transpose(0, 1)  # [H,T,d]
            k_new = (h @ self.Wk[layer]).view(t, self.num_heads, self.head_dim).transpose(0, 1)
            v_new = (h @ self.Wv[layer]).view(t, self.num_heads, self.head_dim).transpose(0, 1)
            past_k, past_v = past[layer]
            k = k_new if past_k is None else torch.cat([past_k, k_new], dim=1)
            v = v_new if past_v is None else torch.cat([past_v, v_new], dim=1)
            new_past.append((k, v))

            scores = (q @ k.transpose(-1, -2)) / (self.head_dim ** 0.5)  # [H,T,S]
            past_len = k.shape[1] - t
            causal = torch.zeros(t, k.shape[1], dtype=torch.float64)
            for i in range(t):
                causal[i, past_len + i + 1:] = float('-inf')
            attn = torch.softmax(scores + causal.unsqueeze(0), dim=-1)  # [H,T,S]
            if output_attentions:
                attentions.append(attn.unsqueeze(0))  # add batch dim -> [1,H,T,S]
            h = (attn @ v).transpose(0, 1).reshape(t, self.num_heads * self.head_dim)
        return SimpleNamespace(
            attentions=tuple(attentions) if attentions is not None else None,
            past_key_values=new_past,
        )


class _FakeTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        return ' '.join(str(i) for i in ids)


def _reference_importance(model, input_ids, window_size, kernel_size, pooling):
    """Replicates the ORIGINAL (pre-fix) single-pass formula exactly, as an
    independent oracle to compare the fixed two-pass code against."""
    from ttcompress.compression import _normalize
    import torch.nn.functional as F

    n = input_ids.shape[1]
    outputs = model(input_ids=input_ids, output_attentions=True, use_cache=True)
    attn_stack = torch.stack(outputs.attentions, dim=0)  # [L,1,H,S,S]
    num_layers_to_use = max(1, len(outputs.attentions) // 4)
    recent_attns = attn_stack[-num_layers_to_use:]
    window_start = max(0, n - window_size)
    window_attn = recent_attns[:, :, :, window_start:, :]
    importance = window_attn.mean(dim=0).sum(dim=2).mean(dim=0)
    if pooling in ('maxpool', 'avgpool') and kernel_size > 1:
        pool = F.max_pool1d if pooling == 'maxpool' else F.avg_pool1d
        importance = pool(importance.unsqueeze(0), kernel_size=kernel_size, stride=1, padding=kernel_size // 2).squeeze(0)
    for h in range(importance.shape[0]):
        importance[h] = _normalize(importance[h])
    return importance


@pytest.mark.parametrize('n_tokens,window_size', [(40, 8), (5, 8)])  # long context, and n <= window_size
def test_windowed_matches_single_pass_reference(n_tokens, window_size):
    model = _TinyAttentionLM(num_layers=8, seed=1)
    torch.manual_seed(0)
    input_ids = torch.randint(0, 64, (1, n_tokens))

    expected = _reference_importance(model, input_ids, window_size, kernel_size=5, pooling='maxpool')

    compressor = SnapKVCompressor(
        _FakeTokenizer(), model, CompressionConfig(target_ratio=4.0), device='cpu',
        window_size=window_size, kernel_size=5, mode='snapkv',
    )
    actual = compressor._compute_attention_importance(input_ids)

    # float64 model -> the two-pass split's only remaining "error" is
    # genuine floating-point noise (~1e-10, verified), not logic drift; a
    # real bug in the split would show up several orders of magnitude
    # larger (float32 non-associativity noise alone was ~1e-1 by layer 8).
    assert torch.allclose(actual, expected, atol=1e-8)


class _CallSpy:
    """Wraps a model to record output_attentions=True calls' shapes.
    (Assigning to `instance.__call__` doesn't work in Python -- dunder
    lookup goes through the type, not the instance -- so this wraps in a
    real class instead.)"""

    def __init__(self, inner):
        self.inner = inner
        self.captured = {}

    def __call__(self, *args, **kwargs):
        out = self.inner(*args, **kwargs)
        if kwargs.get('output_attentions'):
            self.captured['query_dim'] = out.attentions[0].shape[2]
            self.captured['key_dim'] = out.attentions[0].shape[3]
        return out


def test_windowed_pass_only_materializes_the_observation_window():
    """The actual point of the fix: when n > window_size, the returned
    attention tensors' query dimension must be `window_size`, not `n` --
    this is what keeps memory bounded at real context lengths."""
    n_tokens, window_size = 100, 12
    input_ids = torch.randint(0, 64, (1, n_tokens))
    spy = _CallSpy(_TinyAttentionLM(num_layers=4, seed=2))

    compressor = SnapKVCompressor(
        _FakeTokenizer(), spy, CompressionConfig(target_ratio=4.0), device='cpu',
        window_size=window_size, mode='snapkv',
    )
    compressor._compute_attention_importance(input_ids)
    captured = spy.captured

    assert captured['query_dim'] == window_size  # NOT n_tokens (100) -- the whole point of the fix
    assert captured['key_dim'] == n_tokens        # still attends over the full context


def test_compress_end_to_end_smoke():
    model = _TinyAttentionLM(num_layers=4, seed=3)
    input_ids = list(range(60))
    compressor = SnapKVCompressor(
        _FakeTokenizer(), model, CompressionConfig(target_ratio=4.0), device='cpu',
        window_size=8, max_capacity_prompt=32, mode='h2o',
    )
    result = compressor.compress(input_ids)
    assert 0 < result.compressed_length <= len(input_ids)
    assert result.compressed_ids == sorted(result.compressed_ids)  # original order preserved
