"""The amortized pruner: a query-aware cross-encoder that scores every chunk
of a document in one forward pass.

Input (per window):   [CLS] question [SEP] chunk_1 chunk_2 ... chunk_m [SEP]
Each chunk is tokenized on its own and concatenated, so chunk token spans
are exact. The encoder contextualizes the whole window jointly and each
chunk's score is a linear head on the MEAN of its token states ("late
chunking": pool after contextualization, not before). A document longer
than `max_len` is packed into several windows of whole chunks, each
repeating the question; every chunk appears in exactly one window.

Default backbone: BAAI/bge-reranker-v2-m3 (XLM-R large, 8k positions,
already trained for multilingual query-passage relevance -- the same
initialization XProvence starts from, so the comparison isolates the
supervision signal). The reranker's classification head is dropped; only
the encoder body is loaded.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

DEFAULT_BACKBONE = 'BAAI/bge-reranker-v2-m3'
HEAD_FILE = 'chunk_head.pt'
CONFIG_FILE = 'pruner_config.json'


@dataclass
class Window:
    input_ids: List[int]
    chunk_slot: List[int]      # per token: local slot (0..m-1) of its chunk, -1 for question/special tokens
    chunk_ids: List[int]       # local slot -> global chunk index


def _find(seq: Sequence[int], sub: Sequence[int], start: int = 0) -> int:
    for i in range(start, len(seq) - len(sub) + 1):
        if list(seq[i:i + len(sub)]) == list(sub):
            return i
    return -1


def pair_template(tokenizer):
    """(head, middle, tail): the special tokens the tokenizer puts around a (query, passage) pair --
    <s> | </s></s> | </s> for XLM-R / bge-reranker-v2-m3, [CLS] | [SEP] | [SEP] for BERT -- read off an
    encoded ('a', 'b') pair (build_inputs_with_special_tokens is gone in transformers 5). None if the
    tokenizer has no pair template."""
    cached = getattr(tokenizer, '_ttc_pair_template', 0)
    if cached != 0:
        return cached
    tmpl = None
    try:
        a = tokenizer.encode('a', add_special_tokens=False)
        b = tokenizer.encode('b', add_special_tokens=False)
        pair = list(tokenizer('a', 'b')['input_ids'])
        ia = _find(pair, a)
        ib = _find(pair, b, ia + len(a)) if ia >= 0 else -1
        if ia >= 0 and ib >= 0:
            tmpl = (pair[:ia], pair[ia + len(a):ib], pair[ib + len(b):])
    except Exception:  # noqa: BLE001 -- any tokenizer without a pair encoding: single-sequence format
        tmpl = None
    try:
        tokenizer._ttc_pair_template = tmpl
    except AttributeError:
        pass
    return tmpl


def pack_windows(tokenizer, question: str, chunks: Sequence[str], max_len: int = 4096,
                 max_query_tokens: int = 96, max_chunk_tokens: int = 512, pair_format: bool = True) -> List[Window]:
    """pair_format: frame each window the way the backbone saw (query, passage) pairs in its cross-encoder
    training (bge-reranker-v2-m3: <s> q </s></s> chunks </s>, like the reranker baseline's input). False = the
    pre-2026-09-28 single-sequence frame <s> q </s> chunks </s>, kept for checkpoints trained with it
    (PrunerScorer reads `pair_format` from pruner_config.json)."""
    cls_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
    sep_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
    q = tokenizer.encode(question, add_special_tokens=False)[:max_query_tokens]
    tmpl = pair_template(tokenizer) if pair_format else None
    if tmpl:
        head, middle, tail = tmpl
        prefix, suffix = list(head) + q + list(middle), list(tail)
    else:
        prefix, suffix = [cls_id] + q + [sep_id], [sep_id]
    capacity = max_len - len(prefix) - len(suffix)
    if capacity < 8:
        raise ValueError(f"max_len={max_len} leaves no room for chunks after a {len(prefix)}-token question")
    per_chunk = [tokenizer.encode(c, add_special_tokens=False)[:min(max_chunk_tokens, capacity)] or [sep_id]
                 for c in chunks]

    windows: List[Window] = []
    body, slots, ids = [], [], []

    def flush():
        if ids:
            windows.append(Window(prefix + body + suffix, [-1] * len(prefix) + slots + [-1] * len(suffix), list(ids)))

    for gi, toks in enumerate(per_chunk):
        if len(body) + len(toks) > capacity:
            flush()
            body, slots, ids = [], [], []
        slots.extend([len(ids)] * len(toks))
        body.extend(toks)
        ids.append(gi)
    flush()
    return windows


def compact_windows(windows: Sequence[Window]) -> List[Window]:
    """Same windows with token ids / slots as int32 arrays (~6x less RAM than
    lists of Python ints) -- for caching every training document's windows."""
    return [Window(np.asarray(w.input_ids, dtype=np.int32), np.asarray(w.chunk_slot, dtype=np.int32),
                   list(w.chunk_ids)) for w in windows]


def effective_max_len(encoder_config, requested: int) -> int:
    """Longest input the encoder accepts. RoBERTa-family models (XLM-R,
    bge-m3, bge-reranker-v2-m3) offset positions by padding_idx + 1, so
    max_position_embeddings=8194 means 8192 usable positions (514 -> 512)."""
    limit = getattr(encoder_config, 'max_position_embeddings', None)
    if not limit:
        return requested
    if getattr(encoder_config, 'model_type', '') in ('roberta', 'xlm-roberta', 'camembert'):
        limit -= (getattr(encoder_config, 'pad_token_id', 1) or 1) + 1
    return min(requested, limit)


class ChunkPruner(nn.Module):
    def __init__(self, encoder, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        hidden = encoder.config.hidden_size
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    @classmethod
    def from_backbone(cls, backbone: str = DEFAULT_BACKBONE, dropout: float = 0.1) -> 'ChunkPruner':
        from transformers import AutoModel
        return cls(AutoModel.from_pretrained(backbone), dropout)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, chunk_slot: torch.Tensor,
                num_slots: int) -> torch.Tensor:
        """-> [B, num_slots] chunk scores (logits); empty slots are 0 and
        must be masked by the caller (slot_mask from collate)."""
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        onehot = (chunk_slot.unsqueeze(-1) == torch.arange(num_slots, device=chunk_slot.device)).to(hidden.dtype)
        sums = torch.einsum('bsh,bsm->bmh', hidden, onehot)
        counts = onehot.sum(1).clamp_min(1.0).unsqueeze(-1)
        return self.head(sums / counts).squeeze(-1)

    def save_pretrained(self, out_dir: str, tokenizer=None, extra: Optional[dict] = None) -> None:
        os.makedirs(out_dir, exist_ok=True)
        self.encoder.save_pretrained(out_dir)
        torch.save(self.head.state_dict(), os.path.join(out_dir, HEAD_FILE))
        if tokenizer is not None:
            tokenizer.save_pretrained(out_dir)
        with open(os.path.join(out_dir, CONFIG_FILE), 'w', encoding='utf-8') as f:
            json.dump({'dropout': self.head[0].p, **(extra or {})}, f, ensure_ascii=False, indent=2)

    @classmethod
    def from_pretrained(cls, path: str) -> 'ChunkPruner':
        from transformers import AutoModel
        with open(os.path.join(path, CONFIG_FILE), encoding='utf-8') as f:
            cfg = json.load(f)
        model = cls(AutoModel.from_pretrained(path), cfg.get('dropout', 0.1))
        model.head.load_state_dict(torch.load(os.path.join(path, HEAD_FILE), map_location='cpu'))
        return model


def collate(windows: Sequence[Window], pad_id: int, device):
    width = max(len(w.input_ids) for w in windows)
    num_slots = max(len(w.chunk_ids) for w in windows)
    ids = torch.full((len(windows), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(windows), width), dtype=torch.long)
    slot = torch.full((len(windows), width), -1, dtype=torch.long)
    slot_mask = torch.zeros((len(windows), num_slots), dtype=torch.bool)
    for i, w in enumerate(windows):
        n = len(w.input_ids)
        ids[i, :n] = torch.as_tensor(w.input_ids)
        mask[i, :n] = 1
        slot[i, :n] = torch.as_tensor(w.chunk_slot)
        slot_mask[i, :len(w.chunk_ids)] = True
    return ids.to(device), mask.to(device), slot.to(device), slot_mask.to(device), num_slots


def document_scores(model: ChunkPruner, windows: Sequence[Window], num_chunks: int, pad_id: int, device,
                    window_batch: int = 4) -> torch.Tensor:
    """[num_chunks] logits for one document (keeps the autograd graph --
    the training loop calls this; inference wraps it in no_grad)."""
    out = [None] * num_chunks
    for start in range(0, len(windows), window_batch):
        batch = windows[start:start + window_batch]
        ids, mask, slot, _slot_mask, num_slots = collate(batch, pad_id, device)
        logits = model(ids, mask, slot, num_slots)
        for row, w in enumerate(batch):
            for local, gi in enumerate(w.chunk_ids):
                out[gi] = logits[row, local]
    return torch.stack(out)


def _looks_like_repo_id(path: str) -> bool:
    """'namespace/name' on the Hub vs a local path (absolute, relative with ./, or Windows)."""
    return (path.count('/') == 1 and '\\' not in path and ':' not in path
            and not path.startswith(('/', '.', '~')))


class PrunerScorer:
    """Inference wrapper: score_chunks(question, chunks) -> list of floats."""

    def __init__(self, path: str, device: str = 'cuda', max_len: Optional[int] = None, dtype: str = 'bfloat16'):
        from transformers import AutoTokenizer
        from .reader import resolve_device

        if not os.path.isdir(path):
            if not _looks_like_repo_id(path):
                raise FileNotFoundError(f"pruner checkpoint {path} does not exist -- did the train stage for it "
                                        f"finish? (see $RUN_ROOT/logs/train_<name>.log)")
            from huggingface_hub import snapshot_download  # a Hub repo id uploaded by scripts/upload_hf.py
            try:
                path = snapshot_download(path)
            except Exception as exc:
                raise FileNotFoundError(f"pruner {path!r} is neither a local directory nor a downloadable Hub repo "
                                        f"({type(exc).__name__}: {exc})") from exc
        if not os.path.exists(os.path.join(path, CONFIG_FILE)):
            raise FileNotFoundError(f"{path} has no {CONFIG_FILE}: not a finished pruner checkpoint "
                                    f"(train_pruner.py writes it together with the weights)")
        with open(os.path.join(path, CONFIG_FILE), encoding='utf-8') as f:
            cfg = json.load(f)
        self.max_len = max_len or cfg.get('max_len', 4096)
        self.pair_format = bool(cfg.get('pair_format', False))   # checkpoints from before the flag: old frame
        self.device = resolve_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = ChunkPruner.from_pretrained(path).to(self.device).eval()
        self.max_len = effective_max_len(self.model.encoder.config, self.max_len)
        # fp32 weights + autocast, exactly as train_pruner.py scores dev: casting the whole model to bf16
        # would give test different numerics (and more near-ties) than the dev metric the epoch was chosen on
        self.autocast_dtype = getattr(torch, dtype) if self.device != 'cpu' else None
        self.pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

    @torch.no_grad()
    def score_chunks(self, question: str, chunks: Sequence[str]) -> List[float]:
        windows = pack_windows(self.tokenizer, question, chunks, self.max_len, pair_format=self.pair_format)
        with torch.autocast(device_type='cuda', dtype=self.autocast_dtype or torch.bfloat16,
                            enabled=self.autocast_dtype is not None):
            scores = document_scores(self.model, windows, len(chunks), self.pad_id, self.device)
        return scores.float().cpu().tolist()
