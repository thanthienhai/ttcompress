"""Stage C prep — turn a constructed document's per-chunk labels (beta_c or
residual_c from Stage A/B) into per-E6-token regression targets, windowed to
the encoder's max length. The actual gradient training loop lives in
train_relevance.py (root); this module is the GPU-free, unit-testable part:
character-span bookkeeping and windowing, not the optimizer step.

OUTCOME_SUPERVISED_RELEVANCE_SPEC.md §4: "tái dùng kiến trúc E6" means the
same base encoder + AutoModelForTokenClassification head as
vncompress/scripts/train_encoder_compressor.py, but num_labels=1 (regression)
instead of 2 (classification) -- HF's built-in token-classification loss is
always CrossEntropyLoss regardless of num_labels and cannot consume
continuous float targets, so train_relevance.py computes MSE itself from
raw logits rather than passing `labels=` into the model call (confirmed by
reading that script; see its module docstring for the exact citation).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

CHUNK_SEP = '\n\n'  # matches build_document.ConstructedDocument.text()


def encode_with_offsets(enc_tokenizer, text: str) -> Tuple[List[int], List[Tuple[int, int]]]:
    """(encoder input_ids, char offsets), no special tokens, so ids and
    offsets align 1:1. Same fallback as
    ttcompress.relevance._BaseEncoderRelevanceProvider._encode_offsets, kept
    as a free function here (not shared via import) since this module must
    stay import-safe without an actual encoder instance around -- it only
    needs the tokenizer, not the model."""
    try:
        enc = enc_tokenizer(text, return_offsets_mapping=True, add_special_tokens=False, truncation=False)
        return list(enc['input_ids']), list(enc['offset_mapping'])
    except (TypeError, NotImplementedError, ValueError, KeyError):
        ids = enc_tokenizer.encode(text, add_special_tokens=False)
        offsets, cursor = [], 0
        for tid in ids:
            piece = enc_tokenizer.decode([tid])
            found = text.find(piece, cursor) if piece else -1
            if found < 0:
                offsets.append((cursor, cursor))
            else:
                offsets.append((found, found + len(piece)))
                cursor = found + len(piece)
        return ids, offsets


def char_spans_for_chunks(chunks: Sequence[str]) -> List[Tuple[int, int]]:
    """(start, end) character span of each chunk within
    CHUNK_SEP.join(chunks) -- the same string ConstructedDocument.text()
    produces."""
    spans = []
    cursor = 0
    for i, chunk in enumerate(chunks):
        start = cursor
        end = start + len(chunk)
        spans.append((start, end))
        cursor = end + (len(CHUNK_SEP) if i < len(chunks) - 1 else 0)
    return spans


def _chunk_index_for_offset(offset: Tuple[int, int], chunk_spans: Sequence[Tuple[int, int]]) -> int:
    """Which chunk a token's (start, end) char span belongs to, by its start
    offset -- -1 if it falls in a separator gap (e.g. a token that is purely
    the '\\n\\n' between two chunks) or outside every span."""
    start, end = offset
    if end <= start:
        return -1
    for i, (s, e) in enumerate(chunk_spans):
        if s <= start < e:
            return i
    return -1


@dataclass
class TokenLabelSequence:
    """One E6-tokenizer window's worth of training data."""
    input_ids: List[int]
    labels: List[float]        # target for each position; meaningless where valid_mask is False
    valid_mask: List[bool]     # False for tokens that fall in a chunk-separator gap


def assign_token_labels(
    enc_input_ids: Sequence[int], enc_offsets: Sequence[Tuple[int, int]],
    chunk_spans: Sequence[Tuple[int, int]], chunk_labels: Sequence[float],
) -> Tuple[List[float], List[bool]]:
    """Per-E6-token (labels, valid_mask) for one already-tokenized document.
    `chunk_labels[c]` is the regression target for chunk c (e.g. a
    normalized beta_c or residual_c value, see position_decompose.py)."""
    if len(chunk_spans) != len(chunk_labels):
        raise ValueError(f"chunk_spans ({len(chunk_spans)}) and chunk_labels ({len(chunk_labels)}) length mismatch")
    labels, valid = [], []
    for offset in enc_offsets:
        idx = _chunk_index_for_offset(offset, chunk_spans)
        if idx < 0:
            labels.append(0.0)
            valid.append(False)
        else:
            labels.append(float(chunk_labels[idx]))
            valid.append(True)
    return labels, valid


def window_sequence(
    input_ids: Sequence[int], labels: Sequence[float], valid_mask: Sequence[bool], max_len: int,
) -> List[TokenLabelSequence]:
    """Split one document's (possibly thousands of tokens) into
    non-overlapping windows of at most `max_len` E6 tokens each, one
    training example per window -- matches the encoder's own inference-time
    window size (PHOBERT_MAX_ENCODER_LEN, vncompress/config.py) so train and
    inference never see a different context length distribution."""
    n = len(input_ids)
    windows = []
    for start in range(0, n, max_len):
        end = min(start + max_len, n)
        windows.append(TokenLabelSequence(
            input_ids=list(input_ids[start:end]), labels=list(labels[start:end]),
            valid_mask=list(valid_mask[start:end]),
        ))
    return windows


def build_training_windows(
    enc_input_ids: Sequence[int], enc_offsets: Sequence[Tuple[int, int]],
    chunks: Sequence[str], chunk_labels: Sequence[float], max_len: int,
) -> List[TokenLabelSequence]:
    """End-to-end: one constructed document's E6 tokenization + per-chunk
    labels -> a list of fixed-size training windows."""
    chunk_spans = char_spans_for_chunks(chunks)
    labels, valid = assign_token_labels(enc_input_ids, enc_offsets, chunk_spans, chunk_labels)
    return window_sequence(enc_input_ids, labels, valid, max_len)


def collate_windows(windows: Sequence[TokenLabelSequence], pad_id: int):
    """Pad a batch of windows to the batch's max length. Returns
    (input_ids, attention_mask, labels, valid_mask) as plain nested lists
    (caller wraps in torch.tensor -- kept framework-free here so this stays
    testable without torch)."""
    width = max(len(w.input_ids) for w in windows)
    input_ids, attention_mask, labels, valid_mask = [], [], [], []
    for w in windows:
        pad_n = width - len(w.input_ids)
        input_ids.append(list(w.input_ids) + [pad_id] * pad_n)
        attention_mask.append([1] * len(w.input_ids) + [0] * pad_n)
        labels.append(list(w.labels) + [0.0] * pad_n)
        valid_mask.append(list(w.valid_mask) + [False] * pad_n)
    return input_ids, attention_mask, labels, valid_mask
