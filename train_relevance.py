#!/usr/bin/env python3
"""Stage C (OUTCOME_SUPERVISED_RELEVANCE_SPEC.md §4): train a regression
checkpoint (relevance_v2 from raw labels, relevance_v3 from residual labels)
by reusing the E6 architecture -- same base encoder +
AutoModelForTokenClassification head as
vncompress/scripts/train_encoder_compressor.py, but num_labels=1 and a
manually-computed MSE loss (see ttcompress/relevance_training.py's module
docstring for why the model's built-in classification loss doesn't apply to
continuous targets). This is the only step in the whole ttcompress project
that runs real gradient training -- everything else (PCS, the lambda sweep)
is bolt-on inference by design (PCS_METHOD_SPEC.md constraint #1).

    python train_relevance.py --label-variant raw \\
        --training-labels-dir training_labels/train_pilot --out-dir models/relevance_v2
    python train_relevance.py --label-variant residual \\
        --training-labels-dir training_labels/train_pilot --out-dir models/relevance_v3

Multilingual upgrade: if --training-labels-dir was built with
generate_labels.py --sources including longbench/ruler/kamradt (English)
alongside uit_viquad (Vietnamese), the base encoder MUST be multilingual --
the default is now xlm-roberta-base (not vinai/phobert-base, which is
Vietnamese-only and can't usefully tokenize the English chunks at all).
Pass --base-checkpoint vinai/phobert-base explicitly for a Vietnamese-only
run (--sources uit_viquad, matching this pipeline's original scope).

The resulting checkpoint is directly loadable by
ttcompress.relevance.RegressionRelevanceProvider (same save_pretrained/
from_pretrained convention E6RelevanceProvider already uses) -- NOT by
E6RelevanceProvider itself, whose softmax-over-2-classes scoring doesn't
apply to a 1-unit regression head.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random

import torch

from ttcompress.position_decompose import load_training_labels
from ttcompress.relevance_training import build_training_windows, collate_windows, encode_with_offsets

DEFAULT_BASE_CHECKPOINT = 'xlm-roberta-base'  # multilingual (Vietnamese + English); vinai/phobert-base is Vietnamese-only
DEFAULT_MAX_ENCODER_LEN = 256                 # same window vncompress/config.py::PHOBERT_MAX_ENCODER_LEN uses


def build_windows_for_documents(documents, label_variant, enc_tokenizer, max_len):
    windows = []
    for doc in documents:
        labels = doc.raw_label if label_variant == 'raw' else doc.residual_label
        text = '\n\n'.join(doc.chunks)
        ids, offsets = encode_with_offsets(enc_tokenizer, text)
        windows.extend(build_training_windows(ids, offsets, doc.chunks, labels, max_len))
    return windows


def masked_mse_loss(logits: torch.Tensor, labels: torch.Tensor, valid_mask: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """logits: [B,S,1] (the model's raw regression output, num_labels=1) ->
    squeezed to [B,S]; labels/valid_mask/attention_mask: [B,S]. Only
    positions that are both real tokens (attention_mask) and not a
    chunk-separator gap (valid_mask, see relevance_training.py) contribute."""
    logits = logits.squeeze(-1)
    mask = valid_mask.float() * attention_mask.float()
    sq_err = (logits - labels) ** 2 * mask
    denom = mask.sum().clamp_min(1.0)
    return sq_err.sum() / denom


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--label-variant', choices=['raw', 'residual'], required=True,
                     help="raw -> relevance_v2 checkpoint, residual -> relevance_v3 (spec §3/§4)")
    ap.add_argument('--training-labels-dir', required=True, help="output of decompose_labels.py")
    ap.add_argument('--base-checkpoint', default=DEFAULT_BASE_CHECKPOINT)
    ap.add_argument('--max-encoder-len', type=int, default=DEFAULT_MAX_ENCODER_LEN)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--grad-clip', type=float, default=1.0)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    from transformers import AutoModelForTokenClassification, AutoTokenizer

    paths = sorted(glob.glob(os.path.join(args.training_labels_dir, '*.jsonl')))
    if not paths:
        raise SystemExit(f"no .jsonl files in {args.training_labels_dir} -- run decompose_labels.py first")
    documents = [load_training_labels(p) for p in paths]
    print(f"Loaded {len(documents)} documents from {args.training_labels_dir}")

    enc_tokenizer = AutoTokenizer.from_pretrained(args.base_checkpoint, use_fast=True)
    # ignore_mismatched_sizes: --base-checkpoint may already carry a
    # classification head sized for a different num_labels (e.g. a prior E6
    # 2-class checkpoint); this re-initializes a fresh 1-unit regression head
    # instead of crashing. A pure backbone checkpoint (no existing head, e.g.
    # the default vinai/phobert-base) is unaffected either way.
    model = AutoModelForTokenClassification.from_pretrained(args.base_checkpoint, num_labels=1, ignore_mismatched_sizes=True)
    device = 'cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    windows = build_windows_for_documents(documents, args.label_variant, enc_tokenizer, args.max_encoder_len)
    print(f"{len(windows)} training windows (<= {args.max_encoder_len} tokens each) from {len(documents)} documents")
    if not windows:
        raise SystemExit("no training windows produced -- check --training-labels-dir isn't empty")

    pad_id = enc_tokenizer.pad_token_id or 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()

    for epoch in range(args.epochs):
        rng = random.Random(args.seed + epoch)
        order = list(range(len(windows)))
        rng.shuffle(order)
        epoch_loss, n_batches = 0.0, 0
        for start in range(0, len(order), args.batch_size):
            batch = [windows[i] for i in order[start:start + args.batch_size]]
            input_ids, attention_mask, labels, valid_mask = collate_windows(batch, pad_id)
            input_ids = torch.tensor(input_ids, device=device)
            attention_mask = torch.tensor(attention_mask, device=device)
            labels = torch.tensor(labels, device=device, dtype=torch.float)
            valid_mask = torch.tensor(valid_mask, device=device, dtype=torch.bool)

            out = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = masked_mse_loss(out.logits, labels, valid_mask, attention_mask)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
        print(f"epoch {epoch + 1}/{args.epochs}  mean_mse={epoch_loss / max(n_batches, 1):.4f}  ({n_batches} batches)")

    os.makedirs(args.out_dir, exist_ok=True)
    model.save_pretrained(args.out_dir)
    enc_tokenizer.save_pretrained(args.out_dir)
    meta = {
        'label_variant': args.label_variant, 'base_checkpoint': args.base_checkpoint,
        'max_encoder_len': args.max_encoder_len, 'epochs': args.epochs, 'batch_size': args.batch_size,
        'lr': args.lr, 'seed': args.seed, 'n_documents': len(documents), 'n_windows': len(windows),
        'training_labels_dir': args.training_labels_dir,
        'note': ('num_labels=1 regression head; load with '
                 'ttcompress.relevance.RegressionRelevanceProvider, NOT E6RelevanceProvider '
                 '(classification softmax path does not apply here).'),
    }
    with open(os.path.join(args.out_dir, 'relevance_regression_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Saved checkpoint + relevance_regression_meta.json to {args.out_dir}/")


if __name__ == '__main__':
    main()
