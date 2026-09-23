#!/usr/bin/env python3
"""Stage B -- distill attribution labels into the one-pass chunk pruner
(METHOD_SPEC.md §4).

    python train_pruner.py --label-source beta \\
        --train-labels labels/fit/Qwen--Qwen3-8B/f1/uit_viquad_train,labels/fit/Qwen--Qwen3-8B/f1/vimqa_train \\
        --dev-labels   labels/fit/Qwen--Qwen3-8B/f1/uit_viquad_dev,labels/fit/Qwen--Qwen3-8B/f1/vimqa_dev \\
        --out-dir models/pruner_beta_qwen3-8b

    # answer-span-supervised control (same docs, same backbone, binary gold label)
    python train_pruner.py --label-source span --train-labels ... --dev-labels ... --out-dir models/pruner_span

Model selection is reader-free: after each epoch the pruner ranks the dev
documents' chunks and is scored against the dev labels (--select-metric);
the best epoch is kept. No test data and no reader calls are involved.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import random
import time

import torch

from ttcompress.attribution import fit_position_prior
from ttcompress.pruner import DEFAULT_BACKBONE, ChunkPruner, document_scores, effective_max_len, pack_windows
from ttcompress.pruner_training import (
    example_loss, load_label_dirs, make_examples, mean_metrics, ranking_metrics,
)
from ttcompress.reader import resolve_device


def evaluate_dev(model, tokenizer, examples, args, device, pad_id):
    model.eval()
    rows = []
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=args.bf16):
        for ex in examples:
            windows = pack_windows(tokenizer, ex.question, ex.chunks, args.max_len)
            scores = document_scores(model, windows, len(ex.chunks), pad_id, device).float().cpu().tolist()
            rows.append(ranking_metrics(scores, ex))
    model.train()
    return mean_metrics(rows)


def main():
    try:  # Vietnamese text / arrows on a non-UTF-8 console (Windows cp1252) must not crash a run
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--label-source', choices=['beta', 'ensemble', 'span'], required=True,
                    help="beta/ensemble: point --train-labels at single-reader or ensemble fit dirs")
    ap.add_argument('--train-labels', required=True, help="comma list of fit dirs")
    ap.add_argument('--dev-labels', required=True, help="comma list of fit dirs (dev split)")
    ap.add_argument('--position-adjust', action='store_true', help="subtract the pooled position prior (ablation)")
    ap.add_argument('--backbone', default=DEFAULT_BACKBONE)
    ap.add_argument('--max-len', type=int, default=4096)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--head-lr', type=float, default=1e-3)
    ap.add_argument('--warmup', type=float, default=0.06)
    ap.add_argument('--weight-decay', type=float, default=0.01)
    ap.add_argument('--docs-per-step', type=int, default=8, help="gradient accumulation over documents")
    ap.add_argument('--tau', type=float, default=1.0, help="listnet target temperature")
    ap.add_argument('--w-mse', type=float, default=0.5)
    ap.add_argument('--grad-clip', type=float, default=1.0)
    ap.add_argument('--bf16', action='store_true', default=True)
    ap.add_argument('--no-bf16', dest='bf16', action='store_false')
    ap.add_argument('--grad-checkpointing', action='store_true')
    ap.add_argument('--select-metric', default=None,
                    help="dev metric to maximize (default: ndcg@3_beta for beta/ensemble, gold_recall@25%% for span)")
    ap.add_argument('--max-train-docs', type=int, default=None)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    from transformers import AutoTokenizer

    train_labels = load_label_dirs(args.train_labels.split(','))
    dev_labels = load_label_dirs(args.dev_labels.split(','))
    prior = fit_position_prior(train_labels) if args.position_adjust else None
    train = make_examples(train_labels, args.label_source, prior)
    # dev keeps the unadjusted label: model selection targets the real attribution
    dev = make_examples(dev_labels, 'span' if args.label_source == 'span' else 'beta')
    if args.max_train_docs:
        train = train[:args.max_train_docs]
    if not train:
        raise SystemExit("no usable training documents (all uninformative?)")
    metric = args.select_metric or ('gold_recall@25%' if args.label_source == 'span' else 'ndcg@3_beta')
    print(f"train: {len(train)} docs ({len(train_labels) - len(train)} dropped as uninformative); dev: {len(dev)} docs; "
          f"selecting on dev {metric}")

    device = resolve_device(args.device)
    args.bf16 = args.bf16 and device.startswith('cuda')
    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    model = ChunkPruner.from_backbone(args.backbone).to(device)
    if effective_max_len(model.encoder.config, args.max_len) < args.max_len:
        args.max_len = effective_max_len(model.encoder.config, args.max_len)
        print(f"--max-len capped to {args.max_len} (encoder position limit)")
    if args.grad_checkpointing:
        try:  # non-reentrant: correct gradients whatever requires_grad the checkpointed inputs have
            model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        except TypeError:  # transformers < 4.35
            model.encoder.gradient_checkpointing_enable()
    model.train()

    params = [{'params': model.encoder.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay},
              {'params': model.head.parameters(), 'lr': args.head_lr, 'weight_decay': 0.0}]
    optimizer = torch.optim.AdamW(params)
    total_steps = math.ceil(len(train) / args.docs_per_step) * args.epochs
    warmup = max(1, int(args.warmup * total_steps))
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / warmup) * max(0.0, (total_steps - s) / max(1, total_steps - warmup)))

    windows_cache = {}
    best, log = -float('inf'), []
    step = 0
    for epoch in range(args.epochs):
        order = list(range(len(train)))
        random.Random(args.seed + epoch).shuffle(order)
        running, n_docs, t0 = 0.0, 0, time.time()
        optimizer.zero_grad()
        for pos, i in enumerate(order):
            ex = train[i]
            if ex.doc_id not in windows_cache:
                windows_cache[ex.doc_id] = pack_windows(tokenizer, ex.question, ex.chunks, args.max_len)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=args.bf16):
                scores = document_scores(model, windows_cache[ex.doc_id], len(ex.chunks), pad_id, device)
            loss = example_loss(scores.float(), ex, args.label_source, args.tau, args.w_mse) / args.docs_per_step
            loss.backward()
            running += loss.item() * args.docs_per_step
            n_docs += 1
            if (pos + 1) % args.docs_per_step == 0 or pos == len(order) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                schedule.step()
                optimizer.zero_grad()
                step += 1
                if step % 50 == 0:
                    print(f"  epoch {epoch + 1} step {step}/{total_steps} loss {running / n_docs:.4f} "
                          f"({(time.time() - t0) / n_docs:.2f}s/doc)", flush=True)
        dev_metrics = evaluate_dev(model, tokenizer, dev, args, device, pad_id)
        entry = {'epoch': epoch + 1, 'train_loss': running / max(1, n_docs), 'dev': dev_metrics}
        log.append(entry)
        print(f"epoch {epoch + 1}: train_loss={entry['train_loss']:.4f} dev={json.dumps(dev_metrics)}", flush=True)
        score = dev_metrics.get(metric, float('nan'))
        if score == score and score > best:
            best = score
            model.save_pretrained(args.out_dir, tokenizer, extra={
                'max_len': args.max_len, 'backbone': args.backbone, 'label_source': args.label_source,
                'train_labels': args.train_labels, 'position_adjust': args.position_adjust,
                'best_epoch': epoch + 1, 'select_metric': metric, 'best_dev': dev_metrics})
            print(f"  saved (best dev {metric}={best:.4f}) -> {args.out_dir}")

    if best == -float('inf'):  # dev metric undefined every epoch (e.g. no informative dev docs): keep the last
        print(f"[WARN] dev {metric} was never defined; saving the final epoch instead of the best one")
        model.save_pretrained(args.out_dir, tokenizer, extra={
            'max_len': args.max_len, 'backbone': args.backbone, 'label_source': args.label_source,
            'train_labels': args.train_labels, 'position_adjust': args.position_adjust,
            'best_epoch': args.epochs, 'select_metric': None})
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'train_log.json'), 'w', encoding='utf-8') as f:
        json.dump({'args': vars(args), 'n_train': len(train), 'n_dev': len(dev), 'log': log}, f, indent=2)


if __name__ == '__main__':
    main()
