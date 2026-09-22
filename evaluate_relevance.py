#!/usr/bin/env python3
"""Stage 6 (OUTCOME_SUPERVISED_RELEVANCE_SPEC.md §6): plug a Stage C checkpoint
(relevance_v2 or relevance_v3) into PCS's existing score(i) formula and run
the existing sweep/eval infrastructure -- ttcompress.registry.make_compressor,
ttcompress.metrics.bootstrap_mean_ci, PCSCompressor's mandatory lambda=1
equivalence guarantee (tests/test_pcs_equivalence.py) -- all reused verbatim,
not reimplemented, per spec §6's explicit instruction.

Multilingual, public-only upgrade: `sweep`'s --sources picks any mix of
UIT-ViQuAD, XQuAD-vi, VIMQA (Vietnamese) and the four English sources (their
own dev/tune pools) to pick lambda* on -- matching whatever mix
generate_labels.py used to train the checkpoint. `evaluate` always reports
the public Vietnamese test pool (UIT-ViQuAD + XQuAD-vi + VIMQA reserved
portions, ttcompress.vietnamese_public_test -- see that module for why
vcc_bench_v2.json, an unpublished internal benchmark, was dropped entirely)
and, unless --vietnamese-only, ALSO the English test pool
(ttcompress.public_datasets' reserved "test" half -- reused from
results/dev_test_split.json if train.py already wrote one, else regenerated
the same way evaluate.py itself falls back).

Two subcommands, matching the spec's train/dev/test discipline (§5):

  sweep    -- pick lambda* on documents built from dev/tune splits ONLY --
              NEVER either reserved test pool.
  evaluate -- the final numbers, both test pools, each touched exactly once.

    python evaluate_relevance.py sweep --relevance-checkpoint models/relevance_v2 \\
        --sources all --reader-model Qwen/Qwen3-8B --n 20 --out results/relevance_v2_sweep.json

    python evaluate_relevance.py evaluate --relevance-checkpoint models/relevance_v2 \\
        --reader-model Qwen/Qwen3-8B --lam 0.4 --out results/relevance_v2_official.json

Per-document prompt language follows dataset_source (reader.py's
'uit_viquad' Vietnamese template vs. the English per-source templates) --
never a single hardcoded language, now that sources can be mixed.
"""
from __future__ import annotations

import argparse
import json
import os

from ttcompress.data import NeedleSample
from ttcompress.metrics import bootstrap_mean_ci, needle_recall_one, token_f1_one
from ttcompress.multilingual_sources import ALL_SOURCES, select_documents
from ttcompress.reader import default_max_new_tokens, generate_answer, load_reader
from ttcompress.registry import make_compressor
from ttcompress.relevance import RegressionRelevanceProvider
from ttcompress.vietnamese_public_test import load as load_vietnamese_public_test

LAMBDA_GRID = [round(0.1 * i, 1) for i in range(1, 10)]  # 9 interior points, PCS_METHOD_SPEC.md §6
RATIOS = (4.0, 8.0)
DEFAULT_SPLIT_FILE = 'results/dev_test_split.json'  # written by train.py, PCS's own dev/test split


def _samples_from_documents(documents: list) -> list:
    return [
        NeedleSample(sample_id=d.doc_id, context=d.text(), query=d.question,
                     reference_answer=d.answer_text, doc_id=d.doc_id, metadata=d.metadata)
        for d in documents
    ]


def _prompt_kind(sample_or_doc) -> str:
    meta = sample_or_doc.metadata if hasattr(sample_or_doc, 'metadata') else {}
    return meta.get('dataset_source', 'uit_viquad')


def _run_one(sample, tokenizer, model, compressor):
    prompt_kind = _prompt_kind(sample)
    input_ids = tokenizer.encode(sample.context, add_special_tokens=False)
    result = compressor.compress(input_ids)
    answer = generate_answer(model, tokenizer, result.compressed_ids, sample.query,
                              max_new_tokens=default_max_new_tokens(prompt_kind), prompt_kind=prompt_kind)
    return token_f1_one(answer or '', sample.reference_answer), needle_recall_one(answer or '', sample.reference_answer)


def cmd_sweep(args):
    sources = list(ALL_SOURCES) if args.sources == 'all' else args.sources.split(',')
    documents = select_documents(sources, uit_viquad_split='dev', n=args.n, seed=args.seed)
    dev_samples = _samples_from_documents(documents)
    from collections import Counter
    print(f"Dev sweep: {len(dev_samples)} documents -- "
          f"{dict(Counter(s.metadata.get('dataset_source') for s in dev_samples))} "
          f"(never a reserved test pool)")

    model, tokenizer = load_reader(args.reader_model, device=args.device)
    relevance = RegressionRelevanceProvider(tokenizer, args.relevance_checkpoint, device=args.device)

    curve = {}
    for lam in [0.0] + LAMBDA_GRID + [1.0]:
        per_ratio = {}
        for ratio in RATIOS:
            compressor = make_compressor('encoder_pcs', ratio, tokenizer, model, relevance_provider=relevance, lam=lam)
            f1s = [_run_one(s, tokenizer, model, compressor)[0] for s in dev_samples]
            per_ratio[ratio] = sum(f1s) / len(f1s) if f1s else 0.0
        curve[lam] = {'per_ratio': per_ratio, 'mean_token_f1': sum(per_ratio.values()) / len(per_ratio)}
        print(f"  lambda={lam:.1f}  mean_token_f1={curve[lam]['mean_token_f1']:.4f}  "
              f"({', '.join(f'{r}x={v:.4f}' for r, v in per_ratio.items())})")

    best_lam = max(LAMBDA_GRID, key=lambda l: curve[l]['mean_token_f1'])
    result = {'curve': curve, 'chosen_lambda': best_lam, 'relevance_checkpoint': args.relevance_checkpoint,
              'sources': sources}
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nChosen lambda* = {best_lam:.1f}. Saved to {args.out}")


def _load_english_test_samples(split_file: str):
    if os.path.exists(split_file):
        from ttcompress.public_datasets import load_dev_test_split
        _dev, test = load_dev_test_split(split_file)
        print(f"English test pool loaded from {split_file} (train.py's saved split)")
        return test
    print(f"[WARN] {split_file} not found -- regenerating the English test pool from "
          f"public_datasets.build_dev_test_split() defaults instead of reusing train.py's saved split.")
    from ttcompress.public_datasets import build_dev_test_split
    _dev, test = build_dev_test_split()
    return test


def _evaluate_on(samples, tag, model, tokenizer, relevance, lam, out_results):
    docs_by_id = len(set(s.doc_id for s in samples))
    print(f"\n--- {tag}: {len(samples)} samples over {docs_by_id} documents ---")
    arms = {'encoder_pcs': lam, 'encoder': 0.0, 'truncation': None}
    out_results[tag] = {}
    for arm, arm_lam in arms.items():
        out_results[tag][arm] = {}
        for ratio in RATIOS:
            compressor = make_compressor(arm, ratio, tokenizer, model, relevance_provider=relevance, lam=arm_lam)
            f1s, nrs, docs = [], [], []
            for s in samples:
                f1, nr = _run_one(s, tokenizer, model, compressor)
                f1s.append(f1); nrs.append(nr); docs.append(s.doc_id)
            f1_mean, f1_lo, f1_hi = bootstrap_mean_ci(f1s, clusters=docs)
            nr_mean, nr_lo, nr_hi = bootstrap_mean_ci(nrs, clusters=docs)
            out_results[tag][arm][f'{ratio}x'] = {
                'token_f1': {'mean': f1_mean, 'ci95': [f1_lo, f1_hi]},
                'needle_recall': {'mean': nr_mean, 'ci95': [nr_lo, nr_hi]},
                'n': len(f1s), 'n_clusters': len(set(docs)),
            }
            print(f"  {arm}@{ratio}x: token_f1={f1_mean:.4f} 95% CI [{f1_lo:.4f}, {f1_hi:.4f}] (n={len(f1s)})")


def cmd_evaluate(args):
    model, tokenizer = load_reader(args.reader_model, device=args.device)
    relevance = RegressionRelevanceProvider(tokenizer, args.relevance_checkpoint, device=args.device)

    results = {}
    vi_samples = load_vietnamese_public_test(n=args.vi_n, seed=args.seed)
    print(f"Vietnamese public test pool (UIT-ViQuAD + XQuAD-vi + VIMQA, reserved portions): "
          f"{len(vi_samples)} samples over {len(set(s.doc_id for s in vi_samples))} documents")
    _evaluate_on(vi_samples, 'vietnamese_public_test_pool', model, tokenizer, relevance, args.lam, results)

    if not args.vietnamese_only:
        english_samples = _load_english_test_samples(args.split_file)
        _evaluate_on(english_samples, 'english_public_test_pool', model, tokenizer, relevance, args.lam, results)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump({'lam': args.lam, 'relevance_checkpoint': args.relevance_checkpoint, 'results': results},
                   f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)

    s = sub.add_parser('sweep', help="pick lambda* on dev/tune-split documents")
    s.add_argument('--relevance-checkpoint', required=True, help="relevance_v2 or relevance_v3 dir from train_relevance.py")
    s.add_argument('--sources', default='uit_viquad', help=f"comma list from {ALL_SOURCES}, or 'all'")
    s.add_argument('--reader-model', required=True)
    s.add_argument('--n', type=int, default=20)
    s.add_argument('--device', default='cuda')
    s.add_argument('--seed', type=int, default=0)
    s.add_argument('--out', default='results/relevance_lambda_sweep.json')
    s.set_defaults(func=cmd_sweep)

    e = sub.add_parser('evaluate', help="final numbers: Vietnamese public test pool always, English test pool unless --vietnamese-only")
    e.add_argument('--relevance-checkpoint', required=True)
    e.add_argument('--reader-model', required=True)
    e.add_argument('--lam', type=float, required=True, help="lambda* from `sweep`")
    e.add_argument('--vietnamese-only', action='store_true', help="skip the English test pool (Vietnamese-only checkpoint)")
    e.add_argument('--vi-n', type=int, default=120, help="Vietnamese test pool size, evenly split across UIT-ViQuAD/XQuAD-vi/VIMQA")
    e.add_argument('--split-file', default=DEFAULT_SPLIT_FILE, help="train.py's saved dev/test split to reuse for the English test pool")
    e.add_argument('--seed', type=int, default=1234)
    e.add_argument('--device', default='cuda')
    e.add_argument('--out', default='results/relevance_official.json')
    e.set_defaults(func=cmd_evaluate)

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
