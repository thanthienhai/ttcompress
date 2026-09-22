#!/usr/bin/env python3
"""Official bench (PCS_METHOD_SPEC.md §6/§7/§9/§11): run the chosen arms on
the test half of three published needle_in_haystack sources -- LongBench
passage_retrieval_en, RULER niah_single_1, classic Kamradt NIAH (see
data/SOURCES.md and public_datasets.build_dev_test_split) -- and report point
estimate + 95% CI via the document-clustered bootstrap
(metrics.bootstrap_mean_ci, clustered on doc_id). No TOST/equivalence claims
(hard constraint #4) -- point estimate + CI only, pooled and per-source.

    python train.py ...                          # writes results/dev_test_split.json
    python evaluate.py --reader-model Qwen/Qwen3-8B --encoder-path models/encoder_compressor --lam 0.4

By default this loads its test set from --split-file (train.py's saved
dev_test_split.json) rather than re-deriving one from --num-dev-essays etc.,
so it is structurally impossible for evaluate.py to score on documents
train.py's sweep saw. Pass --num-dev-essays/--samples-per-essay/
--longbench-dev/--split-seed (and omit --split-file) only to regenerate a
split standalone, e.g. for a smoke test with no prior train.py run --
regenerating independently reintroduces the leakage risk if the numbers
don't match whatever train.py used.

Default scope matches the checklist exactly: encoder_pcs @ {4x, 8x},
h2o + snapkv @ {8x} only. Pass --arms to add encoder/truncation as extra
reference points (both already exist as lambda=0/lambda=1 in encoder_pcs's
own curve, so they are optional here, not part of the required run).
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict

from ttcompress.metrics import bootstrap_mean_ci, needle_recall_one, token_f1_one
from ttcompress.public_datasets import build_dev_test_split, load_dev_test_split
from ttcompress.reader import default_max_new_tokens, generate_answer, load_reader
from ttcompress.registry import ARM_RATIOS, make_compressor
from ttcompress.relevance import E6RelevanceProvider, SyntheticRelevanceProvider

DEFAULT_ARMS = ['encoder_pcs', 'h2o', 'snapkv']
# SnapKVCompressor needs real attention weights (see reader.load_reader
# docstring) -- forced automatically whenever one of these arms is requested.
ATTENTION_ARMS = {'h2o', 'snapkv'}


def _ci_block(values, clusters):
    mean, lo, hi = bootstrap_mean_ci(values, clusters=clusters)
    return {'mean': mean, 'ci95': [lo, hi], 'n': len(values), 'n_clusters': len(set(clusters))}


def run_arm(arm, ratio, samples, tokenizer, model, relevance, lam, max_new_tokens):
    compressor = make_compressor(arm, ratio, tokenizer, model, relevance_provider=relevance, lam=lam)
    per_source = defaultdict(lambda: {'f1': [], 'nr': [], 'doc': []})
    f1s, nrs, docs = [], [], []
    for sample in samples:
        source = sample.metadata.get('dataset_source', 'generic')
        input_ids = tokenizer.encode(sample.context, add_special_tokens=False)
        result = compressor.compress(input_ids)
        answer = generate_answer(
            model, tokenizer, result.compressed_ids, sample.query,
            max_new_tokens=max_new_tokens or default_max_new_tokens(source), prompt_kind=source,
        )
        f1 = token_f1_one(answer or '', sample.reference_answer)
        nr = needle_recall_one(answer or '', sample.reference_answer)
        f1s.append(f1); nrs.append(nr); docs.append(sample.doc_id)
        per_source[source]['f1'].append(f1)
        per_source[source]['nr'].append(nr)
        per_source[source]['doc'].append(sample.doc_id)
    return {
        'pooled': {'token_f1': _ci_block(f1s, docs), 'needle_recall': _ci_block(nrs, docs)},
        'by_source': {
            src: {'token_f1': _ci_block(v['f1'], v['doc']), 'needle_recall': _ci_block(v['nr'], v['doc'])}
            for src, v in per_source.items()
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--reader-model', required=True)
    ap.add_argument('--encoder-path', default=None)
    ap.add_argument('--relevance', choices=['e6', 'synthetic'], default='e6')
    ap.add_argument('--lam', type=float, required=True, help="lambda* chosen by train.py's dev sweep")
    ap.add_argument('--arms', default=','.join(DEFAULT_ARMS))
    ap.add_argument('--split-file', default='results/dev_test_split.json',
                     help="load the test half from here (written by train.py); falls back to "
                          "regenerating from --num-dev-essays et al. if this path doesn't exist")
    ap.add_argument('--num-dev-essays', type=int, default=10, help="regeneration fallback only -- must match train.py's run")
    ap.add_argument('--samples-per-essay', type=int, default=2, help="regeneration fallback only -- must match train.py's run")
    ap.add_argument('--longbench-dev', type=int, default=20, help="regeneration fallback only -- must match train.py's run")
    ap.add_argument('--split-seed', type=int, default=1234, help="regeneration fallback only -- must match train.py's run")
    ap.add_argument('--max-new-tokens', type=int, default=None,
                     help="overrides the per-source default (32 LongBench / 128 RULER / 32 Kamradt) for every source")
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--attn-implementation', default=None,
                     help="forced to 'eager' automatically when --arms includes h2o/snapkv; set explicitly to override")
    ap.add_argument('--out', default='results/official_bench.json')
    args = ap.parse_args()

    arms = args.arms.split(',')

    if os.path.exists(args.split_file):
        _dev, samples = load_dev_test_split(args.split_file)
        print(f"Test set loaded from {args.split_file} (train.py's saved split -- no leakage risk)")
    else:
        print(f"[WARN] {args.split_file} not found -- regenerating a split from --num-dev-essays et al. "
              f"instead of loading train.py's saved one. This ONLY matches train.py's dev pool if every "
              f"one of --num-dev-essays/--samples-per-essay/--longbench-dev/--split-seed is identical to "
              f"what train.py used -- otherwise dev documents can silently leak into this test run.")
        _dev, samples = build_dev_test_split(
            num_dev_essays=args.num_dev_essays, samples_per_essay=args.samples_per_essay,
            longbench_num_dev=args.longbench_dev, seed=args.split_seed,
        )
    counts = Counter(s.metadata['dataset_source'] for s in samples)
    print(f"Test set: {len(samples)} samples over {len(set(s.doc_id for s in samples))} documents -- {dict(counts)}")

    attn_implementation = args.attn_implementation
    if attn_implementation is None and ATTENTION_ARMS.intersection(arms):
        attn_implementation = 'eager'
        print(f"[INFO] --arms includes {sorted(ATTENTION_ARMS.intersection(arms))}: "
              f"loading reader with attn_implementation='eager' (required for real attention weights)")
    model, tokenizer = load_reader(args.reader_model, device=args.device, attn_implementation=attn_implementation)
    if args.relevance == 'e6':
        if not args.encoder_path:
            raise SystemExit("--relevance e6 requires --encoder-path (a trained token-classifier checkpoint)")
        relevance = E6RelevanceProvider(tokenizer, args.encoder_path, device=args.device)
    else:
        print("[WARN] --relevance synthetic: NOT a reportable number, smoke-test only")
        relevance = SyntheticRelevanceProvider()

    results = {}
    for arm in arms:
        results[arm] = {}
        for ratio in ARM_RATIOS[arm]:
            print(f"\n=== {arm} @ {ratio}x ===")
            stats = run_arm(arm, ratio, samples, tokenizer, model, relevance, args.lam, args.max_new_tokens)
            results[arm][f'{ratio}x'] = stats
            f1 = stats['pooled']['token_f1']
            print(f"  pooled token_f1 = {f1['mean']:.4f}  95% CI [{f1['ci95'][0]:.4f}, {f1['ci95'][1]:.4f}]  "
                  f"(n={f1['n']}, clusters={f1['n_clusters']})")
            for src, src_stats in stats['by_source'].items():
                sf1 = src_stats['token_f1']
                print(f"    {src}: token_f1 = {sf1['mean']:.4f}  95% CI [{sf1['ci95'][0]:.4f}, {sf1['ci95'][1]:.4f}]  (n={sf1['n']})")

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump({'lam': args.lam, 'arms': results}, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == '__main__':
    main()
