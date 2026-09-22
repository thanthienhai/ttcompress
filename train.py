#!/usr/bin/env python3
""""Training" flow for PCS = the lambda sweep (PCS_METHOD_SPEC.md §6/§11).

No gradient training happens here -- E6 is bolt-on, frozen (hard constraint
#1). This script sweeps the mixing weight lambda on a dev split drawn from
three published needle_in_haystack sources -- LongBench passage_retrieval_en,
RULER niah_single_1, classic Kamradt NIAH (see data/SOURCES.md) -- and picks
the single global lambda* that evaluate.py then runs officially on test --
lambda is fit once, never per task/ratio/source (constraint #3).
public_datasets.build_dev_test_split() guarantees the dev pool shares no
source document with the test pool evaluate.py uses.

    python train.py --reader-model Qwen/Qwen3-8B --encoder-path models/encoder_compressor

Without --encoder-path this runs with a synthetic relevance stand-in so the
sweep/curve/CI plumbing can be smoke-tested on a machine with no GPU and no
trained E6 checkpoint yet -- pass --relevance synthetic explicitly to
acknowledge that (the default requires --encoder-path and will refuse to run
without it, so a "real" run can never accidentally use fake relevance).
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

from ttcompress.metrics import token_f1_one
from ttcompress.pcs import PCSCompressor
from ttcompress.public_datasets import build_dev_test_split, save_dev_test_split
from ttcompress.reader import default_max_new_tokens, generate_answer, load_reader
from ttcompress.relevance import E6RelevanceProvider, SyntheticRelevanceProvider

LAMBDA_GRID = [round(0.1 * i, 1) for i in range(1, 10)]  # 9 interior points


def sweep(args) -> dict:
    dev, test = build_dev_test_split(
        num_dev_essays=args.num_dev_essays, samples_per_essay=args.samples_per_essay,
        longbench_num_dev=args.longbench_dev, seed=args.seed,
    )
    counts = Counter(s.metadata['dataset_source'] for s in dev)
    print(f"Dev split: {len(dev)} samples -- {dict(counts)}")

    # Persist the ACTUAL resolved samples (not just the args that produced
    # them) so evaluate.py can load the exact same test pool later instead of
    # re-deriving a split that silently drifts if any flag doesn't match
    # (PCS_METHOD_SPEC.md §7 dev/test contamination). --skip-split-write is
    # for scripts/shard_pipeline.py's multi-GPU sweep: every shard resolves
    # the IDENTICAL split (same seed/args), so the orchestrator writes it
    # once upfront and every shard just skips re-writing it, instead of N
    # processes racing to write the same file concurrently.
    if not args.skip_split_write:
        save_dev_test_split(dev, test, args.split_out)
        print(f"Dev/test split saved to {args.split_out} -- evaluate.py reads its 'test' half from here by default")

    model, tokenizer = load_reader(args.reader_model, device=args.device)
    if args.relevance == 'e6':
        if not args.encoder_path:
            raise SystemExit("--relevance e6 requires --encoder-path (a trained token-classifier checkpoint)")
        relevance = E6RelevanceProvider(tokenizer, args.encoder_path, device=args.device)
    else:
        print("[WARN] --relevance synthetic: pipeline smoke-test only, NOT a reportable curve (relevance.py docstring)")
        relevance = SyntheticRelevanceProvider(seed=args.seed)

    ratios = [float(r) for r in args.ratios.split(',')]
    # --lambdas restricts the sweep to a subset (scripts/shard_pipeline.py's
    # multi-GPU sweep: each GPU shard gets a disjoint slice of the grid).
    # Default is the full grid, identical to a plain single-process run.
    lambdas = [round(float(l), 1) for l in args.lambdas.split(',')] if args.lambdas else [0.0] + LAMBDA_GRID + [1.0]

    curve = {}
    for lam in lambdas:
        per_ratio = {}
        for ratio in ratios:
            compressor = PCSCompressor(tokenizer, model, lam=lam, relevance_provider=relevance)
            compressor.config.target_ratio = ratio
            scores = []
            for sample in dev:
                source = sample.metadata.get('dataset_source', 'generic')
                input_ids = tokenizer.encode(sample.context, add_special_tokens=False)
                result = compressor.compress(input_ids)
                answer = generate_answer(
                    model, tokenizer, result.compressed_ids, sample.query,
                    max_new_tokens=args.max_new_tokens or default_max_new_tokens(source),
                    prompt_kind=source,
                )
                scores.append(token_f1_one(answer or '', sample.reference_answer))
            per_ratio[ratio] = sum(scores) / len(scores) if scores else 0.0
        curve[lam] = {
            'per_ratio': per_ratio,
            'mean_token_f1': sum(per_ratio.values()) / len(per_ratio),
        }
        print(f"  lambda={lam:.1f}  mean_token_f1={curve[lam]['mean_token_f1']:.4f}  "
              f"({', '.join(f'{r}x={v:.4f}' for r, v in per_ratio.items())})")

    # Interior grid points actually present in `curve` -- the full set on a
    # plain run, possibly a strict subset on one shard of a multi-GPU sweep
    # (scripts/shard_pipeline.py's merge step recomputes the authoritative
    # chosen_lambda from every shard's combined curve; a lone shard's value
    # here is not meaningful on its own and is overwritten by that merge).
    interior_present = [l for l in LAMBDA_GRID if l in curve]
    best_lam = max(interior_present, key=lambda l: curve[l]['mean_token_f1']) if interior_present else None
    return {'curve': curve, 'chosen_lambda': best_lam, 'ratios': ratios, 'relevance_source': args.relevance}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--reader-model', required=True, help="HF causal LM id/path, e.g. Qwen/Qwen3-8B")
    ap.add_argument('--encoder-path', default=None, help="trained E6 token-classifier checkpoint")
    ap.add_argument('--relevance', choices=['e6', 'synthetic'], default='e6')
    ap.add_argument('--ratios', default='4,8')
    ap.add_argument('--lambdas', default=None,
                     help="comma list overriding the full 11-point grid (0.0..1.0 step 0.1) -- "
                          "scripts/shard_pipeline.py uses this to give each GPU shard a disjoint slice")
    ap.add_argument('--skip-split-write', action='store_true',
                     help="don't (re)write --split-out -- for shard_pipeline.py, where the orchestrator "
                          "writes it once upfront and every shard resolves an identical split anyway")
    ap.add_argument('--num-dev-essays', type=int, default=10, help="Paul Graham essays held out for dev RULER/Kamradt samples")
    ap.add_argument('--samples-per-essay', type=int, default=2)
    ap.add_argument('--longbench-dev', type=int, default=20, help="LongBench passage_retrieval_en rows held out for dev")
    ap.add_argument('--max-new-tokens', type=int, default=None,
                     help="overrides the per-source default (32 LongBench / 128 RULER / 32 Kamradt) for every source")
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default='results/lambda_sweep_dev.json')
    ap.add_argument('--split-out', default='results/dev_test_split.json',
                     help="where the resolved dev+test samples are saved for evaluate.py to load")
    args = ap.parse_args()

    result = sweep(args)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    if result['chosen_lambda'] is not None:
        print(f"\nChosen lambda* = {result['chosen_lambda']:.1f} (peak of the 9-point interior grid)")
        print(f"Curve saved to {args.out} -- pass --lam {result['chosen_lambda']:.1f} to evaluate.py")
    else:
        print(f"\nNo interior grid point in this run's --lambdas ({args.lambdas}) -- "
              f"partial curve saved to {args.out} (shard_pipeline.py merge-train picks the real chosen_lambda)")


if __name__ == '__main__':
    main()
