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
    # (PCS_METHOD_SPEC.md §7 dev/test contamination).
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
    lambdas = [0.0] + LAMBDA_GRID + [1.0]  # anchors included for the full curve, excluded from the pick below

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

    best_lam = max(LAMBDA_GRID, key=lambda l: curve[l]['mean_token_f1'])
    return {'curve': curve, 'chosen_lambda': best_lam, 'ratios': ratios, 'relevance_source': args.relevance}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--reader-model', required=True, help="HF causal LM id/path, e.g. Qwen/Qwen3-8B")
    ap.add_argument('--encoder-path', default=None, help="trained E6 token-classifier checkpoint")
    ap.add_argument('--relevance', choices=['e6', 'synthetic'], default='e6')
    ap.add_argument('--ratios', default='4,8')
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
    print(f"\nChosen lambda* = {result['chosen_lambda']:.1f} (peak of the 9-point interior grid)")
    print(f"Curve saved to {args.out} -- pass --lam {result['chosen_lambda']:.1f} to evaluate.py")


if __name__ == '__main__':
    main()
