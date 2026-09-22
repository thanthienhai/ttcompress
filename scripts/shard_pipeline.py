#!/usr/bin/env python3
"""Multi-GPU work partitioning for run_pipeline_multi_gpu.sh.

Neither train.py's lambda sweep nor evaluate.py's arm/ratio bench
parallelizes internally -- each is a single process pinned to one --device.
This script computes, for shard `i` of `N` total shards, which slice of work
that shard's train.py/evaluate.py invocation should run (`*-for-shard`), then
merges every shard's output JSON back into the exact file shape a single,
unsharded process would have produced (`merge-*`) -- so nothing downstream
(evaluate.py reading chosen_lambda, scripts/push_results.py reading
official_bench.json) needs to know sharding happened at all. Not meant to be
run standalone; see run_pipeline_multi_gpu.sh for how it's wired together.

    python scripts/shard_pipeline.py lambdas-for-shard --shard-index 0 --num-shards 4
    python scripts/shard_pipeline.py cells-for-shard --shard-index 0 --num-shards 4 --arms encoder_pcs,h2o,snapkv
    python scripts/shard_pipeline.py merge-train --shard-files a.json,b.json,c.json,d.json --out results/lambda_sweep_dev.json
    python scripts/shard_pipeline.py merge-eval --shard-files a.json,b.json,c.json,d.json --out results/official_bench.json

Partitioning is round-robin (item i goes to shard i % num_shards), not
contiguous blocks, so it stays reasonably balanced regardless of how work-unit
count (11 lambdas; 4 arm/ratio cells by default) relates to shard count -- a
shard can legitimately get zero items (prints nothing) when num_shards
exceeds the work-unit count; the orchestrator must skip launching a process
for an empty shard.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import LAMBDA_GRID  # noqa: E402 -- single source of truth for the grid, not duplicated here
from ttcompress.registry import ARM_RATIOS  # noqa: E402


def _partition(items, shard_index: int, num_shards: int) -> list:
    return [item for i, item in enumerate(items) if i % num_shards == shard_index]


def lambdas_for_shard(shard_index: int, num_shards: int) -> list:
    return _partition([0.0] + LAMBDA_GRID + [1.0], shard_index, num_shards)


def cells_for_shard(arms: list, shard_index: int, num_shards: int) -> list:
    cells = [(arm, ratio) for arm in arms for ratio in ARM_RATIOS[arm]]
    return _partition(cells, shard_index, num_shards)


def merge_train_curves(shard_data: list) -> dict:
    """`shard_data`: the already-loaded (json.load'd) dict from each shard's
    --out file. Returns the merged dict in train.py's exact single-process
    schema (curve/chosen_lambda/ratios/relevance_source)."""
    merged_curve = {}
    ratios = relevance_source = None
    for data in shard_data:
        merged_curve.update(data['curve'])
        ratios = ratios or data.get('ratios')
        relevance_source = relevance_source or data.get('relevance_source')

    # json.dump serializes float dict keys via str(), e.g. curve[0.1] ->
    # "0.1" -- every shard's file already went through that, so merged_curve
    # is keyed by those same strings; str(l) for l in LAMBDA_GRID (all
    # round(x, 1) values) matches them exactly.
    interior_present = [str(l) for l in LAMBDA_GRID if str(l) in merged_curve]
    if not interior_present:
        raise ValueError("No interior lambda grid point found across the given shards -- "
                          "did every shard actually run (and not crash before writing its --out)?")
    best = max(interior_present, key=lambda s: merged_curve[s]['mean_token_f1'])
    chosen_lambda = float(best)
    return {'curve': merged_curve, 'chosen_lambda': chosen_lambda, 'ratios': ratios, 'relevance_source': relevance_source}


def merge_eval_arms(shard_data: list) -> dict:
    """`shard_data`: the already-loaded dict from each shard's --out file.
    Returns the merged dict in evaluate.py's exact single-process schema
    (lam/arms)."""
    merged_arms = {}
    lam = None
    for data in shard_data:
        if lam is None:
            lam = data['lam']
        elif data['lam'] != lam:
            raise ValueError(f"lam={data['lam']} != {lam} from an earlier shard -- "
                              f"every shard must have been run with the same --lam")
        for arm, ratio_results in data['arms'].items():
            merged_arms.setdefault(arm, {}).update(ratio_results)
    return {'lam': lam, 'arms': merged_arms}


def _load_all(paths: str) -> list:
    loaded = []
    for path in paths.split(','):
        with open(path, encoding='utf-8') as f:
            loaded.append(json.load(f))
    return loaded


def _write_json(obj: dict, out_path: str):
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def cmd_lambdas_for_shard(args):
    print(','.join(str(l) for l in lambdas_for_shard(args.shard_index, args.num_shards)))


def cmd_cells_for_shard(args):
    for arm, ratio in cells_for_shard(args.arms.split(','), args.shard_index, args.num_shards):
        print(f"{arm} {ratio}")


def cmd_merge_train(args):
    try:
        result = merge_train_curves(_load_all(args.shard_files))
    except ValueError as exc:
        raise SystemExit(str(exc))
    _write_json(result, args.out)
    n = len(args.shard_files.split(','))
    print(f"Merged {n} shard(s) -> {args.out}")
    print(f"Chosen lambda* = {result['chosen_lambda']:.1f}")


def cmd_merge_eval(args):
    try:
        result = merge_eval_arms(_load_all(args.shard_files))
    except ValueError as exc:
        raise SystemExit(str(exc))
    _write_json(result, args.out)
    n = len(args.shard_files.split(','))
    cells = sum(len(v) for v in result['arms'].values())
    print(f"Merged {n} shard(s), {cells} arm/ratio cell(s) -> {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)

    p = sub.add_parser('lambdas-for-shard', help="print this shard's comma-separated lambda subset")
    p.add_argument('--shard-index', type=int, required=True)
    p.add_argument('--num-shards', type=int, required=True)
    p.set_defaults(func=cmd_lambdas_for_shard)

    p = sub.add_parser('cells-for-shard', help="print this shard's 'arm ratio' lines, one per assigned cell")
    p.add_argument('--shard-index', type=int, required=True)
    p.add_argument('--num-shards', type=int, required=True)
    p.add_argument('--arms', required=True, help="full arm list being swept, e.g. encoder_pcs,h2o,snapkv")
    p.set_defaults(func=cmd_cells_for_shard)

    p = sub.add_parser('merge-train', help="merge train.py --lambdas shard outputs into one lambda_sweep_dev.json")
    p.add_argument('--shard-files', required=True, help="comma list of shard --out paths")
    p.add_argument('--out', required=True)
    p.set_defaults(func=cmd_merge_train)

    p = sub.add_parser('merge-eval', help="merge evaluate.py --ratios shard outputs into one official_bench.json")
    p.add_argument('--shard-files', required=True, help="comma list of shard --out paths")
    p.add_argument('--out', required=True)
    p.set_defaults(func=cmd_merge_eval)

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
