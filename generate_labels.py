#!/usr/bin/env python3
"""Stage A -- utility-attribution labels (METHOD_SPEC.md §3).

    # 1. measure (GPU): masks -> reader answers -> F1 (+ answer log-prob). One
    #    process per GPU; shards split documents round-robin; reruns skip
    #    documents already on disk, so a killed job just resumes.
    CUDA_VISIBLE_DEVICES=0 python generate_labels.py measure --source uit_viquad --split train --n 4000 \\
        --reader-model Qwen/Qwen3-8B --backend vllm --shard 0 --num-shards 4

    # 2. fit (CPU): one global alpha by CV on the dev measurements, then ridge
    #    per document + label-quality diagnostics (summary.json)
    python generate_labels.py fit --raw-dir labels/raw/Qwen--Qwen3-8B/uit_viquad_train \\
        --alpha-from labels/raw/Qwen--Qwen3-8B/uit_viquad_dev --target f1 \\
        --out-dir labels/fit/Qwen--Qwen3-8B/f1/uit_viquad_train

    # 3. ensemble (CPU): reader-agnostic label = mean within-document z-score
    python generate_labels.py ensemble --fit-dirs labels/fit/A/f1/uit_viquad_train,labels/fit/B/f1/uit_viquad_train \\
        --out-dir labels/fit/ensemble/f1/uit_viquad_train
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

from ttcompress.attribution import (
    MaskOutcomes, ensemble_labels, fit_document, fit_position_prior, load_chunk_labels, load_mask_outcomes,
    masks_for_document, position_r2, record_paths, save_record, select_alpha,
)
from ttcompress.metrics import spearman, token_f1
from ttcompress.reader import MAX_NEW_TOKENS, load_reader, reader_tag
from ttcompress.sources import SOURCES, load_documents


def add_document_args(p):
    p.add_argument('--source', choices=SOURCES, required=True)
    p.add_argument('--split', choices=['train', 'dev', 'test'], required=True)
    p.add_argument('--n', type=int, default=None, help="documents (hash-chosen, nested across n); default all")
    p.add_argument('--haystack-chars', type=int, default=30000, help="single-hop haystack length")
    p.add_argument('--distractors', choices=['random', 'hard'], default='random')
    p.add_argument('--multihop-pad-chars', type=int, default=0, help="lengthen multi-hop docs with easy distractors")


def cmd_measure(args):
    docs = load_documents(args.source, args.split, args.n, args.haystack_chars, args.distractors,
                          args.multihop_pad_chars)
    docs = [d for i, d in enumerate(docs) if i % args.num_shards == args.shard]
    out_dir = os.path.join(args.out_root, reader_tag(args.reader_model), f'{args.source}_{args.split}')
    todo = [d for d in docs if not os.path.exists(os.path.join(out_dir, f'{d.doc_id}.json'))]
    print(f"{args.source}/{args.split} shard {args.shard}/{args.num_shards}: {len(docs)} docs, "
          f"{len(docs) - len(todo)} already measured, {len(todo)} to go -> {out_dir}")
    if not todo:
        return
    keep_rates = [float(x) for x in args.keep_rates.split(',')]
    outcomes = set(args.outcomes.split(','))
    # a resumed run must measure with exactly the settings the records on disk were measured with
    config = {'keep_rates': keep_rates, 'k_min': args.k_min, 'k_per_chunk': args.k_per_chunk, 'k_max': args.k_max,
              'outcomes': sorted(outcomes), 'haystack_chars': args.haystack_chars, 'distractors': args.distractors,
              'multihop_pad_chars': args.multihop_pad_chars, 'max_new_tokens': args.max_new_tokens}
    os.makedirs(out_dir, exist_ok=True)
    config_path = os.path.join(out_dir, 'measure_config.json')
    if os.path.exists(config_path):
        with open(config_path, encoding='utf-8') as f:
            on_disk = json.load(f)
        if on_disk != config:
            raise SystemExit(f"{config_path} was measured with {on_disk}, this run asks for {config}; "
                             f"use another --out-root instead of mixing settings in one label set")
    else:
        tmp = f'{config_path}.{args.shard}.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        os.replace(tmp, config_path)
    reader = load_reader(args.reader_model, args.backend, args.device, args.dtype, args.batch_size, args.max_model_len,
                         args.tp, args.gpu_memory_utilization)

    total_calls, t0 = 0, time.time()
    for g in range(0, len(todo), args.docs_per_call):
        group = todo[g:g + args.docs_per_call]
        prompts, owners = [], []
        per_doc_masks = []
        for di, d in enumerate(group):
            masks = masks_for_document(d.doc_id, d.num_chunks, keep_rates, args.k_min, args.k_per_chunk, args.k_max)
            per_doc_masks.append(masks)
            for m in masks:
                prompts.append(reader.build_prompt(d.text([i for i, k in enumerate(m) if k]), d.question, d.language))
                owners.append(di)
            prompts.append(reader.build_prompt(d.text(), d.question, d.language))  # full context, last
            owners.append(di)
        start = time.time()
        # one generate call per hop type (budgets differ)
        answers = [''] * len(prompts)
        for hop in {d.hop for d in group}:
            idx = [i for i, o in enumerate(owners) if group[o].hop == hop]
            for i, a in zip(idx, reader.generate([prompts[i] for i in idx], args.max_new_tokens or MAX_NEW_TOKENS[hop])):
                answers[i] = a
        logprobs = None
        if 'logprob' in outcomes:
            logprobs = reader.answer_logprob(prompts, [group[o].answers[0] for o in owners])
        elapsed = time.time() - start
        total_calls += len(prompts)

        cursor = 0
        for di, d in enumerate(group):
            n = len(per_doc_masks[di]) + 1
            ans = answers[cursor:cursor + n]
            f1 = [token_f1(a, d.answers) for a in ans]
            lp = logprobs[cursor:cursor + n] if logprobs is not None else None
            save_record(MaskOutcomes(
                doc=d.to_dict(), reader=reader_tag(args.reader_model), masks=per_doc_masks[di], f1=f1[:-1],
                logprob=lp[:-1] if lp else None, full_f1=f1[-1], full_logprob=lp[-1] if lp else None,
                full_answer=ans[-1], seconds=elapsed * n / len(prompts),
            ), out_dir)
            cursor += n
        done = g + len(group)
        rate = (time.time() - t0) / total_calls * 1000
        print(f"  [{done}/{len(todo)}] {len(prompts)} reader calls in {elapsed:.1f}s; "
              f"running mean {rate:.0f} ms/call; truncated prompts so far: {reader.n_truncated}", flush=True)


def _summary_stats(labels, alpha_info=None):
    inf = [lab for lab in labels if lab.informative]
    by_hop = defaultdict(list)
    for lab in inf:
        by_hop[lab.doc['hop']].append(lab)
    prior = fit_position_prior(labels)
    out = {
        'n_docs': len(labels), 'n_informative': len(inf),
        'mean_full_f1': float(np.nanmean([lab.full_f1 for lab in labels])) if labels else float('nan'),
        'mean_cv_r2': float(np.nanmean([lab.cv_r2 for lab in inf])) if inf else float('nan'),
        'gold_recall_at_g': float(np.nanmean([lab.diagnostics['gold_recall_at_g'] for lab in inf])) if inf else float('nan'),
        'gold_mrr': float(np.nanmean([lab.diagnostics['gold_mrr'] for lab in inf])) if inf else float('nan'),
        'position_prior_by_decile': prior, 'position_r2': position_r2(labels, prior),
        'by_hop': {h: {'n': len(v), 'mean_cv_r2': float(np.nanmean([x.cv_r2 for x in v])),
                       'gold_recall_at_g': float(np.nanmean([x.diagnostics['gold_recall_at_g'] for x in v]))}
                   for h, v in by_hop.items()},
    }
    if alpha_info:
        out.update(alpha_info)
    return out


def cmd_fit(args):
    paths = record_paths(args.raw_dir)
    if not paths:
        raise SystemExit(f"no measure output in {args.raw_dir}")
    alpha_info = {}
    if args.alpha is not None:
        alpha = args.alpha
    else:
        if not args.alpha_from:
            raise SystemExit("pass --alpha, or --alpha-from <dev measure dir> (alpha is chosen on dev, never on train)")
        dev = [load_mask_outcomes(p) for p in record_paths(args.alpha_from)]
        if not dev:
            raise SystemExit(f"no measure output in {args.alpha_from}")
        grid = [float(a) for a in args.alpha_grid.split(',')]
        alpha, scores = select_alpha([(r.masks, r.f1 if args.target == 'f1' else r.logprob) for r in dev], grid)
        alpha_info = {'alpha': alpha, 'alpha_cv_mse': scores, 'alpha_dev_dir': args.alpha_from, 'n_alpha_docs': len(dev)}
        print(f"alpha={alpha} by dev CV over {len(dev)} docs: {scores}")
    labels = []
    for p in paths:
        lab = fit_document(load_mask_outcomes(p), args.target, alpha, args.n_boot)
        save_record(lab, args.out_dir)
        labels.append(lab)
    summary = _summary_stats(labels, alpha_info or {'alpha': alpha})
    summary.update({'target': args.target, 'raw_dir': args.raw_dir})
    with open(os.path.join(args.out_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != 'position_prior_by_decile'}, indent=2))


def cmd_ensemble(args):
    dirs = [d for d in args.fit_dirs.split(',') if d]
    per_dir = []
    for d in dirs:
        per_dir.append({lab.doc['doc_id']: lab for lab in
                        (load_chunk_labels(p) for p in record_paths(d))})
    common = sorted(set.intersection(*(set(m) for m in per_dir)))
    print(f"{len(common)} documents labeled by all {len(dirs)} readers")
    labels, agreement = [], defaultdict(list)
    for doc_id in common:
        group = [m[doc_id] for m in per_dir]
        lab = ensemble_labels(group)
        save_record(lab, args.out_dir)
        labels.append(lab)
        for (i, a), (j, b) in itertools.combinations(enumerate(group), 2):
            if a.informative and b.informative:
                agreement[f'{a.reader}~{b.reader}'].append(spearman(a.beta, b.beta))
    summary = _summary_stats(labels)
    summary['cross_reader_spearman'] = {k: float(np.nanmean(v)) for k, v in agreement.items()}
    summary['fit_dirs'] = dirs
    with open(os.path.join(args.out_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != 'position_prior_by_decile'}, indent=2))


def main():
    try:  # Vietnamese text / arrows on a non-UTF-8 console (Windows cp1252) must not crash a run
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)

    m = sub.add_parser('measure')
    add_document_args(m)
    m.add_argument('--reader-model', required=True)
    m.add_argument('--backend', choices=['hf', 'vllm'], default='hf')
    m.add_argument('--device', default='cuda')
    m.add_argument('--dtype', default='bfloat16')
    m.add_argument('--batch-size', type=int, default=8, help="hf backend batch size")
    m.add_argument('--max-model-len', type=int, default=None)
    m.add_argument('--tp', type=int, default=1, help="vLLM tensor parallel size (GPUs visible to this process)")
    m.add_argument('--gpu-memory-utilization', type=float, default=0.9)
    m.add_argument('--max-new-tokens', type=int, default=None, help="default: per hop type (32 single / 48 multi)")
    m.add_argument('--outcomes', default='f1,logprob', help="f1 is always measured; add logprob for the ablation")
    m.add_argument('--keep-rates', default='0.5,0.25', help="Bernoulli keep probabilities, cycled over masks")
    m.add_argument('--k-min', type=int, default=64)
    m.add_argument('--k-per-chunk', type=float, default=1.0)
    m.add_argument('--k-max', type=int, default=256)
    m.add_argument('--docs-per-call', type=int, default=16, help="documents whose masks go to the reader together")
    m.add_argument('--shard', type=int, default=0)
    m.add_argument('--num-shards', type=int, default=1)
    m.add_argument('--out-root', default='labels/raw')
    m.set_defaults(func=cmd_measure)

    f = sub.add_parser('fit')
    f.add_argument('--raw-dir', required=True)
    f.add_argument('--target', choices=['f1', 'logprob'], default='f1')
    f.add_argument('--alpha', type=float, default=None)
    f.add_argument('--alpha-from', default=None, help="dev measure dir for alpha CV")
    f.add_argument('--alpha-grid', default='0.1,0.3,1,3,10')
    f.add_argument('--n-boot', type=int, default=0, help="bootstrap CIs per chunk (slow; diagnostics only)")
    f.add_argument('--out-dir', required=True)
    f.set_defaults(func=cmd_fit)

    e = sub.add_parser('ensemble')
    e.add_argument('--fit-dirs', required=True, help="comma list of fit dirs (one per reader, same source/split)")
    e.add_argument('--out-dir', required=True)
    e.set_defaults(func=cmd_ensemble)

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
