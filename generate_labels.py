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
    MaskOutcomes, ensemble_labels, fit_document, fit_document_loo, fit_position_prior, load_chunk_labels,
    load_mask_outcomes, loo_masks, masks_for_document, position_r2, record_paths, save_record, select_alpha,
    subsample_masks,
)
from ttcompress.metrics import spearman, token_f1
from ttcompress.reader import MAX_NEW_TOKENS, load_reader, reader_tag
from ttcompress.reader import BASE_PROMPT_VERSION, PROMPT_VERSION, reader_is_chat
from ttcompress.sources import HOP, SOURCES, data_version, first_n_docs, load_documents


def add_document_args(p):
    p.add_argument('--source', choices=SOURCES, required=True)
    p.add_argument('--split', choices=['train', 'dev', 'test'], required=True)
    p.add_argument('--n', type=int, default=None, help="documents (hash-chosen, nested across n); default all")
    p.add_argument('--haystack-chars', type=int, default=30000, help="single-hop haystack length")
    p.add_argument('--distractors', choices=['random', 'hard'], default='random')
    p.add_argument('--multihop-pad-chars', type=int, default=0, help="lengthen multi-hop docs with easy distractors")
    p.add_argument('--multihop-units', choices=['paragraph', 'sentence'], default='paragraph',
                   help="multi-hop chunk unit: titled paragraphs (default) or sentences (sources.multihop_row_to_doc)")


def cmd_measure(args):
    docs = load_documents(args.source, args.split, args.n, args.haystack_chars, args.distractors,
                          args.multihop_pad_chars, args.multihop_units)
    docs = [d for i, d in enumerate(docs) if i % args.num_shards == args.shard]
    out_dir = os.path.join(args.out_root, reader_tag(args.reader_model), f'{args.source}_{args.split}')
    todo = [d for d in docs if not os.path.exists(os.path.join(out_dir, f'{d.doc_id}.json'))]
    print(f"{args.source}/{args.split} shard {args.shard}/{args.num_shards}: {len(docs)} docs, "
          f"{len(docs) - len(todo)} already measured, {len(todo)} to go -> {out_dir}")
    keep_rates = [float(x) for x in args.keep_rates.split(',')]
    outcomes = set(args.outcomes.split(','))
    # a resumed run must measure with exactly the settings the records on disk were measured with.
    # num_shards is deliberately NOT part of it: records are per document, masks are seeded by doc_id
    # (masks_for_document) and the document set is hash-chosen and nested across --n, so a label dir
    # may be extended with a different GPU count (pilot -> mid -> full share labels/raw). Guarded by
    # tests/test_attribution.py::test_masks_do_not_depend_on_sharding.
    # everything that changes a record: document set (data_version), masks, prompt/answer handling, backend
    # only the document settings that act on this source (load_documents ignores the others), so a distractor
    # ablation run reads the unchanged sources' labels from the main run instead of refusing them
    single = HOP[args.source] == 'single'
    config = {'keep_rates': keep_rates, 'k_min': args.k_min, 'k_per_chunk': args.k_per_chunk, 'k_max': args.k_max,
              'outcomes': sorted(outcomes), 'haystack_chars': args.haystack_chars if single else None,
              'distractors': args.distractors if single else None,
              'multihop_pad_chars': None if single else args.multihop_pad_chars,
              # from the source, not docs[0]: a shard can be empty (n < num_shards)
              'max_new_tokens': args.max_new_tokens or MAX_NEW_TOKENS[HOP[args.source]],
              'data_version': data_version(args.source), 'prompt_version': PROMPT_VERSION,
              'backend': args.backend, 'dtype': args.dtype, 'max_model_len': args.max_model_len}
    if not reader_is_chat(args.reader_model):  # only base readers: the chat readers' label dirs keep their config
        config['base_prompt_version'] = BASE_PROMPT_VERSION
    if not single and args.multihop_units != 'paragraph':
        # only when it differs from the default, so label dirs measured before the option still match
        config['multihop_units'] = args.multihop_units
    if args.mask_scheme != 'random':
        # same pattern: random-mask label dirs measured before the option keep matching their config. A LOO
        # dir is a different label set (its masks are not the method's), so it must never share a dir with them
        # -- and `fit` reads this key to refuse fitting one with the other's estimator
        config['mask_scheme'] = args.mask_scheme
    os.makedirs(out_dir, exist_ok=True)
    config_path = os.path.join(out_dir, 'measure_config.json')
    # checked even when every document is on disk: a finished dir from other settings must not be fitted
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
    if not todo:
        return
    # how these records were produced, for provenance only (never compared: num_shards would stop a label dir
    # from growing on another GPU count, and vLLM batching is not bit-deterministic anyway)
    import importlib.metadata as im
    versions = {}
    for pkg in ('vllm', 'torch', 'transformers'):
        try:
            versions[pkg] = im.version(pkg)
        except im.PackageNotFoundError:
            pass
    with open(os.path.join(out_dir, 'measure_runs.jsonl'), 'a', encoding='utf-8') as f:
        f.write(json.dumps({'time': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'shard': args.shard,
                            'num_shards': args.num_shards, 'n_todo': len(todo), 'tp': args.tp,
                            'docs_per_call': args.docs_per_call, 'batch_size': args.batch_size,
                            'gpu_memory_utilization': args.gpu_memory_utilization, **versions}) + '\n')
    reader = load_reader(args.reader_model, args.backend, args.device, args.dtype, args.batch_size, args.max_model_len,
                         args.tp, args.gpu_memory_utilization)

    total_calls, t0 = 0, time.time()
    for g in range(0, len(todo), args.docs_per_call):
        group = todo[g:g + args.docs_per_call]
        prompts, owners = [], []
        per_doc_masks = []
        for di, d in enumerate(group):
            if args.mask_scheme == 'loo':
                masks = loo_masks(d.num_chunks)
            else:
                masks = masks_for_document(d.doc_id, d.num_chunks, keep_rates, args.k_min, args.k_per_chunk,
                                           args.k_max)
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
        elapsed = time.time() - start          # the F1 labels' cost (the RQ1 cost table reads it)
        logprobs, lp_elapsed = None, 0.0
        if 'logprob' in outcomes:               # ablation only: timed apart so it never inflates `seconds`
            t_lp = time.time()
            logprobs = reader.answer_logprob(prompts, [group[o].answers[0] for o in owners])
            lp_elapsed = time.time() - t_lp
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
                seconds_logprob=lp_elapsed * n / len(prompts),
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


def _first_n_records(raw_dir: str, n):
    """The measure records of the first n documents in load_documents order. A raw dir is shared by runs
    with different N (pilot -> mid -> full) and holds every document any of them measured; fitting it
    whole would silently train a small-N run on the large-N documents."""
    records = [load_mask_outcomes(p) for p in record_paths(raw_dir)]
    if n is None:
        return records
    if len(records) < n:
        raise SystemExit(f"{raw_dir} has {len(records)} measured documents, --n asks for {n}: finish the labels stage")
    keep = {d['doc_id'] for d in first_n_docs([r.doc for r in records], n)}
    return [r for r in records if r.doc['doc_id'] in keep]


def _mask_scheme(raw_dir: str) -> str:
    """How a raw dir's masks were drawn (measure --mask-scheme). A dir without measure_config.json, or one
    measured before the option existed, holds random masks."""
    path = os.path.join(raw_dir, 'measure_config.json')
    if not os.path.exists(path):
        return 'random'
    with open(path, encoding='utf-8') as f:
        return json.load(f).get('mask_scheme', 'random')


def _fit_loo(args, records):
    """estimator=loo: no surrogate, so nothing to choose on dev and nothing to bootstrap."""
    if args.alpha is not None or args.alpha_from:
        raise SystemExit("--estimator loo has no alpha: drop --alpha/--alpha-from")
    if args.max_masks is not None:
        raise SystemExit("--max-masks subsamples random masks for ridge; LOO needs every leave-one-out mask")
    if args.n_boot:
        raise SystemExit("--n-boot is a ridge diagnostic; LOO has no resampling CI")
    labels = []
    for rec in records:
        lab = fit_document_loo(rec, args.target)
        save_record(lab, args.out_dir)
        labels.append(lab)
    import warnings
    with warnings.catch_warnings():  # cv_r2 is nan by construction: the nanmeans of it warn on every LOO fit
        warnings.simplefilter('ignore', RuntimeWarning)
        summary = _summary_stats(labels, {'alpha': None})
    return summary


def cmd_fit(args):
    # the estimator must match how the raw dir was measured: ridge on LOO masks is a near-singular fit of C
    # rows, and LOO on random masks has no "full minus one" contrast to read off
    scheme = _mask_scheme(args.raw_dir)
    if args.estimator == 'loo' and scheme != 'loo':
        raise SystemExit(f"{args.raw_dir} was measured with mask_scheme={scheme}; --estimator loo needs a dir "
                         f"measured with --mask-scheme loo")
    if args.estimator == 'ridge' and scheme != 'random':
        raise SystemExit(f"{args.raw_dir} was measured with mask_scheme={scheme}; --estimator ridge needs random "
                         f"masks (fit it with --estimator {scheme})")
    if args.estimator == 'ridge' and args.max_masks is not None and args.max_masks <= 0:
        raise SystemExit(f"--max-masks must be > 0; got {args.max_masks}")
    records = _first_n_records(args.raw_dir, args.n)
    if not records:
        raise SystemExit(f"no measure output in {args.raw_dir}")
    os.makedirs(args.out_dir, exist_ok=True)
    for stale in record_paths(args.out_dir):  # a re-fit with a smaller --n must not leave extra documents behind
        os.remove(stale)
    if args.estimator == 'loo':
        summary = _fit_loo(args, records)
        summary.update({'target': args.target, 'raw_dir': args.raw_dir, 'n_requested': args.n, 'estimator': 'loo'})
        with open(os.path.join(args.out_dir, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(json.dumps({k: v for k, v in summary.items() if k != 'position_prior_by_decile'}, indent=2))
        return

    def cut(recs):
        # equal-cost baseline: every record (train AND the dev records alpha is chosen on) cut to the same K,
        # so alpha is selected for the regime it is used in. Seeded per (mask_seed, doc_id): a document's
        # subset does not depend on --n, sharding or which other documents are fitted
        if args.max_masks is None:
            return recs
        return [subsample_masks(r, args.max_masks, args.mask_seed) for r in recs]

    records = cut(records)
    alpha_info = {}
    if args.alpha is not None:
        alpha = args.alpha
    else:
        if not args.alpha_from:
            raise SystemExit("pass --alpha, or --alpha-from <dev measure dir> (alpha is chosen on dev, never on train)")
        # comma list: an eval-only source (xquad_vi, 2wiki) has no dev labels of its own -> pooled train-source dev
        dev_dirs = [d for d in args.alpha_from.split(',') if d]
        for d in dev_dirs:
            if _mask_scheme(d) != 'random':
                raise SystemExit(f"--alpha-from {d} was measured with mask_scheme={_mask_scheme(d)}; "
                                 f"ridge alpha CV needs random masks")
        dev = cut([r for d in dev_dirs for r in _first_n_records(d, args.alpha_n)])
        if not dev:
            raise SystemExit(f"no measure output in {args.alpha_from}")
        grid = [float(a) for a in args.alpha_grid.split(',')]
        alpha, scores = select_alpha([(r.masks, r.f1 if args.target == 'f1' else r.logprob) for r in dev], grid)
        alpha_info = {'alpha': alpha, 'alpha_cv_mse': scores, 'alpha_dev_dir': args.alpha_from, 'n_alpha_docs': len(dev)}
        print(f"alpha={alpha} by dev CV over {len(dev)} docs: {scores}")
    labels = []
    for rec in records:
        lab = fit_document(rec, args.target, alpha, args.n_boot)
        save_record(lab, args.out_dir)
        labels.append(lab)
    summary = _summary_stats(labels, alpha_info or {'alpha': alpha})
    summary.update({'target': args.target, 'raw_dir': args.raw_dir, 'n_requested': args.n})
    if args.max_masks is not None:  # only off the default path, so the main fit dirs' summaries keep their keys
        summary.update({'estimator': 'ridge', 'max_masks': args.max_masks, 'mask_seed': args.mask_seed,
                        'mean_masks_used': float(np.mean([len(r.masks) for r in records]))})
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
    # a partially recovered shard shrinks the ensemble silently otherwise
    dropped = {d: len(m) - len(common) for d, m in zip(dirs, per_dir)}
    for d, m in zip(dirs, per_dir):
        if dropped[d]:
            print(f"[WARN] {d}: {dropped[d]} of {len(m)} documents dropped (not labeled by every reader)")
    os.makedirs(args.out_dir, exist_ok=True)
    for stale in record_paths(args.out_dir):
        os.remove(stale)
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
    summary['n_dropped_per_dir'] = dropped
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
    m.add_argument('--max-new-tokens', type=int, default=None, help="default: per hop type (reader.MAX_NEW_TOKENS)")
    m.add_argument('--outcomes', default='f1', help="f1 is always measured; f1,logprob for the log-prob ablation "
                                                     "(run_pipeline.sh: primary reader, train/dev only)")
    m.add_argument('--keep-rates', default='0.5,0.25', help="Bernoulli keep probabilities, cycled over masks")
    m.add_argument('--k-min', type=int, default=64)
    m.add_argument('--k-per-chunk', type=float, default=1.0)
    m.add_argument('--k-max', type=int, default=256)
    m.add_argument('--mask-scheme', choices=['random', 'loo'], default='random',
                   help="random: Bernoulli masks (the method; --keep-rates/--k-*); loo: the C leave-one-out masks "
                        "(LOO baseline, fit with --estimator loo; --keep-rates/--k-* unused)")
    m.add_argument('--docs-per-call', type=int, default=16, help="documents whose masks go to the reader together")
    m.add_argument('--shard', type=int, default=0)
    m.add_argument('--num-shards', type=int, default=1)
    m.add_argument('--out-root', default='labels/raw')
    m.set_defaults(func=cmd_measure)

    f = sub.add_parser('fit')
    f.add_argument('--raw-dir', required=True)
    f.add_argument('--target', choices=['f1', 'logprob'], default='f1')
    f.add_argument('--alpha', type=float, default=None)
    f.add_argument('--alpha-from', default=None, help="dev measure dir(s) for alpha CV (comma list = pooled)")
    f.add_argument('--alpha-grid', default='0.1,0.3,1,3,10')
    f.add_argument('--n', type=int, default=None,
                   help="fit only the first n documents (load_documents order) of --raw-dir; default all")
    f.add_argument('--alpha-n', type=int, default=None, help="same, for each --alpha-from dir")
    f.add_argument('--n-boot', type=int, default=0, help="bootstrap CIs per chunk (slow; diagnostics only)")
    f.add_argument('--estimator', choices=['ridge', 'loo'], default='ridge',
                   help="ridge: surrogate on random masks (the method); loo: full-minus-one deltas on a "
                        "--mask-scheme loo raw dir (baseline; no alpha)")
    f.add_argument('--max-masks', type=int, default=None,
                   help="ridge only: fit each document (and the --alpha-from dev documents) on a random subset of "
                        "at most N of its masks (equal-cost baseline); default all")
    f.add_argument('--mask-seed', type=int, default=0, help="seed of the --max-masks subset (per document)")
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
