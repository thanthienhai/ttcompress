#!/usr/bin/env python3
"""Selection latency, measured on its own (RQ1 cost claim).

The `select_ms` of evaluate.py's report is the wall-clock of the select stage, which runs four shards side by
side with other arms loading and unloading: in the 2026-09-29 full run ours_beta took 36-76 ms per document
while its seed reruns -- the same architecture, the same documents -- took 12-50 ms, at every document, not
only the first. This script times the arms one at a time on one GPU, after a warm-up, with CUDA synchronized
around each document, and repeats the arms in reversed order so a drift or a neighbour shows up as a
disagreement between the two passes. It times what `select` times (scoring + selection at one ratio).

    python scripts/bench_latency.py --eval-dir runs/main/results/eval_test --budget-tokenizer Qwen/Qwen3-8B \\
        --arms ours_beta=pruner:runs/main/models/pruner_beta_primary,reranker=reranker:BAAI/bge-reranker-v2-m3,exit \\
        --n-per-source 100 --out runs/main/results/eval_test/bench_latency.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluate import _arm_rows, _read_jsonl, parse_arms  # noqa: E402
from ttcompress.data import QADocument  # noqa: E402
from ttcompress.selection import make_arm  # noqa: E402


def _sync():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def load_docs(eval_dir, n_per_source):
    docs = []
    for path in sorted(glob.glob(os.path.join(eval_dir, 'documents_shard*.jsonl'))):
        docs.extend(QADocument.from_dict(d) for d in _read_jsonl(path))
    by_source = defaultdict(list)
    for d in sorted(docs, key=lambda d: d.doc_id):
        by_source[d.source].append(d)
    return {s: ds[:n_per_source] for s, ds in sorted(by_source.items())}


def time_arm(label, spec, docs_by_source, ratio, tok, count, device, warmup):
    arm = make_arm(spec, device)
    every = [d for ds in docs_by_source.values() for d in ds]
    lengths = {d.doc_id: [count(c) for c in d.chunks] for d in every}
    full = {d.doc_id: count(d.text()) for d in every}
    for d in every[:warmup]:
        _arm_rows(arm, d, label, [ratio], full[d.doc_id], lengths[d.doc_id], tok, count)
    out = defaultdict(list)
    for source, ds in docs_by_source.items():
        for d in ds:
            _sync()
            t0 = time.perf_counter()
            _arm_rows(arm, d, label, [ratio], full[d.doc_id], lengths[d.doc_id], tok, count)
            _sync()
            out[source].append(1000 * (time.perf_counter() - t0))
    del arm
    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--eval-dir', required=True, help="a select out-dir (reads documents_shard*.jsonl)")
    ap.add_argument('--arms', required=True, help="comma list, as evaluate.py select --arms")
    ap.add_argument('--budget-tokenizer', required=True)
    ap.add_argument('--n-per-source', type=int, default=100)
    ap.add_argument('--ratio', type=float, default=4.0)
    ap.add_argument('--warmup', type=int, default=5)
    ap.add_argument('--repeats', type=int, default=2, help="passes over the arms, every other one in reverse order")
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', required=True, help="JSON output; a .md table is written next to it")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.budget_tokenizer, trust_remote_code=True)
    count = lambda text: len(tok.encode(text, add_special_tokens=False))  # noqa: E731
    docs = load_docs(args.eval_dir, args.n_per_source)
    arms = parse_arms(args.arms)
    env = {'n_per_source': {s: len(ds) for s, ds in docs.items()}, 'ratio': args.ratio, 'warmup': args.warmup,
           'repeats': args.repeats}
    try:
        import torch
        env['torch'] = torch.__version__
        if args.device != 'cpu' and torch.cuda.is_available():
            env['gpu'] = torch.cuda.get_device_name(0)
            try:   # other processes on the GPU would make the numbers meaningless: recorded, not enforced
                env['gpu_processes'] = torch.cuda.list_gpu_processes(0)
            except Exception as exc:  # noqa: BLE001 -- needs pynvml and permissions; optional
                env['gpu_processes'] = f'unavailable ({type(exc).__name__})'
    except ImportError:
        pass
    times = defaultdict(lambda: defaultdict(list))   # arm -> source -> ms per document, over every pass
    per_pass = defaultdict(dict)                     # (arm, source) -> {pass: median}
    for rep in range(args.repeats):
        order = arms if rep % 2 == 0 else list(reversed(arms))
        for label, spec in order:
            print(f"pass {rep + 1}/{args.repeats}: {label}", flush=True)
            for source, ms in time_arm(label, spec, docs, args.ratio, tok, count, args.device, args.warmup).items():
                times[label][source].extend(ms)
                per_pass[(label, source)][rep] = float(np.median(ms))
    rows = []
    for label, _ in arms:
        for source, ms in sorted(times[label].items()):
            passes = per_pass[(label, source)]
            rows.append({'arm': label, 'source': source, 'n': len(ms), 'median_ms': float(np.median(ms)),
                         'p10_ms': float(np.percentile(ms, 10)), 'p90_ms': float(np.percentile(ms, 90)),
                         'mean_ms': float(np.mean(ms)), 'pass_medians_ms': [passes[k] for k in sorted(passes)]})
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump({'env': env, 'rows': rows}, f, ensure_ascii=False, indent=2)
    lines = ['# Selection latency (one arm at a time, one GPU, CUDA synchronized, after warm-up)', '',
             f"GPU: {env.get('gpu', 'cpu')}; ratio {args.ratio}; {args.n_per_source} documents per source; "
             f"{args.repeats} passes (alternating order).", '',
             '| arm | source | n | median ms | p10-p90 ms | mean ms | median per pass |', '|---|---|---|---|---|---|---|']
    for r in rows:
        lines.append(f"| {r['arm']} | {r['source']} | {r['n']} | {r['median_ms']:.1f} | {r['p10_ms']:.1f}-{r['p90_ms']:.1f} | "
                     f"{r['mean_ms']:.1f} | {', '.join(f'{v:.1f}' for v in r['pass_medians_ms'])} |")
    md = '\n'.join(lines) + '\n'
    with open(os.path.splitext(args.out)[0] + '.md', 'w', encoding='utf-8') as f:
        f.write(md)
    print(md)


if __name__ == '__main__':
    main()
