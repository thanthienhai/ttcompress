#!/usr/bin/env python3
"""Evaluation at fixed token budgets (METHOD_SPEC.md §5-6).

Three subcommands, so the expensive parts are never recomputed:

  select   compress every document with every arm at every ratio (no reader).
           Budgets are counted with ONE reference tokenizer (--budget-tokenizer),
           so every reader later sees the identical compressed context -- the
           "fixed compressor" setting in which reader upgrades are measured.
  answer   one reader answers every selection; one invocation per reader
           (and per GPU shard).
  report   CPU-only: per reader x source x ratio x arm means with cluster
           CIs, PAIRED bootstrap differences of our arms vs every other arm,
           upgrade retention for every reader pair, and needle-depth slices.
           Results are never pooled across sources or languages.

    python evaluate.py select --sources uit_viquad,xquad_vi,vimqa,hotpotqa --split test --n 500 \\
        --arms full,lead,random,bm25,embed,oracle_span,oracle_support,beta=pruner:models/pruner_beta,span=pruner:models/pruner_span \\
        --budget-tokenizer Qwen/Qwen3-8B --out-dir results/eval_test --shard 0 --num-shards 4
    python evaluate.py answer --out-dir results/eval_test --reader-model Qwen/Qwen3-8B --backend vllm --shard 0 --num-shards 4
    python evaluate.py report --out-dir results/eval_test --ours beta
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import re
import sys
import time
from collections import defaultdict

import numpy as np

from ttcompress.data import QADocument
from ttcompress.metrics import (
    answer_recall, bootstrap_mean_ci, exact_match, gold_chunk_recall, paired_bootstrap_diff, token_f1,
    upgrade_retention,
)
from ttcompress.reader import MAX_NEW_TOKENS, load_reader, reader_tag
from ttcompress.selection import Selection, budget_for, make_arm, select_by_scores
from ttcompress.sources import load_documents, parse_source_list

_PREFIXES = ('pruner:', 'provence:', 'embed:', 'llmlingua2:')


def parse_arms(spec: str):
    """'lead,beta=pruner:models/x' -> [('lead','lead'), ('beta','pruner:models/x')]."""
    out = []
    for item in [s.strip() for s in spec.split(',') if s.strip()]:
        if '=' in item and not item.startswith(_PREFIXES):
            label, arm = item.split('=', 1)
        else:
            label, arm = item, item
        out.append((label, arm))
    return out


def _read_jsonl(path):
    """Rows of a jsonl file; a torn LAST line (process killed mid-append) is
    dropped with a warning -- its rows are simply recomputed on resume."""
    if not os.path.exists(path):
        return []
    with open(path, encoding='utf-8') as f:
        lines = [line for line in f if line.strip()]
    rows = []
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
            print(f"[WARN] {path}: dropping a truncated last line (interrupted write)")
    return rows


def _write_json_atomic(path, obj):
    tmp = f'{path}.{os.getpid()}.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _shard_index(path):
    return int(re.search(r'_shard(\d+)\.jsonl$', path).group(1))


def _append_jsonl(path, rows):
    with open(path, 'a', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


# ---------------------------------------------------------------------------
# select
# ---------------------------------------------------------------------------

def cmd_select(args):
    from transformers import AutoTokenizer

    os.makedirs(args.out_dir, exist_ok=True)
    docs = []
    for src in parse_source_list(args.sources):
        docs.extend(load_documents(src, args.split, args.n, args.haystack_chars, args.distractors,
                                   args.multihop_pad_chars))
    ratios = [float(r) for r in args.ratios.split(',')]
    # the document set and its shard assignment must be identical across resumed runs,
    # otherwise documents would be duplicated or dropped between shard files
    meta = {'split': args.split, 'budget_tokenizer': args.budget_tokenizer, 'sources': args.sources, 'n': args.n,
            'num_shards': args.num_shards, 'haystack_chars': args.haystack_chars, 'distractors': args.distractors,
            'multihop_pad_chars': args.multihop_pad_chars}
    config_path = os.path.join(args.out_dir, 'select_config.json')
    if os.path.exists(config_path):
        with open(config_path, encoding='utf-8') as f:
            on_disk = json.load(f)
        if on_disk != meta:
            raise SystemExit(f"{config_path} has {on_disk}, this run asks for {meta}; use a new --out-dir "
                             f"(arms/ratios may change between runs, the document set may not)")
    else:
        _write_json_atomic(config_path, meta)
    docs = [d for i, d in enumerate(docs) if i % args.num_shards == args.shard]
    doc_path = os.path.join(args.out_dir, f'documents_shard{args.shard}.jsonl')
    tmp = f'{doc_path}.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        f.writelines(json.dumps(d.to_dict(), ensure_ascii=False) + '\n' for d in docs)
    os.replace(tmp, doc_path)
    sel_path = os.path.join(args.out_dir, f'selections_shard{args.shard}.jsonl')
    done = {(r['doc_id'], r['arm'], r['ratio']) for r in _read_jsonl(sel_path)}

    tok = AutoTokenizer.from_pretrained(args.budget_tokenizer, trust_remote_code=True)
    count = lambda text: len(tok.encode(text, add_special_tokens=False))  # noqa: E731
    lengths = {d.doc_id: [count(c) for c in d.chunks] for d in docs}
    full_tokens = {d.doc_id: count(d.text()) for d in docs}
    print(f"{len(docs)} documents in shard {args.shard}/{args.num_shards} ({args.split}); {len(done)} selections on disk")

    for label, spec in parse_arms(args.arms):
        wanted = [('full',)] if spec == 'full' else [(r,) for r in ratios]
        pending = [d for d in docs if any((d.doc_id, label, w[0]) not in done for w in wanted)]
        if not pending:
            continue
        arm = make_arm(spec, args.device, args.oracle_beta_dir)
        rows, t_arm = [], time.time()
        for d in pending:
            base = {'doc_id': d.doc_id, 'arm': label, 'source': d.source, 'language': d.language, 'hop': d.hop,
                    'cluster_id': d.cluster_id, 'needle_relpos': d.metadata.get('needle_relpos'),
                    'num_chunks': d.num_chunks, 'full_tokens': full_tokens[d.doc_id]}
            if arm.kind == 'full':
                rows.append({**base, 'ratio': 'full', 'kept': list(range(d.num_chunks)), 'text': d.text(),
                             'kept_tokens': full_tokens[d.doc_id], 'budget': full_tokens[d.doc_id],
                             'gold_recall': 1.0, 'seconds': 0.0})
                continue
            if arm.kind == 'text':
                for r in ratios:
                    t0 = time.time()
                    text = arm.compress_text(d, r)
                    rows.append({**base, 'ratio': r, 'kept': [], 'text': text, 'kept_tokens': count(text),
                                 'budget': budget_for(full_tokens[d.doc_id], r), 'gold_recall': None,
                                 'seconds': time.time() - t0})
                continue
            t0 = time.time()
            scores = arm.scores(d)
            seconds = time.time() - t0
            if scores is None:  # e.g. oracle_beta without labels for this document
                continue
            for r in ratios:
                sel: Selection = select_by_scores(d, scores, lengths[d.doc_id], budget_for(full_tokens[d.doc_id], r), tok)
                # a chunk cut to the budget is not counted as kept evidence
                rows.append({**base, 'ratio': r, 'kept': sel.kept, 'text': sel.text, 'kept_tokens': sel.kept_tokens,
                             'budget': sel.budget, 'truncated': sel.truncated,
                             'gold_recall': 0.0 if sel.truncated else gold_chunk_recall(sel.kept, d.gold_chunks),
                             'seconds': seconds})
        _append_jsonl(sel_path, rows)
        print(f"  {label}: {len(rows)} selections in {time.time() - t_arm:.1f}s", flush=True)
        del arm
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# answer
# ---------------------------------------------------------------------------

def cmd_answer(args):
    """Answers every selection file whose shard index k satisfies
    k % num_shards == shard, so the answer stage can use a different number
    of processes than select did (e.g. 2 processes x 2 GPUs for a 32B reader)."""
    files = sorted(glob.glob(os.path.join(args.out_dir, 'selections_shard*.jsonl')), key=_shard_index)
    files = [p for p in files if _shard_index(p) % args.num_shards == args.shard]
    if not files:
        print(f"shard {args.shard}/{args.num_shards}: no selection files to answer")
        return
    tag = reader_tag(args.reader_model)
    work = []  # (k, docs, todo, out_path)
    for path in files:
        k = _shard_index(path)
        docs = {d['doc_id']: QADocument.from_dict(d)
                for d in _read_jsonl(os.path.join(args.out_dir, f'documents_shard{k}.jsonl'))}
        selections = _read_jsonl(path)
        out_path = os.path.join(args.out_dir, f'answers_{tag}_shard{k}.jsonl')
        done = {(r['doc_id'], r['arm'], r['ratio']) for r in _read_jsonl(out_path)}
        todo = [s for s in selections if (s['doc_id'], s['arm'], s['ratio']) not in done]
        missing = {s['doc_id'] for s in todo} - set(docs)
        if missing:
            raise SystemExit(f"{path}: {len(missing)} selections refer to documents missing from "
                             f"documents_shard{k}.jsonl -- rerun `select` with the original settings")
        print(f"{tag} selection shard {k}: {len(todo)} of {len(selections)} selections to answer")
        if todo:
            work.append((k, docs, todo, out_path))
    if not work:
        return
    reader = load_reader(args.reader_model, args.backend, args.device, args.dtype, args.batch_size, args.max_model_len,
                         args.tp, args.gpu_memory_utilization)
    for k, docs, todo, out_path in work:
        for start in range(0, len(todo), args.chunk):
            block = todo[start:start + args.chunk]
            answers = [''] * len(block)
            for hop in {s['hop'] for s in block}:
                idx = [i for i, s in enumerate(block) if s['hop'] == hop]
                prompts = [reader.build_prompt(block[i]['text'], docs[block[i]['doc_id']].question,
                                               docs[block[i]['doc_id']].language) for i in idx]
                for i, a in zip(idx, reader.generate(prompts, MAX_NEW_TOKENS[hop])):
                    answers[i] = a
            rows = []
            for s, a in zip(block, answers):
                gold = docs[s['doc_id']].answers
                rows.append({key: v for key, v in s.items() if key not in ('text', 'kept')} | {
                    'reader': tag, 'answer': a, 'em': exact_match(a, gold), 'f1': token_f1(a, gold),
                    'answer_recall': answer_recall(a, gold)})
            _append_jsonl(out_path, rows)
            print(f"  shard {k}: {min(start + args.chunk, len(todo))}/{len(todo)} "
                  f"(truncated prompts so far: {reader.n_truncated})", flush=True)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

METRICS = ('f1', 'em', 'answer_recall', 'gold_recall')


def _ratio_key(r):
    return str(r)


def cmd_report(args):
    rows = []
    for p in glob.glob(os.path.join(args.out_dir, 'answers_*_shard*.jsonl')):
        rows.extend(_read_jsonl(p))
    if not rows:
        raise SystemExit(f"no answers_* files in {args.out_dir}")
    readers = sorted({r['reader'] for r in rows})
    arms = sorted({r['arm'] for r in rows})
    ours = [a for a in (args.ours.split(',') if args.ours else []) if a]

    # cell[(reader, source, ratio, arm)][doc_id] = row
    cell = defaultdict(dict)
    for r in rows:
        cell[(r['reader'], r['source'], _ratio_key(r['ratio']), r['arm'])][r['doc_id']] = r
    report = {'readers': readers, 'arms': arms, 'ours': ours, 'cells': [], 'paired': [], 'upgrade_retention': [],
              'by_depth': []}

    for (reader, source, ratio, arm), by_doc in sorted(cell.items()):
        docs = sorted(by_doc)
        clusters = [by_doc[d]['cluster_id'] for d in docs]
        entry = {'reader': reader, 'source': source, 'ratio': ratio, 'arm': arm, 'n': len(docs),
                 'n_clusters': len(set(clusters)),
                 'compression': float(np.mean([by_doc[d]['full_tokens'] / max(1, by_doc[d]['kept_tokens']) for d in docs])),
                 'select_ms': 1000 * float(np.mean([by_doc[d]['seconds'] for d in docs])),
                 'truncated_rate': float(np.mean([bool(by_doc[d].get('truncated')) for d in docs]))}
        for m in METRICS:
            vals = [np.nan if by_doc[d].get(m) is None else by_doc[d][m] for d in docs]
            mean, lo, hi = bootstrap_mean_ci(vals, clusters, n_boot=args.n_boot)
            entry[m] = {'mean': mean, 'ci95': [lo, hi]}
        report['cells'].append(entry)

    # paired differences: each of our arms vs every other arm, same reader/source/ratio, same documents
    for (reader, source, ratio, arm), by_doc in sorted(cell.items()):
        if arm not in ours:
            continue
        for other in arms:
            if other == arm:
                continue
            other_cell = cell.get((reader, source, ratio, other)) or cell.get((reader, source, 'full', other))
            if not other_cell:
                continue
            common = sorted(set(by_doc) & set(other_cell))
            if len(common) < 2:
                continue
            clusters = [by_doc[d]['cluster_id'] for d in common]
            res = paired_bootstrap_diff([by_doc[d]['f1'] for d in common], [other_cell[d]['f1'] for d in common],
                                        clusters, n_boot=args.n_boot)
            report['paired'].append({'reader': reader, 'source': source, 'ratio': ratio, 'ours': arm, 'vs': other,
                                     'metric': 'f1', **res})

    # upgrade retention for every reader pair ordered by full-context F1 on that source
    sources = sorted({r['source'] for r in rows})
    ratios = sorted({_ratio_key(r['ratio']) for r in rows if r['ratio'] != 'full'})
    for source in sources:
        full_mean = {}
        for reader in readers:
            c = cell.get((reader, source, 'full', 'full'))
            if c:
                full_mean[reader] = float(np.mean([v['f1'] for v in c.values()]))
        for weak, strong in itertools.permutations(full_mean, 2):
            if full_mean[strong] <= full_mean[weak]:
                continue
            for ratio in ratios:
                for arm in arms:
                    cw, cs = cell.get((weak, source, ratio, arm)), cell.get((strong, source, ratio, arm))
                    fw, fs = cell.get((weak, source, 'full', 'full')), cell.get((strong, source, 'full', 'full'))
                    if not (cw and cs and fw and fs):
                        continue
                    common = sorted(set(cw) & set(cs) & set(fw) & set(fs))
                    if len(common) < 2:
                        continue
                    arr = np.array([[cw[d]['f1'], cs[d]['f1'], fw[d]['f1'], fs[d]['f1']] for d in common])
                    point = upgrade_retention(*arr.mean(axis=0))
                    lo, hi = _retention_ci(arr, [cw[d]['cluster_id'] for d in common], args.n_boot)
                    report['upgrade_retention'].append({'source': source, 'weak': weak, 'strong': strong,
                                                        'ratio': ratio, 'arm': arm, 'retention': point,
                                                        'ci95': [lo, hi], 'n': len(common)})

    # single-hop: gold recall and F1 by needle depth (quintiles of relative position)
    for (reader, source, ratio, arm), by_doc in sorted(cell.items()):
        depth = [(v['needle_relpos'], v) for v in by_doc.values() if v.get('needle_relpos') is not None]
        if not depth or ratio == 'full':
            continue
        bins = defaultdict(list)
        for pos, v in depth:
            bins[min(4, int(pos * 5))].append(v)
        report['by_depth'].append({'reader': reader, 'source': source, 'ratio': ratio, 'arm': arm, 'bins': {
            f'q{b + 1}': {'n': len(vs), 'f1': float(np.mean([x['f1'] for x in vs])),
                          'gold_recall': float(np.nanmean([np.nan if x['gold_recall'] is None else x['gold_recall']
                                                           for x in vs]))}
            for b, vs in sorted(bins.items())}})

    with open(os.path.join(args.out_dir, 'report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    md = _markdown(report)
    with open(os.path.join(args.out_dir, 'report.md'), 'w', encoding='utf-8') as f:
        f.write(md)
    print(md)


def _retention_ci(arr: np.ndarray, clusters, n_boot: int, seed: int = 42):
    groups = defaultdict(list)
    for i, c in enumerate(clusters):
        groups[c].append(i)
    members = list(groups.values())
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([members[j] for j in rng.integers(0, len(members), size=len(members))])
        vals.append(upgrade_retention(*arr[idx].mean(axis=0)))
    vals = np.asarray(vals)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return None, None
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def _fmt(m):
    lo, hi = m['ci95']
    return f"{m['mean']:.3f}" + (f" [{lo:.3f}, {hi:.3f}]" if lo is not None else '')


def _markdown(report) -> str:
    lines = ['# Evaluation report', '', 'Token F1 with 95% cluster-bootstrap CI; never pooled across sources.', '']
    groups = defaultdict(list)
    for c in report['cells']:
        groups[(c['reader'], c['source'])].append(c)
    for (reader, source), cells in sorted(groups.items()):
        lines += [f'## {source} — reader `{reader}`', '',
                  '| ratio | arm | F1 | EM | gold recall | realized compression | truncated | select ms |',
                  '|---|---|---|---|---|---|---|---|']
        for c in sorted(cells, key=lambda c: (c['ratio'] != 'full', c['ratio'], -c['f1']['mean'])):
            gr = c['gold_recall']['mean']
            gr_text = '' if gr != gr else f"{gr:.3f}"
            lines.append(f"| {c['ratio']} | {c['arm']} | {_fmt(c['f1'])} | {c['em']['mean']:.3f} | "
                         f"{gr_text} | {c['compression']:.1f}x | {c['truncated_rate']:.0%} | {c['select_ms']:.0f} |")
        lines.append('')
    if report['paired']:
        lines += ['## Paired differences (ours − other, token F1)', '',
                  '| reader | source | ratio | ours | vs | Δ F1 [95% CI] | p |', '|---|---|---|---|---|---|---|']
        for p in report['paired']:
            lo, hi = p['ci95']
            ci = f" [{lo:+.3f}, {hi:+.3f}]" if lo is not None else ''
            pval = '' if p['p'] is None else f"{p['p']:.3f}"
            lines.append(f"| {p['reader']} | {p['source']} | {p['ratio']} | {p['ours']} | {p['vs']} | "
                         f"{p['diff']:+.3f}{ci} | {pval} |")
        lines.append('')
    if report['upgrade_retention']:
        lines += ['## Upgrade retention (share of the full-context weak→strong gain kept)', '',
                  '| source | weak → strong | ratio | arm | retention [95% CI] |', '|---|---|---|---|---|']
        for u in report['upgrade_retention']:
            lo, hi = u['ci95']
            ci = f" [{lo:.2f}, {hi:.2f}]" if lo is not None else ''
            lines.append(f"| {u['source']} | {u['weak']} → {u['strong']} | {u['ratio']} | {u['arm']} | "
                         f"{u['retention']:.2f}{ci} |")
        lines.append('')
    return '\n'.join(lines)


def main():
    try:  # Vietnamese text / arrows on a non-UTF-8 console (Windows cp1252) must not crash a run
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)

    s = sub.add_parser('select')
    s.add_argument('--sources', required=True, help="comma list or 'all'")
    s.add_argument('--split', choices=['dev', 'test'], default='test')
    s.add_argument('--n', type=int, default=None, help="documents per source")
    s.add_argument('--haystack-chars', type=int, default=30000)
    s.add_argument('--distractors', choices=['random', 'hard'], default='random')
    s.add_argument('--multihop-pad-chars', type=int, default=0)
    s.add_argument('--arms', required=True, help="comma list; 'label=spec' to name an arm, e.g. beta=pruner:models/x")
    s.add_argument('--ratios', default='4,8')
    s.add_argument('--budget-tokenizer', required=True, help="reference tokenizer for budgets (usually the first reader)")
    s.add_argument('--oracle-beta-dir', default=None)
    s.add_argument('--device', default='cuda')
    s.add_argument('--shard', type=int, default=0)
    s.add_argument('--num-shards', type=int, default=1)
    s.add_argument('--out-dir', required=True)
    s.set_defaults(func=cmd_select)

    a = sub.add_parser('answer')
    a.add_argument('--out-dir', required=True)
    a.add_argument('--reader-model', required=True)
    a.add_argument('--backend', choices=['hf', 'vllm'], default='hf')
    a.add_argument('--device', default='cuda')
    a.add_argument('--dtype', default='bfloat16')
    a.add_argument('--batch-size', type=int, default=8)
    a.add_argument('--max-model-len', type=int, default=None)
    a.add_argument('--tp', type=int, default=1, help="vLLM tensor parallel size (GPUs visible to this process)")
    a.add_argument('--gpu-memory-utilization', type=float, default=0.9)
    a.add_argument('--chunk', type=int, default=512, help="selections per reader call (and per checkpoint write)")
    a.add_argument('--shard', type=int, default=0)
    a.add_argument('--num-shards', type=int, default=1, help="answer processes (independent of select's shards)")
    a.set_defaults(func=cmd_answer)

    r = sub.add_parser('report')
    r.add_argument('--out-dir', required=True)
    r.add_argument('--ours', default='', help="comma list of arm labels to compare against every other arm")
    r.add_argument('--n-boot', type=int, default=5000)
    r.set_defaults(func=cmd_report)

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
