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
    answer_recall, bh_adjust, bootstrap_mean_ci, cluster_bootstrap_column_means, directional_p, exact_match,
    gold_chunk_recall, holm_adjust, paired_bootstrap_diff, paired_bootstrap_test, token_f1, upgrade_retention,
    upgrade_retention_array,
)
from ttcompress.reader import MAX_NEW_TOKENS, load_reader, reader_tag
from ttcompress.selection import (
    WRAPPERS, Selection, budget_for, make_arm, select_by_scores, select_sentences, select_with_fill,
)
from ttcompress.sources import contains_answer, load_documents, parse_source_list

_PREFIXES = ('pruner:', 'provence:', 'embed:', 'reranker:', 'llmlingua2:', 'llmlingua:', 'longllmlingua:',
             'llmlingua_tt:', 'longllmlingua_tt:', 'recomp:', 'exit:') + tuple(WRAPPERS)


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


def _repair_torn_tail(path):
    """A process killed mid-append leaves a last line without its newline. _read_jsonl drops it, but an
    append after it would glue the next row onto the fragment: a corrupt line mid-file that every later read
    refuses. Cut the file back to its last newline before appending."""
    if not os.path.exists(path):
        return
    with open(path, 'rb+') as f:
        f.seek(0, os.SEEK_END)
        end = f.tell()
        if end == 0:
            return
        f.seek(end - 1)
        if f.read(1) == b'\n':
            return
        pos = end
        while pos > 0:
            step = min(65536, pos)
            f.seek(pos - step)
            block = f.read(step)
            nl = block.rfind(b'\n')
            if nl >= 0:
                pos = pos - step + nl + 1
                break
            pos -= step
        f.truncate(pos)
        print(f"[WARN] {path}: cut a torn last line ({end - pos} bytes) before appending")


def _append_jsonl(path, rows):
    _repair_torn_tail(path)
    with open(path, 'a', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------------------
# select
# ---------------------------------------------------------------------------

def _text_within_budget(arm, doc, ratio, budget, tok, count, attempts=3):
    """Text arms (LLMLingua-2) compress to a rate in THEIR tokenizer (XLM-R), so the result can overshoot
    the budget counted in the reference tokenizer. Tighten the rate by the observed overshoot and retry;
    if it still does not fit, cut the tail to the budget (flagged `truncated`, like a chunk arm whose best
    chunk had to be cut). Arms that take a token target (llmlingua_tt, longllmlingua_tt) are asked for the
    budget itself, then for less by the observed overshoot. -> (text, truncated)"""
    if getattr(arm, 'by_tokens', False):
        target_tokens = float(budget)
        for _ in range(attempts + 2):
            text = arm.compress_text(doc, ratio, target_tokens=max(1, int(target_tokens)))
            n = count(text)
            if n <= budget:
                return text, False
            target_tokens *= 0.98 * budget / n
        ids = tok.encode(text, add_special_tokens=False)[:budget]
        return tok.decode(ids), True
    target = ratio
    for _ in range(attempts):
        text = arm.compress_text(doc, target)
        n = count(text)
        if n <= budget:
            return text, False
        target *= 1.02 * n / budget
    ids = tok.encode(text, add_special_tokens=False)[:budget]
    return tok.decode(ids), True


# published compressors (third-party code): a failure on one document is recorded, not fatal
PUBLISHED_ARMS = ('provence', 'recomp', 'exit', 'llmlingua', 'longllmlingua', 'llmlingua2')
# the same compressors asked for a token target (exploratory: outside the pre-registered H1 family)
TOLERANT_ARMS = PUBLISHED_ARMS + ('llmlingua_tt', 'longllmlingua_tt')


def _arm_rows(arm, d, label, ratios, full_tokens, lengths, tok, count):
    """Selection rows of one document for the given ratios ('full' for the full arm)."""
    base = {'doc_id': d.doc_id, 'arm': label, 'source': d.source, 'language': d.language, 'hop': d.hop,
            'cluster_id': d.cluster_id, 'needle_relpos': d.metadata.get('needle_relpos'),
            'num_chunks': d.num_chunks, 'n_gold': len(d.gold_chunks), 'full_tokens': full_tokens}
    rows = []
    if arm.kind == 'full':
        return [{**base, 'ratio': 'full', 'kept': list(range(d.num_chunks)), 'text': d.text(),
                 'kept_tokens': full_tokens, 'budget': full_tokens, 'gold_recall': 1.0, 'seconds': 0.0}]
    if arm.kind == 'text':
        for r in ratios:
            t0 = time.time()
            budget = budget_for(full_tokens, r)
            text, truncated = _text_within_budget(arm, d, r, budget, tok, count)
            rows.append({**base, 'ratio': r, 'kept': [], 'text': text, 'kept_tokens': count(text), 'budget': budget,
                         'truncated': truncated, 'gold_recall': None, 'seconds': time.time() - t0})
        return rows
    if arm.kind == 'sentence':
        t0 = time.time()
        units, scores = arm.sentence_scores(d)
        seconds = time.time() - t0
        for r in ratios:
            sel = select_sentences(d, units, scores, count, budget_for(full_tokens, r), tok)
            # `kept` = chunks touched by a kept sentence; a partly kept chunk is not whole evidence,
            # so gold-chunk recall is not defined for sentence arms (like text arms)
            rows.append({**base, 'ratio': r, 'kept': sel.kept, 'text': sel.text, 'kept_tokens': sel.kept_tokens,
                         'budget': sel.budget, 'truncated': sel.truncated, 'gold_recall': None, 'seconds': seconds})
        return rows
    if arm.kind == 'fill':
        t0 = time.time()
        scores = arm.scores(d)
        units, unit_scores = arm.sentence_scores(d)
        seconds = time.time() - t0
        for r in ratios:
            sel = select_with_fill(d, scores, lengths, units, unit_scores, count, budget_for(full_tokens, r), tok)
            # `kept` = whole chunks (gold recall counts whole evidence), `partial` = chunks cut to sentences
            rows.append({**base, 'ratio': r, 'kept': sel.kept, 'partial': sel.partial, 'text': sel.text,
                         'kept_tokens': sel.kept_tokens, 'budget': sel.budget, 'truncated': sel.truncated,
                         'gold_recall': gold_chunk_recall(sel.kept, d.gold_chunks) if sel.kept else 0.0,
                         'seconds': seconds})
        return rows
    t0 = time.time()
    scores = arm.scores(d)
    seconds = time.time() - t0
    if scores is None:  # e.g. oracle_beta without labels for this document
        return rows
    for r in ratios:
        sel: Selection = select_by_scores(d, scores, lengths, budget_for(full_tokens, r), tok)
        # a chunk cut to the budget is not counted as kept evidence
        rows.append({**base, 'ratio': r, 'kept': sel.kept, 'text': sel.text, 'kept_tokens': sel.kept_tokens,
                     'budget': sel.budget, 'truncated': sel.truncated,
                     'gold_recall': 0.0 if sel.truncated else gold_chunk_recall(sel.kept, d.gold_chunks),
                     'seconds': seconds})
    return rows


def cmd_select(args):
    from transformers import AutoTokenizer

    os.makedirs(args.out_dir, exist_ok=True)
    ratios = [float(r) for r in args.ratios.split(',')]
    # the document set and its shard assignment must be identical across resumed runs,
    # otherwise documents would be duplicated or dropped between shard files
    meta = {'split': args.split, 'budget_tokenizer': args.budget_tokenizer, 'sources': args.sources, 'n': args.n,
            'num_shards': args.num_shards, 'haystack_chars': args.haystack_chars, 'distractors': args.distractors,
            'multihop_pad_chars': args.multihop_pad_chars}
    if args.multihop_units != 'paragraph':   # only then: select_config.json files written before the option match
        meta['multihop_units'] = args.multihop_units
    if args.offset:
        meta['offset'] = args.offset
    config_path = os.path.join(args.out_dir, 'select_config.json')
    matched = os.path.exists(config_path)
    if matched:
        with open(config_path, encoding='utf-8') as f:
            on_disk = json.load(f)
        if on_disk != meta:
            raise SystemExit(f"{config_path} has {on_disk}, this run asks for {meta}; use a new --out-dir "
                             f"(arms/ratios may change between runs, the document set may not)")
    else:
        _write_json_atomic(config_path, meta)
    doc_path = os.path.join(args.out_dir, f'documents_shard{args.shard}.jsonl')
    if matched and os.path.exists(doc_path):
        # same settings => same documents: reuse the shard written by the first select. This also lets the
        # llmlingua arms run with an old transformers on PYTHONPATH without importing `datasets`.
        docs = [QADocument.from_dict(d) for d in _read_jsonl(doc_path)]
    else:
        docs = []
        for src in parse_source_list(args.sources):
            # --offset k: skip the first k documents of the (nested, hash-ordered) sample, e.g. the main run's test
            # set, so a replication evaluates documents no earlier result has seen
            n = None if args.n is None else args.n + args.offset
            docs.extend(load_documents(src, args.split, n, args.haystack_chars, args.distractors,
                                       args.multihop_pad_chars, args.multihop_units)[args.offset:])
        docs = [d for i, d in enumerate(docs) if i % args.num_shards == args.shard]
        tmp = f'{doc_path}.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            f.writelines(json.dumps(d.to_dict(), ensure_ascii=False) + '\n' for d in docs)
        os.replace(tmp, doc_path)
    if args.restrict_sources:
        # a pass over part of the document set (e.g. extra ratios for single-hop only); the set itself and its
        # shard assignment stay the ones select_config.json fixed
        keep = set(parse_source_list(args.restrict_sources))
        docs = [d for d in docs if d.source in keep]
    sel_path = os.path.join(args.out_dir, f'selections_shard{args.shard}.jsonl')
    done = {(r['doc_id'], r['arm'], r['ratio']) for r in _read_jsonl(sel_path)}

    tok = AutoTokenizer.from_pretrained(args.budget_tokenizer, trust_remote_code=True)
    count = lambda text: len(tok.encode(text, add_special_tokens=False))  # noqa: E731
    lengths = {d.doc_id: [count(c) for c in d.chunks] for d in docs}
    full_tokens = {d.doc_id: count(d.text()) for d in docs}
    print(f"{len(docs)} documents in shard {args.shard}/{args.num_shards} ({args.split}); {len(done)} selections on disk")

    fail_path = os.path.join(args.out_dir, f'select_failures_shard{args.shard}.jsonl')
    failed = {(r['doc_id'], r['arm']) for r in _read_jsonl(fail_path)} if not args.retry_failed else set()
    for label, spec in parse_arms(args.arms):
        if spec.partition(':')[0] in PUBLISHED_ARMS and label not in H1_PUBLISHED:
            print(f"[WARN] {label}={spec}: a published compressor under a label H1 does not know "
                  f"({', '.join(H1_PUBLISHED)}): it is evaluated but left out of H1", flush=True)
        wanted = ['full'] if spec == 'full' else ratios
        # only the missing (doc, ratio) rows: a re-select with an extra ratio must not duplicate the others
        need = {d.doc_id: [r for r in wanted if (d.doc_id, label, r) not in done] for d in docs}
        pending = [d for d in docs if need[d.doc_id] and (d.doc_id, label) not in failed]
        if not pending:
            continue
        arm = make_arm(spec, args.device, args.oracle_beta_dir)
        tolerant = spec.partition(':')[0] in TOLERANT_ARMS   # third-party compressors
        if arm.kind != 'full':
            # untimed warm-up: the first call pays model start-up (CUDA kernels, caches, lazy imports), which
            # would otherwise land in one document's time in the RQ1 cost table
            try:
                _arm_rows(arm, pending[0], label, wanted[:1], full_tokens[pending[0].doc_id],
                          lengths[pending[0].doc_id], tok, count)
            except Exception:  # noqa: BLE001 -- the timed call below records (or raises) the same failure
                if not tolerant:
                    raise
        rows, n_rows, n_failed, t_arm = [], 0, 0, time.time()
        for i, d in enumerate(pending):
            try:
                rows.extend(_arm_rows(arm, d, label, need[d.doc_id], full_tokens[d.doc_id], lengths[d.doc_id],
                                      tok, count))
            except Exception as exc:  # noqa: BLE001 -- a published compressor's own failure on one document
                if not tolerant:
                    raise
                n_failed += 1
                _append_jsonl(fail_path, [{'doc_id': d.doc_id, 'arm': label, 'error': f'{type(exc).__name__}: {exc}'}])
                print(f"  [WARN] {label} failed on {d.doc_id}: {type(exc).__name__}: {exc}", flush=True)
            if len(rows) >= args.flush_every * len(wanted) or i == len(pending) - 1:
                _append_jsonl(sel_path, rows)   # progress survives a crash late in a slow arm
                n_rows += len(rows)
                rows = []
        print(f"  {label}: {n_rows} selections in {time.time() - t_arm:.1f}s"
              + (f"; {n_failed} documents failed (see {fail_path})" if n_failed else ''), flush=True)
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
              'by_depth': [], 'heldout_readers': [h for h in (args.heldout_readers or '').split(',') if h],
              'n_boot': args.n_boot}

    for (reader, source, ratio, arm), by_doc in sorted(cell.items()):
        docs = sorted(by_doc)
        clusters = [by_doc[d]['cluster_id'] for d in docs]
        entry = {'reader': reader, 'source': source, 'ratio': ratio, 'arm': arm, 'n': len(docs),
                 'n_clusters': len(set(clusters)),
                 'compression': float(np.mean([by_doc[d]['full_tokens'] / max(1, by_doc[d]['kept_tokens']) for d in docs])),
                 'select_ms': 1000 * float(np.mean([by_doc[d]['seconds'] for d in docs])),
                 'select_ms_median': 1000 * float(np.median([by_doc[d]['seconds'] for d in docs])),
                 'truncated_rate': float(np.mean([bool(by_doc[d].get('truncated')) for d in docs])),
                 # text arms (LLMLingua-2) compress to a rate, not a hard budget: realized tokens may overshoot
                 'over_budget_rate': 0.0 if ratio == 'full' else float(np.mean(
                     [by_doc[d]['kept_tokens'] > by_doc[d]['budget'] * (1 + args.budget_tolerance) for d in docs]))}
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
    # the whole paired table is one family: hundreds of tests at alpha=0.05 yield dozens of raw "wins" by chance
    raw = [p['p'] for p in report['paired']]
    for p, holm, bh in zip(report['paired'], holm_adjust(raw), bh_adjust(raw)):
        p['p_holm'], p['q_bh'] = holm, bh

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
                    gap = full_mean[strong] - full_mean[weak]
                    # a ratio over a near-zero full-context gap is noise, whatever its CI says
                    report['upgrade_retention'].append({'source': source, 'weak': weak, 'strong': strong,
                                                        'ratio': ratio, 'arm': arm, 'retention': point,
                                                        'ci95': [lo, hi], 'n': len(common), 'gap': gap,
                                                        'stable': gap >= args.min_upgrade_gap})

    # the confirmatory families are fixed to the pre-registered ratios: ratios added later (e.g. 16x/32x on
    # single-hop) are reported and compared, never tested in a family
    confirm = ratios
    if args.confirm_ratios:
        wanted = {_ratio_key(float(x)) for x in args.confirm_ratios.split(',') if x.strip()}
        confirm = [r for r in ratios if r in wanted]
    report['confirm_ratios'] = confirm
    report['hypotheses'] = hypothesis_families(cell, rows, readers, sources, confirm, args)
    report['outside_families'] = arms_outside_families(arms)
    report['h2a_multi_support'] = multi_support_slice(cell, readers, rows, confirm, args)
    report['seeds'] = seed_table(report['cells'])
    if args.labels_dir:
        report['cost'] = cost_tables(report['cells'], args)
    if args.fit_dir:
        report['label_quality'] = label_quality(args.fit_dir)
    report['oracle_gap_by_reader'] = oracle_gap_by_reader(cell, readers, confirm, args.n_boot)
    if args.prereg:
        report['prereg'] = prereg_families(cell, rows, args.prereg, args.n_boot)
    if not args.no_diagnostics:
        report['diagnostics'] = selection_diagnostics(args.out_dir, cell, args.primary_reader, ours)

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


def oracle_gap_by_reader(cell, readers, ratios, n_boot):
    """ours_beta - oracle_beta for EVERY reader, on the oracle documents (exploratory). oracle_beta's labels
    were measured with the primary reader on these very documents, so its lead on that reader mixes what
    distillation loses with what only fits that reader's behaviour; the lead on the other readers is the part
    that transfers."""
    out = []
    for reader, source, ratio in sorted({(k[0], k[1], k[2]) for k in cell if k[3] == 'oracle_beta'}):
        if ratio not in ratios:
            continue
        a, b = cell.get((reader, source, ratio, 'ours_beta')), cell.get((reader, source, ratio, 'oracle_beta'))
        if not (a and b):
            continue
        common = sorted(set(a) & set(b))
        if len(common) < 2:
            continue
        res = paired_bootstrap_diff([a[d]['f1'] for d in common], [b[d]['f1'] for d in common],
                                    [a[d]['cluster_id'] for d in common], n_boot=n_boot)
        out.append({'reader': reader, 'source': source, 'ratio': ratio, 'n': len(common), **res})
    return out


def _jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 1.0


def selection_diagnostics(out_dir, cell, primary_reader, ours):
    """Exploratory, from the selection files (METHOD_SPEC.md §6, run report 2026-09-30):
    - coverage: share of selections whose text contains a gold answer string, share keeping every gold chunk
      whole, chunks touched, share of the budget used -- with 10-paragraph documents, whether the answer
      survives decides most of the F1 difference between arms;
    - F1 of the primary reader given the answer string is / is not in the compressed context;
    - bridge effect: F1 given an answer chunk is kept, split by whether every gold chunk is kept (chunk arms);
    - overlap (Jaccard of kept chunks) of our arms with their seed reruns, each other, span_sup, reranker and
      oracle_beta: a label change that moves the selection less than a new training seed cannot show up in
      any F1 comparison."""
    docs = {}
    for path in glob.glob(os.path.join(out_dir, 'documents_shard*.jsonl')):
        for d in _read_jsonl(path):
            docs[d['doc_id']] = {'source': d['source'], 'hop': d['hop'], 'answers': d['answers'],
                                 'gold': set(d['gold_chunks']),
                                 'ans_chunks': {i for i, c in enumerate(d['chunks']) if contains_answer(c, d['answers'])}}
    if not docs:
        return {}
    cov = defaultdict(lambda: defaultdict(list))
    kept_sets = {}   # (doc, arm, ratio) -> kept, chunk arms only
    in_ctx = {}      # (doc, arm, ratio) -> answer in context
    for path in glob.glob(os.path.join(out_dir, 'selections_shard*.jsonl')):
        with open(path, encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    s = json.loads(line)
                except json.JSONDecodeError:
                    continue
                d = docs.get(s['doc_id'])
                if d is None or s['ratio'] == 'full':
                    continue
                ratio = _ratio_key(s['ratio'])
                key = (s['doc_id'], s['arm'], ratio)
                inside = contains_answer(s.get('text', ''), d['answers'])
                in_ctx[key] = inside
                c = cov[(d['source'], ratio, s['arm'])]
                c['ans_in_ctx'].append(float(inside))
                c['budget_use'].append(s['kept_tokens'] / max(1, s['budget']))
                touched = set(s.get('kept') or []) | set(s.get('partial') or [])
                if touched:
                    c['touched'].append(len(touched))
                if s.get('gold_recall') is not None and d['gold']:
                    whole = set(s.get('kept') or [])
                    c['all_gold'].append(float(d['gold'] <= whole))
                    kept_sets[key] = whole
    coverage = []
    for (source, ratio, arm), c in sorted(cov.items()):
        row = {'source': source, 'ratio': ratio, 'arm': arm, 'n': len(c['ans_in_ctx']),
               'answer_in_context': float(np.mean(c['ans_in_ctx'])), 'budget_use': float(np.mean(c['budget_use'])),
               'touched': float(np.mean(c['touched'])) if c['touched'] else None,
               'all_gold': float(np.mean(c['all_gold'])) if c['all_gold'] else None}
        by_doc = cell.get((primary_reader, source, ratio, arm)) if primary_reader else None
        if by_doc:
            f_in = [v['f1'] for doc_id, v in by_doc.items() if in_ctx.get((doc_id, arm, ratio)) is True]
            f_out = [v['f1'] for doc_id, v in by_doc.items() if in_ctx.get((doc_id, arm, ratio)) is False]
            row['f1_answer_in'] = float(np.mean(f_in)) if f_in else None
            row['f1_answer_out'] = float(np.mean(f_out)) if f_out else None
        coverage.append(row)

    bridge = []
    if primary_reader:
        acc = defaultdict(lambda: ([], []))
        for (doc_id, arm, ratio), whole in kept_sets.items():
            d = docs[doc_id]
            if d['hop'] != 'multi' or not (d['ans_chunks'] & whole) or arm.startswith('oracle'):
                continue
            v = cell.get((primary_reader, d['source'], ratio, arm), {}).get(doc_id)
            if v is not None:
                acc[(d['source'], ratio)][0 if d['gold'] <= whole else 1].append(v['f1'])
        for (source, ratio), (all_gold, missing) in sorted(acc.items()):
            bridge.append({'source': source, 'ratio': ratio,
                           'f1_all_gold': float(np.mean(all_gold)) if all_gold else None, 'n_all_gold': len(all_gold),
                           'f1_some_gold_missing': float(np.mean(missing)) if missing else None,
                           'n_some_gold_missing': len(missing)})

    overlap = []
    arms_present = {k[1] for k in kept_sets}
    for base in ours:
        others = [a for a in sorted(arms_present) if a != base and (
            SEED_ARM.match(a) and SEED_ARM.match(a).group(1) == base or a in ours
            or a in ('span_sup', 'reranker', 'oracle_beta', 'abl_logprob', 'abl_posadj'))]
        for source, ratio in sorted({(docs[k[0]]['source'], k[2]) for k in kept_sets if k[1] == base}):
            row = {'source': source, 'ratio': ratio, 'arm': base}
            for other in others:
                js = [_jaccard(kept_sets[k], kept_sets[(k[0], other, k[2])]) for k in kept_sets
                      if k[1] == base and k[2] == ratio and docs[k[0]]['source'] == source
                      and (k[0], other, k[2]) in kept_sets]
                if js:
                    row[other] = float(np.mean(js))
            overlap.append(row)
    return {'coverage': coverage, 'bridge': bridge, 'overlap': overlap}


def _retention_ci(arr: np.ndarray, clusters, n_boot: int, seed: int = 42):
    boot = cluster_bootstrap_column_means(arr, clusters, n_boot, seed)
    vals = upgrade_retention_array(boot[:, 0], boot[:, 1], boot[:, 2], boot[:, 3])
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return None, None
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


# ---------------------------------------------------------------------------
# confirmatory hypothesis families (METHOD_SPEC.md §1)
# ---------------------------------------------------------------------------
# Each hypothesis is a small pre-specified family of one-sided tests, Holm-corrected WITHIN the family
# (the big paired table above is exploratory). Arm labels are the ones run_pipeline.sh assigns; a family
# whose arms were not run comes out empty ("not run").

H1_BASELINES = ('bm25', 'embed', 'reranker', 'lead', 'random')
# published compressors (EXTRA_ARMS labels); only those that were run are tested
H1_PUBLISHED = ('provence', 'xprovence', 'recomp', 'exit', 'llmlingua', 'longllmlingua', 'llmlingua2')
H2A_VS = ('span_sup', 'oracle_span')
FAMILY_ALPHA = 0.05


def _paired_family(cell, readers, meta, source_ok, ratios, ours_arms, vs_arms, kind, margin, n_boot):
    tests = []
    for reader, source, ratio, ours, vs in itertools.product(readers, sorted(meta), ratios, ours_arms, vs_arms):
        a, b = cell.get((reader, source, ratio, ours)), cell.get((reader, source, ratio, vs))
        if not (source_ok(meta[source]) and a and b):
            continue
        common = sorted(set(a) & set(b))
        if len(common) < 2:
            continue
        res = paired_bootstrap_test([a[d]['f1'] for d in common], [b[d]['f1'] for d in common],
                                    [a[d]['cluster_id'] for d in common], kind, margin, n_boot)
        # budget matching is checked on realized tokens, not assumed (text arms compress to a rate)
        tokens = float(np.mean([a[d]['kept_tokens'] for d in common]) /
                       max(1.0, np.mean([b[d]['kept_tokens'] for d in common])))
        tests.append({'reader': reader, 'source': source, 'ratio': ratio, 'ours': ours, 'vs': vs, **res,
                      'n_clusters': len({a[d]['cluster_id'] for d in common}), 'tokens_ours_over_vs': tokens,
                      'budget_mismatch': abs(tokens - 1.0) > 0.1})
    return tests


def _retention_family(cell, meta, readers, ratios, arm_a, arm_b, heldout, want_heldout, min_gap, n_boot):
    """retention(arm_a) - retention(arm_b) for the same weak->strong pair, source, ratio and documents
    (a paired bootstrap of the difference, not two overlapping marginal CIs)."""
    tests = []
    for source in sorted(meta):
        full = {r: cell.get((r, source, 'full', 'full')) for r in readers}
        full_mean = {r: float(np.mean([v['f1'] for v in c.values()])) for r, c in full.items() if c}
        for weak, strong in itertools.permutations(full_mean, 2):
            gap = full_mean[strong] - full_mean[weak]
            if gap < min_gap or ((weak in heldout or strong in heldout) != want_heldout):
                continue
            for ratio in ratios:
                cols = [cell.get((weak, source, ratio, arm_a)), cell.get((strong, source, ratio, arm_a)),
                        cell.get((weak, source, ratio, arm_b)), cell.get((strong, source, ratio, arm_b)),
                        full[weak], full[strong]]
                if not all(cols):
                    continue
                common = sorted(set.intersection(*(set(c) for c in cols)))
                if len(common) < 2:
                    continue
                arr = np.array([[c[d]['f1'] for c in cols] for d in common])

                def stat(m):
                    return (upgrade_retention_array(m[..., 0], m[..., 1], m[..., 4], m[..., 5])
                            - upgrade_retention_array(m[..., 2], m[..., 3], m[..., 4], m[..., 5]))

                boot = stat(cluster_bootstrap_column_means(arr, [cols[4][d]['cluster_id'] for d in common], n_boot))
                finite = boot[~np.isnan(boot)]
                lo, hi = (float(q) for q in np.quantile(finite, [0.025, 0.975])) if len(finite) else (None, None)
                tests.append({'source': source, 'weak': weak, 'strong': strong, 'ratio': ratio, 'ours': arm_a,
                              'vs': arm_b, 'diff': float(stat(arr.mean(axis=0))), 'ci95': [lo, hi],
                              'p': directional_p(boot, 'superiority') if len(finite) else None, 'n': len(common),
                              'gap': gap})
    return tests


def hypothesis_families(cell, rows, readers, sources, ratios, args):
    meta = {r['source']: {'language': r['language'], 'hop': r['hop']} for r in rows}
    main = [args.primary_reader] if args.primary_reader in readers else readers
    heldout = {h for h in (args.heldout_readers or '').split(',') if h}
    every = lambda m: True  # noqa: E731
    fam = [
        # one family: the cheap baselines and the published compressors are the same claim ("beats every
        # non-oracle compressor"), so Holm corrects over all of them together
        ('H1', 'ours_beta > each non-oracle compressor (bm25, embed, reranker, lead, random and every published '
               'compressor that was run), every source and ratio', 'superiority', 0.0,
         _paired_family(cell, main, meta, every, ratios, ['ours_beta'], H1_BASELINES + H1_PUBLISHED, 'superiority',
                        0.0, args.n_boot)),
        ('H1-oracle', f'ours_beta within {args.oracle_margin} F1 of oracle_beta (non-inferiority; '
                      f'oracle_beta covers the ORACLE_N labeled test docs)', 'noninferiority', args.oracle_margin,
         _paired_family(cell, main, meta, every, ratios, ['ours_beta'], ['oracle_beta'], 'noninferiority',
                        args.oracle_margin, args.n_boot)),
        ('H2a', 'multi-hop: ours_beta > span_sup and > oracle_span', 'superiority', 0.0,
         _paired_family(cell, main, meta, lambda m: m['hop'] == 'multi', ratios, ['ours_beta'], H2A_VS,
                        'superiority', 0.0, args.n_boot)),
        ('H2b', f'single-hop: ours_beta equivalent to span_sup within ±{args.equiv_margin} F1 (TOST)',
         'equivalence', args.equiv_margin,
         _paired_family(cell, main, meta, lambda m: m['hop'] == 'single', ratios, ['ours_beta'], ['span_sup'],
                        'equivalence', args.equiv_margin, args.n_boot)),
        ('H3', f'retention(ours_ens) > retention(ours_beta), label-reader pairs with full-context gap '
                f'>= {args.min_upgrade_gap}', 'superiority', 0.0,
         _retention_family(cell, meta, readers, ratios, 'ours_ens', 'ours_beta', heldout, False,
                           args.min_upgrade_gap, args.n_boot)),
        ('H3-heldout', 'same, pairs involving the held-out reader', 'superiority', 0.0,
         _retention_family(cell, meta, readers, ratios, 'ours_ens', 'ours_beta', heldout, True,
                           args.min_upgrade_gap, args.n_boot)),
    ]
    out = []
    variants = seed_variants(sorted({a for (_, _, _, a) in cell}))
    for name, claim, kind, margin, tests in fam:
        for t, h in zip(tests, holm_adjust([t['p'] for t in tests])):
            t['p_holm'] = h
            t['supported'] = h is not None and h < FAMILY_ALPHA
            if 'reader' in t and variants.get(t['ours']):
                _seed_check(t, cell, variants[t['ours']], kind, margin)
        seeded = [t for t in tests if 'seeds_agree' in t]
        out.append({'name': name, 'claim': claim, 'kind': kind,
                    'margin': margin, 'n_tests': len(tests), 'n_supported': sum(t['supported'] for t in tests),
                    'n_budget_mismatch': sum(bool(t.get('budget_mismatch')) for t in tests),
                    # supported AND every other training seed's point estimate points the same way
                    'n_seed_robust': sum(t['supported'] and t['seeds_agree'] for t in seeded) if seeded else None,
                    'tests': tests})
    return out


def prereg_families(cell, rows, prereg_path, n_boot):
    """Families fixed in a file BEFORE the documents they test were evaluated (a replication of an exploratory
    finding, docs/PREREG_*.md). Each family: one reader, one of our arms against one or more arms, a test kind
    (superiority | noninferiority | equivalence) and margin, sources and ratios; one-sided paired cluster
    bootstrap, Holm within the family, the same machinery as the confirmatory families. Seed reruns of `ours`
    (<ours>_s<k>) give the "every seed agrees" check."""
    with open(prereg_path, encoding='utf-8') as f:
        spec = json.load(f)
    meta = {r['source']: {'language': r['language'], 'hop': r['hop']} for r in rows}
    variants = seed_variants(sorted({a for (_, _, _, a) in cell}))
    out = []
    for fam in spec['families']:
        sub = {s: meta[s] for s in fam['sources'] if s in meta}
        ratios = [_ratio_key(float(r)) for r in fam['ratios']]
        tests = _paired_family(cell, [fam['reader']], sub, lambda m: True, ratios, [fam['ours']], fam['vs'],
                               fam['kind'], fam.get('margin', 0.0), n_boot)
        for t, h in zip(tests, holm_adjust([t['p'] for t in tests])):
            t['p_holm'] = h
            t['supported'] = h is not None and h < fam.get('alpha', FAMILY_ALPHA)
            if variants.get(t['ours']):
                _seed_check(t, cell, variants[t['ours']], fam['kind'], fam.get('margin', 0.0))
        seeded = [t for t in tests if 'seeds_agree' in t]
        out.append({'name': fam['name'], 'claim': fam['claim'], 'kind': fam['kind'], 'margin': fam.get('margin', 0.0),
                    'n_tests': len(tests), 'n_supported': sum(t['supported'] for t in tests),
                    'n_budget_mismatch': sum(bool(t.get('budget_mismatch')) for t in tests),
                    'n_seed_robust': sum(t['supported'] and t['seeds_agree'] for t in seeded) if seeded else None,
                    'tests': tests})
    return {'name': spec.get('name', os.path.basename(prereg_path)), 'fixed': spec.get('fixed'), 'families': out}


def arms_outside_families(arms) -> list:
    """Arms no confirmatory family reads, beyond the expected ones (reference, oracles, ablations, seed
    reruns). A published compressor run under a label H1 does not know lands here instead of in H1."""
    known = {'full', 'ours_beta', 'ours_ens', 'oracle_beta', 'oracle_support', *H1_BASELINES, *H1_PUBLISHED, *H2A_VS}
    return sorted(a for a in arms if a not in known and not a.startswith('abl_') and not SEED_ARM.match(a))


def multi_support_slice(cell, readers, rows, ratios, args):
    """Exploratory: H2a's comparisons on multi-hop documents with >= 2 supporting paragraphs only. About half
    of VIMQA's questions have a single supporting paragraph, where the answer-span label already covers the
    evidence and H2a's mechanism (bridge paragraphs) cannot act. Not a confirmatory family."""
    meta = {r['source']: r['hop'] for r in rows}
    main = [args.primary_reader] if args.primary_reader in readers else readers
    tests = []
    for reader, source, ratio, vs in itertools.product(main, sorted(meta), ratios, H2A_VS):
        a, b = cell.get((reader, source, ratio, 'ours_beta')), cell.get((reader, source, ratio, vs))
        if meta[source] != 'multi' or not (a and b):
            continue
        common = sorted(d for d in set(a) & set(b) if (a[d].get('n_gold') or 0) >= 2)
        if len(common) < 2:
            continue
        res = paired_bootstrap_test([a[d]['f1'] for d in common], [b[d]['f1'] for d in common],
                                    [a[d]['cluster_id'] for d in common], 'superiority', 0.0, args.n_boot)
        tests.append({'reader': reader, 'source': source, 'ratio': ratio, 'vs': vs, **res, 'n': len(common)})
    return tests


def _multi_support_markdown(tests) -> list:
    if not tests:
        return []
    lines = ['## H2a on multi-hop documents with >= 2 supporting paragraphs (exploratory)', '',
             '| reader | source | ratio | ours_beta vs | n | Δ F1 [95% CI] | p (one-sided) |', '|---|---|---|---|---|---|---|']
    for t in tests:
        lo, hi = t['ci95']
        ci = '' if lo is None else f" [{lo:+.3f}, {hi:+.3f}]"
        p = '' if t['p'] is None else f"{t['p']:.4f}"
        lines.append(f"| {t['reader']} | {t['source']} | {t['ratio']} | {t['vs']} | {t['n']} | {t['diff']:+.3f}{ci} | {p} |")
    return lines + ['']


SEED_ARM = re.compile(r'^(ours_beta|ours_ens|ours_sent|ours_fill)_s(\d+)$')


def seed_variants(arms):
    """{'ours_beta': ['ours_beta_s1', ...], ...}: the same pruner retrained with other seeds (EXTRA_SEEDS)."""
    out = defaultdict(list)
    for a in arms:
        m = SEED_ARM.match(a)
        if m:
            out[m.group(1)].append(a)
    return dict(out)


def _seed_check(test, cell, variants, kind, margin):
    """Point estimate of the same comparison for every other training seed, on that seed's common documents."""
    diffs = {}
    for v in variants:
        a = cell.get((test['reader'], test['source'], test['ratio'], v))
        b = cell.get((test['reader'], test['source'], test['ratio'], test['vs']))
        common = sorted(set(a or {}) & set(b or {}))
        if common:
            diffs[v] = float(np.mean([a[d]['f1'] - b[d]['f1'] for d in common]))
    holds = {'superiority': lambda d: d > 0, 'noninferiority': lambda d: d > -margin,
             'equivalence': lambda d: abs(d) < margin}[kind]
    test['seed_diffs'] = diffs
    test['seeds_agree'] = bool(diffs) and all(holds(d) for d in diffs.values())


def seed_table(cells):
    """F1 of ours_beta / ours_ens per training seed, per reader x source x ratio: mean and SD across seeds,
    to set against the effect sizes (a gap smaller than the seed SD is not a finding)."""
    variants = seed_variants(sorted({c['arm'] for c in cells}))
    by = {(c['reader'], c['source'], c['ratio'], c['arm']): c['f1']['mean'] for c in cells}
    out = []
    for base, vs in sorted(variants.items()):
        for reader, source, ratio in sorted({k[:3] for k in by if k[3] == base}):
            f1 = {a: by[(reader, source, ratio, a)] for a in [base] + vs if (reader, source, ratio, a) in by}
            if len(f1) > 1:
                vals = list(f1.values())
                out.append({'arm': base, 'reader': reader, 'source': source, 'ratio': ratio, 'f1_by_seed': f1,
                            'mean': float(np.mean(vals)), 'sd': float(np.std(vals, ddof=1))})
    return out


def cost_tables(cells, args):
    """RQ1: what one pruner pass replaces. Label cost per (reader, source_split) from the Stage A raw
    records (reader calls = K masks + 1 full context; `seconds` = the document's share of its batch's
    wall-clock on one process), next to per-document selection time of every arm."""
    from ttcompress.attribution import record_paths
    labels, seen = [], set()
    # a comma list: an ablation run's own labels, then the main run's (for the sources it did not re-measure).
    # A label set in both is the ablation's: the first root wins, as in run_pipeline.sh's raw_root.
    roots = [d for d in args.labels_dir.split(',') if d]
    for root in roots:
        for split_dir in sorted(glob.glob(os.path.join(root, 'raw', '*', '*'))):
            reader_dir = os.path.dirname(split_dir)
            key = (os.path.basename(reader_dir), os.path.basename(split_dir))
            if key in seen:
                continue
            secs, calls = [], []
            for p in record_paths(split_dir):
                with open(p, encoding='utf-8') as f:
                    rec = json.load(f)
                secs.append(rec.get('seconds', float('nan')))
                calls.append(len(rec['masks']) + 1)
            if calls:
                seen.add(key)
                labels.append({'reader': key[0], 'set': key[1],
                               'n_docs': len(calls), 'calls_per_doc': float(np.mean(calls)),
                               'seconds_per_doc': float(np.nanmean(secs)), 'total_hours': float(np.nansum(secs) / 3600)})
    labels.sort(key=lambda r: (r['reader'], r['set']))
    select = defaultdict(list)  # selection time does not depend on the reader
    for c in cells:
        if c['ratio'] != 'full':
            select[(c['source'], c['arm'])].append(c['select_ms'])
    return {'labels': labels,
            'select_ms': [{'source': s, 'arm': a, 'select_ms': float(np.mean(v))} for (s, a), v in sorted(select.items())]}


def label_quality(fit_dir):
    """Stage A diagnostics of this run's fits (METHOD_SPEC.md §3): additivity (cv R²), whether β finds the
    evidence (gold recall@|gold|, MRR; ~1 on single-hop = β collapsed onto the answer span, H2b), position
    share of label variance, and cross-reader agreement for ensembles (descriptive, RQ3)."""
    out = []
    for path in sorted(glob.glob(os.path.join(fit_dir, '*', '*', '*', 'summary.json'))):
        with open(path, encoding='utf-8') as f:
            s = json.load(f)
        label_set = os.path.dirname(path)
        target_dir = os.path.dirname(label_set)
        out.append({'reader': os.path.basename(os.path.dirname(target_dir)), 'target': os.path.basename(target_dir),
                    'set': os.path.basename(label_set),
                    **{k: s.get(k) for k in ('n_docs', 'n_informative', 'alpha', 'mean_full_f1', 'mean_cv_r2',
                                             'gold_recall_at_g', 'gold_mrr', 'position_r2')},
                    'cross_reader_spearman': s.get('cross_reader_spearman')})
    return out


def _fmt(m):
    lo, hi = m['ci95']
    return f"{m['mean']:.3f}" + (f" [{lo:.3f}, {hi:.3f}]" if lo is not None else '')


def _markdown(report) -> str:
    lines = ['# Evaluation report', '', 'Token F1 with 95% cluster-bootstrap CI; never pooled across sources.', '']
    lines += _hypotheses_markdown(report.get('hypotheses', []))
    if report.get('prereg'):
        pr = report['prereg']
        lines += _hypotheses_markdown(pr['families'], title=f"Pre-registered families: {pr['name']}",
                                      note=f"Fixed {pr.get('fixed')}, before these documents were evaluated.")
    if report.get('outside_families'):
        lines += [f"Arms in no confirmatory family: {', '.join(report['outside_families'])}. A published "
                  f"compressor is tested in H1 only under one of these labels: {', '.join(H1_PUBLISHED)}.", '']
    lines += _multi_support_markdown(report.get('h2a_multi_support', []))
    lines += _cost_markdown(report.get('cost'))
    lines += _label_quality_markdown(report.get('label_quality'))
    lines += _seeds_markdown(report.get('seeds'))
    lines += _oracle_gap_markdown(report.get('oracle_gap_by_reader'))
    lines += _diagnostics_markdown(report.get('diagnostics'))
    groups = defaultdict(list)
    for c in report['cells']:
        groups[(c['reader'], c['source'])].append(c)
    for (reader, source), cells in sorted(groups.items()):
        lines += [f'## {source} — reader `{reader}`', '',
                  '| ratio | arm | F1 | EM | answer recall | gold recall | n (clusters) | realized compression | '
                  'over budget | truncated | select ms |',
                  '|---|---|---|---|---|---|---|---|---|---|---|']
        for c in sorted(cells, key=lambda c: (c['ratio'] != 'full', c['ratio'], -c['f1']['mean'])):
            gr = c['gold_recall']['mean']
            gr_text = '' if gr != gr else f"{gr:.3f}"
            lines.append(f"| {c['ratio']} | {c['arm']} | {_fmt(c['f1'])} | {c['em']['mean']:.3f} | "
                         f"{c['answer_recall']['mean']:.3f} | "
                         f"{gr_text} | {c['n']} ({c['n_clusters']}) | {c['compression']:.1f}x | "
                         f"{c.get('over_budget_rate', 0.0):.0%} | {c['truncated_rate']:.0%} | {c['select_ms']:.0f} |")
        lines.append('')
    if report['paired']:
        tested = [p for p in report['paired'] if p['p'] is not None]
        n_sig = {k: sum(p[k] < 0.05 for p in tested) for k in ('p', 'q_bh', 'p_holm')}
        fmt = lambda v: '' if v is None else f"{v:.3f}"  # noqa: E731
        lines += ['## Paired differences (ours − other, token F1)', '',
                  f"{len(tested)} tests, one family. Below 0.05: raw p {n_sig['p']}, BH q {n_sig['q_bh']}, "
                  f"Holm p {n_sig['p_holm']}. Quote BH q (FDR), not raw p. A bootstrap p is at least "
                  f"1/(n_boot+1), so over {len(tested)} tests Holm cannot fall below "
                  f"{min(1.0, len(tested) / (report.get('n_boot', 5000) + 1)):.3f}: here it is only a floor "
                  f"check (the confirmatory families are small enough for Holm).", '',
                  '| reader | source | ratio | ours | vs | Δ F1 [95% CI] | p | q (BH) | p (Holm) |',
                  '|---|---|---|---|---|---|---|---|---|']
        for p in report['paired']:
            lo, hi = p['ci95']
            ci = f" [{lo:+.3f}, {hi:+.3f}]" if lo is not None else ''
            lines.append(f"| {p['reader']} | {p['source']} | {p['ratio']} | {p['ours']} | {p['vs']} | "
                         f"{p['diff']:+.3f}{ci} | {fmt(p['p'])} | {fmt(p['q_bh'])} | {fmt(p['p_holm'])} |")
        lines.append('')
    if report['upgrade_retention']:
        lines += ['## Upgrade retention (share of the full-context weak→strong gain kept)', '',
                  'Rows marked ⚠ have a full-context gap below --min-upgrade-gap: the ratio is noise.', '',
                  '| source | weak → strong | full gap | ratio | arm | retention [95% CI] |', '|---|---|---|---|---|---|']
        for u in report['upgrade_retention']:
            lo, hi = u['ci95']
            ci = f" [{lo:.2f}, {hi:.2f}]" if lo is not None else ''
            flag = '' if u.get('stable', True) else ' ⚠'
            lines.append(f"| {u['source']} | {u['weak']} → {u['strong']} | {u.get('gap', float('nan')):.3f}{flag} | "
                         f"{u['ratio']} | {u['arm']} | {u['retention']:.2f}{ci} |")
        lines.append('')
    return '\n'.join(lines)


def _hypotheses_markdown(families, title='Hypotheses (confirmatory; one-sided tests, Holm within each family, '
                                          'alpha 0.05)', note=None) -> list:
    if not families:
        return []
    lines = [f'## {title}', ''] + ([note, ''] if note else []) + [
             '| family | claim | tests | supported | supported, every seed agrees | budget-mismatched |',
             '|---|---|---|---|---|---|']
    for f in families:
        status = 'not run' if not f['n_tests'] else f"{f['n_supported']}/{f['n_tests']}"
        robust = '' if f.get('n_seed_robust') is None else f"{f['n_seed_robust']}/{f['n_tests']}"
        lines.append(f"| {f['name']} | {f['claim']} | {f['n_tests']} | {status} | {robust} | "
                     f"{f['n_budget_mismatch']} |")
    lines.append('')
    for f in families:
        if not f['n_tests']:
            continue
        lines += [f"### {f['name']}: {f['claim']}", '',
                  '| reader / pair | source | ratio | ours | vs | Δ [95% CI] | p | p (Holm) | ok |',
                  '|---|---|---|---|---|---|---|---|---|']
        for t in f['tests']:
            who = t['reader'] if 'reader' in t else f"{t['weak']} → {t['strong']}"
            lo, hi = t['ci95']
            ci = f" [{lo:+.3f}, {hi:+.3f}]" if lo is not None else ''
            p = '' if t['p'] is None else f"{t['p']:.4f}"
            holm = '' if t['p_holm'] is None else f"{t['p_holm']:.4f}"
            ok = ('✓' if t['supported'] else '✗') + (' (budget ≠)' if t.get('budget_mismatch') else '')
            if 'seeds_agree' in t:
                ok += ' seeds ' + ('✓' if t['seeds_agree'] else '✗ ' + ', '.join(
                    f"{d:+.3f}" for d in t['seed_diffs'].values()))
            lines.append(f"| {who} | {t['source']} | {t['ratio']} | {t['ours']} | {t['vs']} | {t['diff']:+.3f}{ci} | "
                         f"{p} | {holm} | {ok} |")
        lines.append('')
    return lines


def _seeds_markdown(rows) -> list:
    if not rows:
        return []
    lines = ['## Training-seed variation (token F1 per seed; compare the SD with the effect sizes above)', '',
             '| arm | reader | source | ratio | F1 per seed | mean ± SD |', '|---|---|---|---|---|---|']
    for r in rows:
        per = ', '.join(f"{a}: {v:.3f}" for a, v in r['f1_by_seed'].items())
        lines.append(f"| {r['arm']} | {r['reader']} | {r['source']} | {r['ratio']} | {per} | "
                     f"{r['mean']:.3f} ± {r['sd']:.3f} |")
    lines.append('')
    return lines


def _oracle_gap_markdown(rows) -> list:
    if not rows:
        return []
    lines = ['## ours_beta − oracle_beta by reader (exploratory)', '',
             "oracle_beta's labels come from the primary reader on these documents: its lead on the other readers is "
             'the part of the gap that transfers.', '',
             '| reader | source | ratio | n | Δ F1 [95% CI] |', '|---|---|---|---|---|']
    for r in rows:
        lo, hi = r['ci95']
        ci = f" [{lo:+.3f}, {hi:+.3f}]" if lo is not None else ''
        lines.append(f"| {r['reader']} | {r['source']} | {r['ratio']} | {r['n']} | {r['diff']:+.3f}{ci} |")
    return lines + ['']


def _diagnostics_markdown(diag) -> list:
    if not diag:
        return []
    num = lambda v: '' if v is None else f"{v:.3f}"  # noqa: E731
    lines = ['## Diagnostics (exploratory): answer coverage, bridge effect, selection overlap', '',
             '| source | ratio | arm | answer in context | every gold chunk kept | chunks touched | budget used | '
             'F1 answer in | F1 answer out |', '|---|---|---|---|---|---|---|---|---|']
    for r in diag.get('coverage', []):
        touched = '' if r['touched'] is None else f"{r['touched']:.2f}"
        lines.append(f"| {r['source']} | {r['ratio']} | {r['arm']} | {r['answer_in_context']:.3f} | {num(r['all_gold'])} | "
                     f"{touched} | {r['budget_use']:.2f} | {num(r.get('f1_answer_in'))} | {num(r.get('f1_answer_out'))} |")
    if diag.get('bridge'):
        lines += ['', 'Bridge effect (primary reader, chunk arms pooled, documents where an answer chunk is kept):', '',
                  '| source | ratio | F1, every gold chunk kept (n) | F1, some gold chunk missing (n) |', '|---|---|---|---|']
        for b in diag['bridge']:
            lines.append(f"| {b['source']} | {b['ratio']} | {num(b['f1_all_gold'])} ({b['n_all_gold']}) | "
                         f"{num(b['f1_some_gold_missing'])} ({b['n_some_gold_missing']}) |")
    if diag.get('overlap'):
        cols = sorted({k for r in diag['overlap'] for k in r} - {'source', 'ratio', 'arm'})
        lines += ['', 'Selection overlap (mean Jaccard of kept chunks):', '',
                  '| source | ratio | arm | ' + ' | '.join(cols) + ' |', '|---|---|---|' + '---|' * len(cols)]
        for r in diag['overlap']:
            lines.append(f"| {r['source']} | {r['ratio']} | {r['arm']} | " + ' | '.join(num(r.get(c)) for c in cols) + ' |')
    return lines + ['']


def _label_quality_markdown(rows) -> list:
    if not rows:
        return []
    num = lambda v, f='.3f': '' if v is None or v != v else format(v, f)  # noqa: E731
    lines = ['## Label quality (Stage A, this run\'s fits)', '',
             '| reader | target | set | docs (informative) | α | full F1 | cv R² | gold recall@g | gold MRR | '
             'position R² | cross-reader ρ |', '|---|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        rho = '; '.join(f"{k}: {v:.2f}" for k, v in (r['cross_reader_spearman'] or {}).items())
        lines.append(f"| {r['reader']} | {r['target']} | {r['set']} | {r['n_docs']} ({r['n_informative']}) | "
                     f"{num(r['alpha'], 'g')} | {num(r['mean_full_f1'])} | {num(r['mean_cv_r2'])} | "
                     f"{num(r['gold_recall_at_g'])} | {num(r['gold_mrr'])} | {num(r['position_r2'])} | {rho} |")
    lines.append('')
    return lines


def _cost_markdown(cost) -> list:
    if not cost:
        return []
    lines = ['## Cost (RQ1): Stage A labels vs one selection pass', '',
             '| reader | label set | docs | reader calls / doc | s / doc | total h |', '|---|---|---|---|---|---|']
    for c in cost['labels']:
        lines.append(f"| {c['reader']} | {c['set']} | {c['n_docs']} | {c['calls_per_doc']:.0f} | "
                     f"{c['seconds_per_doc']:.1f} | {c['total_hours']:.2f} |")
    lines += ['', '| source | arm | select ms / doc |', '|---|---|---|']
    for c in cost['select_ms']:
        lines.append(f"| {c['source']} | {c['arm']} | {c['select_ms']:.0f} |")
    lines.append('')
    return lines


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
    s.add_argument('--offset', type=int, default=0,
                   help="skip the first N documents per source (hash order), e.g. a replication on unseen documents")
    s.add_argument('--haystack-chars', type=int, default=30000)
    s.add_argument('--distractors', choices=['random', 'hard'], default='random')
    s.add_argument('--multihop-pad-chars', type=int, default=0)
    s.add_argument('--multihop-units', choices=['paragraph', 'sentence'], default='paragraph')
    s.add_argument('--arms', required=True, help="comma list; 'label=spec' to name an arm, e.g. beta=pruner:models/x")
    s.add_argument('--ratios', default='4,8')
    s.add_argument('--restrict-sources', default='',
                   help="select only the documents of these sources (comma list); the document set is unchanged")
    s.add_argument('--budget-tokenizer', required=True, help="reference tokenizer for budgets (usually the first reader)")
    s.add_argument('--oracle-beta-dir', default=None,
                   help="Stage A fit dir(s) of the evaluated documents (comma list) for the oracle_beta arm")
    s.add_argument('--device', default='cuda')
    s.add_argument('--shard', type=int, default=0)
    s.add_argument('--num-shards', type=int, default=1)
    s.add_argument('--out-dir', required=True)
    s.add_argument('--flush-every', type=int, default=25, help="append selections every N documents")
    s.add_argument('--retry-failed', action='store_true',
                   help="retry (document, arm) pairs recorded in select_failures_shard*.jsonl")
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
    r.add_argument('--primary-reader', default=None,
                   help="reader tag for H1/H2 (the label reader of ours_beta); default: every reader")
    r.add_argument('--heldout-readers', default='', help="comma list of reader tags that produced no labels (H3-heldout)")
    r.add_argument('--equiv-margin', type=float, default=0.02, help="H2b equivalence margin (token F1)")
    r.add_argument('--oracle-margin', type=float, default=0.05, help="H1 non-inferiority margin to oracle_beta")
    r.add_argument('--min-upgrade-gap', type=float, default=0.05,
                   help="minimum full-context weak->strong F1 gap for a retention ratio to be read (H3)")
    r.add_argument('--budget-tolerance', type=float, default=0.05,
                   help="a selection is over budget when kept_tokens > budget * (1 + this)")
    r.add_argument('--labels-dir', default=None,
                   help="labels root(s), comma list (raw/<reader>/...), for the RQ1 cost table")
    r.add_argument('--fit-dir', default=None, help="this run's fit dir (<reader|ensemble>/<target>/<set>/summary.json) "
                                                    "for the label-quality table")
    r.add_argument('--confirm-ratios', default='',
                   help="ratios the confirmatory families test (the pre-registered ones, e.g. 4,8); default: every ratio")
    r.add_argument('--prereg', default=None,
                   help="JSON of families fixed before the run (docs/prereg_*.json), tested like the confirmatory ones")
    r.add_argument('--no-diagnostics', action='store_true',
                   help="skip the exploratory diagnostics read from the selection files")
    r.set_defaults(func=cmd_report)

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
