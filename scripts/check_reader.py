"""Sanity check of a new evaluation reader before a long answer stage.

Answers the first --n test documents of every source with the FULL context (the `full` arm) and prints, per
source, the share of empty answers, the mean token F1 and a few answers. A reader whose chat template or
stop behaviour does not fit the prompts shows up here in minutes instead of hours into the answer stage
(SEA-LION once answered 57% of the Vietnamese prompts with an empty string).

    python scripts/check_reader.py --eval-dir $R/results/eval_test --reader-model google/gemma-3-27b-it --tp 2

Exit status 3 when a source fails the gate (--max-empty, --min-f1), so a runbook can stop before the long answer
stage; --out writes the per-source numbers as JSON.
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ttcompress.data import CHUNK_SEP, load_documents  # noqa: E402
from ttcompress.metrics import token_f1  # noqa: E402
from ttcompress.reader import MAX_NEW_TOKENS, load_reader  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--eval-dir', required=True, help="holds documents_shard*.jsonl of the evaluation run")
    ap.add_argument('--reader-model', required=True)
    ap.add_argument('--backend', choices=['hf', 'vllm'], default='vllm')
    ap.add_argument('--tp', type=int, default=1)
    ap.add_argument('--max-model-len', type=int, default=16384)
    ap.add_argument('--gpu-memory-utilization', type=float, default=0.9)
    ap.add_argument('--n', type=int, default=30, help="documents per source")
    ap.add_argument('--show', type=int, default=3, help="answers printed per source")
    ap.add_argument('--max-empty', type=float, default=0.05, help="gate: largest share of empty answers per source")
    ap.add_argument('--min-f1', type=float, default=0.30, help="gate: smallest full-context F1 per source")
    ap.add_argument('--out', default=None, help="write the per-source numbers to this JSON file")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.eval_dir, 'documents_shard*.jsonl')))
    if not paths:
        sys.exit(f"no documents_shard*.jsonl in {args.eval_dir}")
    by_source = defaultdict(list)
    for path in paths:
        for doc in load_documents(path):
            if len(by_source[doc.source]) < args.n:
                by_source[doc.source].append(doc)

    reader = load_reader(args.reader_model, args.backend, max_model_len=args.max_model_len,
                         tensor_parallel_size=args.tp, gpu_memory_utilization=args.gpu_memory_utilization)
    print(f"reader {args.reader_model}: chat template = {reader.is_chat}")
    results, failed = {}, []
    for source, docs in sorted(by_source.items()):
        prompts = [reader.build_prompt(CHUNK_SEP.join(d.chunks), d.question, d.language) for d in docs]
        answers = reader.generate(prompts, MAX_NEW_TOKENS[docs[0].hop])
        f1 = sum(token_f1(a, d.answers) for a, d in zip(answers, docs)) / len(docs)
        empty = sum(not a.strip() for a in answers) / len(answers)
        samples = [{'gold': d.answers[0], 'answer': a} for a, d in list(zip(answers, docs))[:args.show]]
        print(f"\n== {source}: n={len(docs)}  empty={empty:.0%}  full-context F1={f1:.3f}")
        for s in samples:
            print(f"   gold={s['gold']!r:.60}  answer={s['answer']!r:.80}")
        results[source] = {'n': len(docs), 'empty': empty, 'f1': f1, 'samples': samples}
        if empty > args.max_empty or f1 < args.min_f1:
            failed.append(source)
    print(f"\ntruncated prompts: {reader.n_truncated}")
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump({'reader': args.reader_model, 'chat': reader.is_chat, 'truncated': reader.n_truncated,
                       'gate': {'max_empty': args.max_empty, 'min_f1': args.min_f1}, 'failed': failed,
                       'sources': results}, f, ensure_ascii=False, indent=1)
    if failed:
        print(f"GATE FAILED on {', '.join(failed)} (empty > {args.max_empty:.0%} or F1 < {args.min_f1})")
        sys.exit(3)
    print("gate passed")


if __name__ == '__main__':
    main()
