#!/usr/bin/env python3
"""OUTCOME_SUPERVISED_RELEVANCE_SPEC.md §1 (checklist item, mandatory before
any other coding): check for train/test content leakage across every source
Stage A/B/C and Stage 6 actually use.

v2: originally checked UIT-ViQuAD against vncompress's internal
wikipedia_vi_raw.json (the corpus the now-dropped vcc_bench_v2.json was built
from -- see ttcompress/vietnamese_public_test.py for why that test set was
replaced with public data). Nothing reads that corpus anymore, so this now
checks the boundary that actually matters for the current pipeline instead:

  1. UIT-ViQuAD TRAIN (Stage A's training corpus) vs. UIT-ViQuAD's own
     reserved test portion (ttcompress.uit_viquad.split_dev_tune_test --
     carved out of the official Dev split, since the official Test split has
     no usable answers at all; see that function's docstring) -- should be
     empty by construction (disjoint official/derived splits), checked
     anyway rather than assumed, per WAVE4_REPORT.md §4.3 discipline.
  2. UIT-ViQuAD TRAIN vs. XQuAD-vi (a wholly different dataset/corpus, but
     both are Vietnamese Wikipedia-adjacent content, so a real check, not a
     formality).

Two checks per pair, both logged:
  a. Title overlap: same Wikipedia article referenced by both (XQuAD-vi has
     no title field, so this only applies to the UIT-ViQuAD-vs-UIT-ViQuAD pair).
  b. Content overlap: any context paragraph appearing verbatim (or
     near-verbatim, via a normalized-prefix check) in the other side.

    python scripts/check_uit_viquad_overlap.py --out results/train_test_overlap.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Vietnamese titles/content in the overlap report can crash `print` on a
# Windows console stuck on cp1252 -- reconfigure stdout to UTF-8 so a real
# NONEMPTY finding (this script's whole reason to exist) is never lost to an
# unrelated encoding error after the JSON report has already been written.
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except AttributeError:
    pass


def _normalize(text: str) -> str:
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFC', text).strip().lower())


def _check_pair(name: str, train_samples, other_samples, other_name: str, prefix_len: int, check_titles: bool) -> dict:
    train_prefixes = {_normalize(s.context)[:prefix_len] for s in train_samples}
    other_prefixes = {_normalize(s.context)[:prefix_len] for s in other_samples}
    prefix_overlap = sorted(train_prefixes & other_prefixes)

    result = {
        'pair': name, 'train_n': len(train_samples), f'{other_name}_n': len(other_samples),
        'content_prefix_overlap_count': len(prefix_overlap),
        'content_prefix_overlap_examples': prefix_overlap[:10],
    }
    if check_titles:
        train_titles = {_normalize(s.title) for s in train_samples if s.title}
        other_titles = {_normalize(s.title) for s in other_samples if s.title}
        title_overlap = sorted(train_titles & other_titles)
        result['title_overlap_count'] = len(title_overlap)
        result['title_overlap'] = title_overlap
    result['verdict'] = 'EMPTY' if result['content_prefix_overlap_count'] == 0 and result.get('title_overlap_count', 0) == 0 else 'NONEMPTY -- REVIEW BEFORE PROCEEDING'
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--prefix-len', type=int, default=120,
                     help="chars of each paragraph's normalized prefix used for the content-overlap check")
    ap.add_argument('--out', default='results/train_test_overlap.json')
    args = ap.parse_args()

    from ttcompress.uit_viquad import load_split, split_dev_tune_test
    from ttcompress.xquad_vi import load_all as load_xquad_vi

    train = load_split('train')
    _tune, uit_test = split_dev_tune_test()
    xquad = load_xquad_vi()
    print(f"UIT-ViQuAD train: {len(train)} samples | reserved test: {len(uit_test)} | XQuAD-vi: {len(xquad)}")

    checks = [
        _check_pair('uit_train_vs_uit_test', train, uit_test, 'uit_test', args.prefix_len, check_titles=True),
        _check_pair('uit_train_vs_xquad_vi', train, xquad, 'xquad_vi', args.prefix_len, check_titles=False),
    ]

    report = {'checks': checks, 'verdict': 'EMPTY' if all(c['verdict'] == 'EMPTY' for c in checks) else 'NONEMPTY -- REVIEW BEFORE PROCEEDING'}
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    for c in checks:
        print(f"\n{c['pair']}: content-prefix overlap = {c['content_prefix_overlap_count']}"
              + (f", title overlap = {c['title_overlap_count']} {c['title_overlap']}" if 'title_overlap_count' in c else ""))
        print(f"  verdict: {c['verdict']}")
    print(f"\nOverall verdict: {report['verdict']}")
    print(f"Logged to {args.out}")


if __name__ == '__main__':
    main()
