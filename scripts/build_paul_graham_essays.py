#!/usr/bin/env python3
"""Assemble data/paul_graham_essays.json from a directory of downloaded .txt
essay files. Called by scripts/fetch_data.sh -- not meant to be run standalone
against anything other than that directory (see fetch_data.sh for where the
.txt files come from: the same URL list NVIDIA/RULER's own data generator
uses, see data/SOURCES.md).
"""
from __future__ import annotations

import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--essays-dir', required=True, help="directory of downloaded <name>.txt essay files")
    ap.add_argument('--out', required=True, help="output path for the combined {name: text} JSON")
    args = ap.parse_args()

    essays = {}
    for filename in sorted(os.listdir(args.essays_dir)):
        if not filename.endswith('.txt'):
            continue
        path = os.path.join(args.essays_dir, filename)
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            essays[filename[:-len('.txt')]] = f.read()

    if not essays:
        raise SystemExit(f"no .txt files found in {args.essays_dir} -- fetch_data.sh's download step failed silently")

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(essays, f, ensure_ascii=False)

    total_chars = sum(len(v) for v in essays.values())
    print(f"Wrote {len(essays)} essays ({total_chars} chars total) to {args.out}")


if __name__ == '__main__':
    main()
