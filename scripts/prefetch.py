#!/usr/bin/env python3
"""Download every dataset and model a run needs, once, in ONE process,
before run_pipeline.sh fans out to one process per GPU.

Why: N processes hitting an empty HF cache at the same time race on the
same downloads (and on `datasets`' cache build), and a gated model without
access should fail in the first minute, not hours into
label generation.

    python scripts/prefetch.py --sources uit_viquad,xquad_vi,vimqa,hotpotqa,2wiki \\
        --models Qwen/Qwen3-8B,Qwen/Qwen3-1.7B,BAAI/bge-reranker-v2-m3,BAAI/bge-m3
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ttcompress.sources import AVAILABLE_SPLITS, load_documents, parse_source_list  # noqa: E402

WEIGHT_PATTERNS = ['*.json', '*.safetensors', '*.model', '*.txt', '*.py', '*.tiktoken', 'tokenizer*', '*.jinja']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sources', default='all')
    ap.add_argument('--models', default='', help="comma list of HF model ids")
    args = ap.parse_args()

    failures = []
    for src in parse_source_list(args.sources):
        for split in AVAILABLE_SPLITS[src]:
            t0 = time.time()
            try:
                docs = load_documents(src, split, n=2)
                print(f"dataset {src}/{split}: ok ({len(docs)} sample docs, {time.time() - t0:.0f}s)", flush=True)
            except Exception as exc:
                failures.append(f"dataset {src}/{split}: {type(exc).__name__}: {exc}")
                print(failures[-1], flush=True)

    from huggingface_hub import snapshot_download
    for model in [m.strip() for m in args.models.split(',') if m.strip()]:
        t0 = time.time()
        try:
            path = snapshot_download(model, allow_patterns=WEIGHT_PATTERNS)
            print(f"model {model}: ok -> {path} ({time.time() - t0:.0f}s)", flush=True)
        except Exception as exc:
            hint = " (gated model: accept its license on huggingface.co and export HF_TOKEN)" \
                if 'gated' in str(exc).lower() or '401' in str(exc) or '403' in str(exc) else ''
            failures.append(f"model {model}: {type(exc).__name__}: {exc}{hint}")
            print(failures[-1], flush=True)

    if failures:
        print(f"\n{len(failures)} prefetch failure(s):\n  " + "\n  ".join(failures))
        sys.exit(1)
    print("\nprefetch complete")


if __name__ == '__main__':
    main()
