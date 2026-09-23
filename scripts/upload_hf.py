#!/usr/bin/env python3
"""Upload a finished run to the HuggingFace Hub.

  - every pruner in --models-dir  -> one model repo each:  <namespace>/<prefix>-<run>-<pruner>
  - the evaluation directory      -> one dataset repo:     <namespace>/<prefix>-<run>-eval
        eval/     documents, selections, answers, report.{md,json}
        labels/   every summary.json (label-quality diagnostics); with --include-labels also the
                  per-document labels (fit/) and raw reader measurements (raw/)

Repos are created if missing (private by default) and re-uploading only commits changed files.

    python scripts/upload_hf.py --check                       # token present and allowed to write?
    python scripts/upload_hf.py --run-name pilot --models-dir runs/pilot/models \\
        --eval-dir runs/pilot/results/eval_test --labels-dir runs/labels --include-labels
    python scripts/upload_hf.py ... --dry-run                 # list repos and files, upload nothing

The token comes from HF_TOKEN (or a cached `huggingface-cli login`). --namespace defaults to the
token's own user; pass an organization name to upload there instead.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys


def _api():
    from huggingface_hub import HfApi
    return HfApi()


def check_token(namespace: str | None) -> str:
    """Owner name of the token; exits with a readable message if the token cannot write."""
    try:
        info = _api().whoami()
    except Exception as exc:
        sys.exit(f"HuggingFace token missing or invalid ({type(exc).__name__}: {exc}). "
                 f"Put HF_TOKEN=hf_... (a WRITE token) in ./.env")
    role = (info.get('auth', {}).get('accessToken', {}) or {}).get('role')
    user = info.get('name')
    orgs = [o.get('name') for o in info.get('orgs', [])]
    if role == 'read':
        sys.exit(f"HF token of '{user}' is read-only; create a token with write access")
    if namespace and namespace != user and namespace not in orgs:
        sys.exit(f"HF token of '{user}' is not a member of '{namespace}' (orgs: {orgs or 'none'})")
    note = ' (fine-grained token: make sure it may write to that namespace)' if role == 'fineGrained' else ''
    print(f"HF token ok: user '{user}', role {role}{note}; uploads go to '{namespace or user}'")
    return user


def _model_card(repo_id: str, pruner_dir: str, run: str) -> str:
    cfg, log = {}, {}
    try:
        with open(os.path.join(pruner_dir, 'pruner_config.json'), encoding='utf-8') as f:
            cfg = json.load(f)
        with open(os.path.join(pruner_dir, 'train_log.json'), encoding='utf-8') as f:
            log = json.load(f)
    except OSError:
        pass
    best = cfg.get('best_dev') or {}
    rows = '\n'.join(f"| {k} | {v:.4f} |" for k, v in best.items() if isinstance(v, (int, float)))
    return f"""---
library_name: transformers
base_model: {cfg.get('backbone', 'BAAI/bge-reranker-v2-m3')}
tags: [context-compression, chunk-pruning, ttcompress]
---

# {repo_id}

Chunk pruner from the ttcompress run `{run}`: a cross-encoder that reads a question and a document's
chunks in one pass and scores every chunk; the highest-scoring chunks that fit a token budget are kept.

| setting | value |
|---|---|
| label source | {cfg.get('label_source')} |
| position adjustment | {cfg.get('position_adjust')} |
| training documents | {log.get('n_train')} |
| max input tokens | {cfg.get('max_len')} |
| best epoch | {cfg.get('best_epoch')} (by dev {cfg.get('select_metric')}) |

Dev metrics at the best epoch:

| metric | value |
|---|---|
{rows or '| – | – |'}

Load with the ttcompress code:

```python
from ttcompress.pruner import PrunerScorer
scorer = PrunerScorer('{repo_id}')   # or a local snapshot path
scores = scorer.score_chunks(question, chunks)
```

`chunk_head.pt` holds the scoring head; the encoder weights are a standard transformers checkpoint.
"""


def _dataset_card(repo_id: str, eval_dir: str, run: str, include_labels: bool) -> str:
    report = ''
    path = os.path.join(eval_dir, 'report.md')
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            report = f.read()
    labels = ('`labels/` holds the label-quality summaries, the per-document labels (`fit/`) and the raw '
              'reader measurements (`raw/`).') if include_labels else \
             '`labels/` holds the label-quality summaries (`summary.json` per reader, target and split).'
    return f"""---
tags: [context-compression, ttcompress]
---

# {repo_id}

Evaluation outputs of the ttcompress run `{run}`.

- `eval/documents_shard*.jsonl` – the test documents (question, chunks, gold chunks)
- `eval/selections_shard*.jsonl` – what every compression arm kept, per document and ratio
- `eval/answers_<reader>_shard*.jsonl` – reader answers with EM / F1 per selection
- `eval/report.md`, `eval/report.json` – aggregated results
- {labels}

{report}
"""


def upload(args, owner: str) -> None:
    from huggingface_hub import HfApi
    api = HfApi()
    ns = args.namespace or owner
    commit = f"ttcompress run '{args.run_name}'"

    pruners = sorted(d for d in glob.glob(os.path.join(args.models_dir, '*'))
                     if os.path.isfile(os.path.join(d, 'pruner_config.json')))
    if not pruners:
        print(f"[WARN] no trained pruners (pruner_config.json) under {args.models_dir}")
    for d in pruners:
        repo_id = f"{ns}/{args.prefix}-{args.run_name}-{os.path.basename(d)}".replace('_', '-')
        files = sorted(os.listdir(d))
        print(f"model   {repo_id}  <- {d} ({len(files)} files)")
        if args.dry_run:
            continue
        api.create_repo(repo_id, repo_type='model', private=args.private, exist_ok=True)
        with open(os.path.join(d, 'README.md'), 'w', encoding='utf-8') as f:
            f.write(_model_card(repo_id, d, args.run_name))
        api.upload_folder(repo_id=repo_id, repo_type='model', folder_path=d, commit_message=commit,
                          ignore_patterns=['*.tmp'])

    repo_id = f"{ns}/{args.prefix}-{args.run_name}-eval".replace('_', '-')
    eval_files = sorted(glob.glob(os.path.join(args.eval_dir, '*')))
    print(f"dataset {repo_id}  <- {args.eval_dir} ({len(eval_files)} files)"
          + (f" + {args.labels_dir}" if args.labels_dir else ''))
    if args.dry_run:
        return
    if not eval_files:
        print(f"[WARN] {args.eval_dir} is empty; uploading labels only")
    api.create_repo(repo_id, repo_type='dataset', private=args.private, exist_ok=True)
    if eval_files:
        api.upload_folder(repo_id=repo_id, repo_type='dataset', folder_path=args.eval_dir, path_in_repo='eval',
                          commit_message=f"{commit}: eval", ignore_patterns=['*.tmp'])
    if args.labels_dir and os.path.isdir(args.labels_dir):
        patterns = None if args.include_labels else ['**/summary.json', '**/measure_config.json']
        api.upload_folder(repo_id=repo_id, repo_type='dataset', folder_path=args.labels_dir, path_in_repo='labels',
                          commit_message=f"{commit}: labels", allow_patterns=patterns, ignore_patterns=['*.tmp'])
    card = _dataset_card(repo_id, args.eval_dir, args.run_name, args.include_labels).encode('utf-8')
    api.upload_file(repo_id=repo_id, repo_type='dataset', path_or_fileobj=card, path_in_repo='README.md',
                    commit_message=f"{commit}: card")
    print(f"done: https://huggingface.co/datasets/{repo_id}")


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--check', action='store_true', help="only verify the token can write, then exit")
    ap.add_argument('--namespace', default=None, help="HF user or organization (default: the token's user)")
    ap.add_argument('--prefix', default='ttcompress')
    ap.add_argument('--run-name', default='run')
    ap.add_argument('--models-dir', default='models')
    ap.add_argument('--eval-dir', default='results/eval_test')
    ap.add_argument('--labels-dir', default=None)
    ap.add_argument('--include-labels', action='store_true', help="also upload per-document labels and raw measurements")
    ap.add_argument('--private', dest='private', action='store_true', default=True)
    ap.add_argument('--public', dest='private', action='store_false')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    if args.dry_run and not args.check:
        upload(args, owner=args.namespace or '<token-user>')
        return
    owner = check_token(args.namespace)
    if not args.check:
        upload(args, owner)


if __name__ == '__main__':
    main()
