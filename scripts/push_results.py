#!/usr/bin/env python3
"""Push run artifacts under results/ to a Hugging Face Dataset repo.

Only run artifacts -- no model weights. PCS never trains anything
(PCS_METHOD_SPEC.md hard constraint #1: bolt-on only); train.py's output is a
chosen lambda* plus a dev sweep curve, not a checkpoint, so there is nothing
to push to a Model repo. What gets pushed here:

  - lambda_sweep_dev.json   the lambda curve + chosen lambda* (train.py)
  - dev_test_split.json     the exact dev/test samples used (train.py) --
                             kept so the official_bench numbers are auditable
                             against the precise rows they were computed on
  - official_bench.json     point estimate + 95% CI per arm/ratio (evaluate.py)

    python scripts/push_results.py --repo-id you/ttcompress-pcs-results

Auth: --token, else the HF_TOKEN environment variable, else whatever
`huggingface-cli login` already cached locally.
"""
from __future__ import annotations

import argparse
import os


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--repo-id', required=True, help="e.g. your-username/ttcompress-pcs-results")
    ap.add_argument('--results-dir', default='results')
    ap.add_argument('--private', action='store_true', default=True, help="default: private repo")
    ap.add_argument('--public', dest='private', action='store_false')
    ap.add_argument('--token', default=None, help="defaults to HF_TOKEN env var, then any cached huggingface-cli login")
    ap.add_argument('--commit-message', default='Update PCS run results')
    args = ap.parse_args()

    from huggingface_hub import HfApi

    token = args.token or os.environ.get('HF_TOKEN')
    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type='dataset', private=args.private, exist_ok=True)

    files = sorted(f for f in os.listdir(args.results_dir) if f.endswith('.json'))
    if not files:
        raise SystemExit(f"No .json files in {args.results_dir}/ -- run train.py/evaluate.py first")

    for fname in files:
        api.upload_file(
            path_or_fileobj=os.path.join(args.results_dir, fname), path_in_repo=fname,
            repo_id=args.repo_id, repo_type='dataset', commit_message=args.commit_message,
        )
        print(f"  pushed {fname}")

    visibility = 'private' if args.private else 'public'
    print(f"Pushed {len(files)} file(s) to {visibility} dataset "
          f"https://huggingface.co/datasets/{args.repo_id}")


if __name__ == '__main__':
    main()
