#!/usr/bin/env python3
"""Stage B (OUTCOME_SUPERVISED_RELEVANCE_SPEC.md §3): beta_c = gamma_0 +
gamma_1 * g(i_c, L) + residual_c -- separates the position-signal component
PCS already captures out of Stage A's per-chunk contributions, so Stage C
can train on either the raw contribution or the position-stripped residual.
CPU-only, no reader/GPU needed.

    python decompose_labels.py --in-dir outcome_labels_raw/train_pilot --out-dir training_labels/train_pilot
"""
from __future__ import annotations

import argparse
import glob
import os

from ttcompress.outcome_labels import load_document_labels
from ttcompress.position_decompose import decompose_document, save_training_labels


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--in-dir', required=True, help="DocumentLabels dir from generate_labels.py fit")
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.in_dir, '*.jsonl')))
    if not paths:
        raise SystemExit(f"no .jsonl files in {args.in_dir} -- run generate_labels.py fit first")

    gammas = []
    for path in paths:
        labels = load_document_labels(path)
        training = decompose_document(labels)
        save_training_labels(training, args.out_dir)
        gammas.append((training.gamma_0, training.gamma_1))

    mean_gamma1 = sum(g1 for _, g1 in gammas) / len(gammas)
    print(f"Decomposed {len(paths)} documents. Mean gamma_1 (position-signal weight) = {mean_gamma1:.4f} "
          f"-- how much of beta_c the U-shaped position score alone explains, averaged across documents.")
    print(f"Saved TrainingLabels (raw_label, residual_label) to {args.out_dir}/")


if __name__ == '__main__':
    main()
