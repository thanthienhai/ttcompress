#!/usr/bin/env python3
"""Stage A (OUTCOME_SUPERVISED_RELEVANCE_SPEC.md §2): generate (mask, token_f1)
outcome labels on needle+haystack documents, then fit per-chunk contributions
via ridge regression. Multilingual, public-only upgrade: `--sources` picks
any mix of UIT-ViQuAD 2.0 + XQuAD-vi (Vietnamese) and PCS_METHOD_SPEC.md's
three English sources (LongBench passage_retrieval_en, RULER niah_single_1,
classic Kamradt NIAH) -- see ttcompress/multilingual_sources.py for how
documents from each are built uniformly and where each source's data comes
from (always a reserved "dev"/"tune" pool, never a source's reserved test
portion -- see that module and ttcompress/vietnamese_public_test.py).

Two subcommands, split because "measure" is GPU-bound (calls the reader once
per mask per document) and "fit" is CPU-only (ridge + bootstrap CI) -- the
same measure/fit separation as this project's train.py vs evaluate.py, so
re-running the CPU-only math never re-pays the GPU cost.

    # 1. measure -- pilot first (spec §7 is explicit: run N=10, K=15 and read
    #    the real per-mask timing before choosing a bigger N/K; do this for
    #    both splits, dev is needed for alpha selection in `fit`)
    python generate_labels.py measure --split train --sources all --reader-model Qwen/Qwen3-8B \\
        --n 10 --k 15 --out-dir outcome_raw/train_pilot
    python generate_labels.py measure --split dev --sources all --reader-model Qwen/Qwen3-8B \\
        --n 10 --k 15 --out-dir outcome_raw/dev_pilot

    # 2. fit -- CV-select alpha on the Dev measure output, fit Train with it
    python generate_labels.py fit --in-dir outcome_raw/train_pilot \\
        --select-alpha-from outcome_raw/dev_pilot --out-dir outcome_labels_raw/train_pilot

Chunk = one paragraph/sentence of the constructed document (§2.1, resolved:
same token-level E6 as PCS_METHOD_SPEC.md §4 -- chunk is only the
label-estimation unit, broadcast to tokens in Stage C). UIT-ViQuAD/XQuAD-vi
documents replicate the same haystack-construction algorithm the existing
internal needle_in_haystack task uses (confirmed by reading
vncompress/scripts/build_vcc_bench_v2.py), so training here and the final
public-Vietnamese-test evaluation (Stage 6) are comparably hard; the three
English sources reuse their own PCS_METHOD_SPEC.md recipes (RULER/Kamradt
sentence-level, LongBench's own paragraph numbering).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

from ttcompress.metrics import token_f1_one
from ttcompress.multilingual_sources import ALL_SOURCES, select_documents
from ttcompress.outcome_labels import (
    DocumentLabels,
    MaskOutcomeRecord,
    bootstrap_beta_ci,
    estimate_chunk_contributions,
    generate_masks,
    load_mask_outcomes,
    low_confidence_chunks,
    measure_outcomes,
    save_document_labels,
    save_mask_outcomes,
    select_alpha,
)
from ttcompress.reader import default_max_new_tokens, generate_answer, load_reader


def cmd_measure(args):
    sources = list(ALL_SOURCES) if args.sources == 'all' else args.sources.split(',')
    exclude_titles = []
    if args.overlap_report:
        with open(args.overlap_report, 'r', encoding='utf-8') as f:
            overlap = json.load(f)
        # scripts/check_uit_viquad_overlap.py's report is {'checks': [...]}
        # -- pull title_overlap from the uit_train_vs_uit_test check, the
        # only one that can name UIT-ViQuAD titles to exclude from training.
        for check in overlap.get('checks', []):
            if check.get('pair') == 'uit_train_vs_uit_test':
                exclude_titles = check.get('title_overlap', [])
                break
        print(f"--overlap-report {args.overlap_report}: will exclude UIT-ViQuAD titles {exclude_titles}")

    documents = select_documents(sources, uit_viquad_split=args.split, n=args.n, seed=args.seed,
                                  exclude_uit_viquad_titles=exclude_titles)
    from collections import Counter
    print(f"Sources: {sources}. Built {len(documents)} documents -- "
          f"{dict(Counter(d.metadata.get('dataset_source') for d in documents))}")
    chunk_counts = [d.num_chunks for d in documents]
    print(f"Chunks/doc: min={min(chunk_counts)} mean={sum(chunk_counts) / len(chunk_counts):.1f} "
          f"max={max(chunk_counts)} (spec §7's missing 'chunks/document' number -- from this run's real "
          f"documents, not guessed)")

    model, tokenizer = load_reader(args.reader_model, device=args.device)

    def make_generate_fn(prompt_kind):
        def generate_fn(context_text, query):
            input_ids = tokenizer.encode(context_text, add_special_tokens=False)
            return generate_answer(
                model, tokenizer, input_ids, query,
                max_new_tokens=args.max_new_tokens or default_max_new_tokens(prompt_kind), prompt_kind=prompt_kind,
            )
        return generate_fn

    os.makedirs(args.out_dir, exist_ok=True)
    per_mask_times = []
    for i, doc in enumerate(documents):
        prompt_kind = doc.metadata.get('dataset_source', 'uit_viquad')
        generate_fn = make_generate_fn(prompt_kind)
        masks = generate_masks(doc.num_chunks, args.k, seed=args.seed + i)
        start = time.time()
        outcomes = measure_outcomes(doc.chunks, masks, doc.question, doc.answer_text, generate_fn, token_f1_one)
        elapsed = time.time() - start
        per_mask_times.append(elapsed / len(masks))
        # doc.doc_id is a CLUSTER label (RULER/Kamradt legitimately reuse one
        # per essay across several distinct documents, for PCS's own bootstrap
        # CI) -- not guaranteed unique, but save_mask_outcomes uses it as a
        # filename. Disambiguate with the loop index so two documents sharing
        # a doc_id never silently overwrite each other's measured labels (the
        # original is kept in metadata for traceability).
        save_id = f"{i:06d}_{doc.doc_id}"
        record = MaskOutcomeRecord(
            doc_id=save_id, chunks=doc.chunks, positions=list(range(doc.num_chunks)),
            question=doc.question, reference_answer=doc.answer_text,
            masks=masks, outcomes=outcomes, metadata={**doc.metadata, 'cluster_id': doc.doc_id},
        )
        save_mask_outcomes(record, args.out_dir)
        print(f"  [{i + 1}/{len(documents)}] {doc.doc_id}: {doc.num_chunks} chunks, {len(masks)} masks, "
              f"mean_token_f1={sum(outcomes) / len(outcomes):.3f}, {elapsed / len(masks) * 1000:.0f} ms/mask")

    mean_ms = sum(per_mask_times) / len(per_mask_times) * 1000
    print(f"\nMean time per (mask, generate) call: {mean_ms:.0f} ms -- spec §7's other missing cost number. "
          f"Use this + the chunks/document figure above to size a full-scale N/K before running beyond this pilot.")
    print(f"Saved {len(documents)} MaskOutcomeRecord files to {args.out_dir}/")


def cmd_fit(args):
    paths = sorted(glob.glob(os.path.join(args.in_dir, '*.jsonl')))
    if not paths:
        raise SystemExit(f"no .jsonl files in {args.in_dir} -- run `measure` first")
    records = [load_mask_outcomes(p) for p in paths]

    if args.alpha is not None:
        alpha = args.alpha
    else:
        if not args.select_alpha_from:
            raise SystemExit("pass --alpha, or --select-alpha-from a Dev-split `measure` output dir (spec §2.5: "
                              "alpha is chosen by CV on the official Dev split, not per-document)")
        dev_paths = sorted(glob.glob(os.path.join(args.select_alpha_from, '*.jsonl')))
        if not dev_paths:
            raise SystemExit(f"no .jsonl files in {args.select_alpha_from}")
        dev_records = [load_mask_outcomes(p) for p in dev_paths]
        dev_documents = [(r.masks, r.outcomes) for r in dev_records]
        alpha_grid = tuple(float(a) for a in args.alpha_grid.split(','))
        alpha = select_alpha(dev_documents, alpha_grid=alpha_grid, seed=args.seed)
        print(f"Selected alpha={alpha} via CV over {len(dev_records)} Dev documents (grid={alpha_grid})")

    os.makedirs(args.out_dir, exist_ok=True)
    n_low_conf_docs = 0
    for record in records:
        beta = estimate_chunk_contributions(record.masks, record.outcomes, alpha)
        _beta_point, ci_lo, ci_hi = bootstrap_beta_ci(record.masks, record.outcomes, alpha,
                                                        n_boot=args.n_boot, seed=args.seed)
        low_conf = low_confidence_chunks(ci_lo, ci_hi, width_threshold=args.low_confidence_width).tolist()
        if any(low_conf):
            n_low_conf_docs += 1
        labels = DocumentLabels(
            doc_id=record.doc_id, chunks=record.chunks, num_chunks=len(record.chunks),
            positions=record.positions, beta=beta.tolist(), ci_lo=ci_lo.tolist(), ci_hi=ci_hi.tolist(),
            low_confidence=low_conf, alpha=alpha, metadata=record.metadata,
        )
        save_document_labels(labels, args.out_dir)

    pct = 100 * n_low_conf_docs / len(records)
    print(f"Fit {len(records)} documents at alpha={alpha}. "
          f"{n_low_conf_docs}/{len(records)} ({pct:.1f}%) have at least one low_confidence chunk "
          f"(CI width > {args.low_confidence_width}) -- spec §2.6/§9 logging requirement.")
    print(f"Saved DocumentLabels to {args.out_dir}/")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)

    m = sub.add_parser('measure', help="build documents + masks, call the reader, save (mask, token_f1) records")
    m.add_argument('--split', choices=['train', 'dev'], required=True,
                    help="UIT-ViQuAD's official split; English sources always draw from their own dev pool "
                         "regardless (see ttcompress/multilingual_sources.py)")
    m.add_argument('--sources', default='uit_viquad',
                    help=f"comma list from {ALL_SOURCES}, or 'all'. Multilingual upgrade: mixing in "
                         f"longbench/ruler/kamradt needs a multilingual --base-checkpoint in train_relevance.py "
                         f"(e.g. xlm-roberta-base), not vinai/phobert-base")
    m.add_argument('--reader-model', required=True)
    m.add_argument('--n', type=int, default=10, help="documents, split evenly across --sources (spec §7 pilot default: 10)")
    m.add_argument('--k', type=int, default=15, help="masks per document (spec §7 pilot default: 15)")
    m.add_argument('--max-new-tokens', type=int, default=None)
    m.add_argument('--device', default='cuda')
    m.add_argument('--seed', type=int, default=0)
    m.add_argument('--overlap-report', default=None,
                    help="scripts/check_uit_viquad_overlap.py's --out report; samples whose title is in "
                         "its uit_train_vs_uit_test check's title_overlap list are excluded (spec §1 -- "
                         "currently comes back EMPTY, see results/train_test_overlap.json, but re-run after "
                         "any change to the split/seed logic)")
    m.add_argument('--out-dir', required=True)
    m.set_defaults(func=cmd_measure)

    f = sub.add_parser('fit', help="ridge regression + bootstrap CI on saved measure output (CPU only)")
    f.add_argument('--in-dir', required=True)
    f.add_argument('--alpha', type=float, default=None)
    f.add_argument('--select-alpha-from', default=None, help="a Dev-split `measure` output dir; runs CV to pick alpha")
    f.add_argument('--alpha-grid', default='0.01,0.1,1.0,10.0,100.0')
    f.add_argument('--n-boot', type=int, default=1000)
    f.add_argument('--low-confidence-width', type=float, default=0.5,
                    help="CI width above which a chunk is flagged low_confidence (judgment call, not from the spec)")
    f.add_argument('--seed', type=int, default=0)
    f.add_argument('--out-dir', required=True)
    f.set_defaults(func=cmd_fit)

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
