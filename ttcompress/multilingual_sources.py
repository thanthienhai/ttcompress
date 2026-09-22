"""Multilingual merge (upgrade over OUTCOME_SUPERVISED_RELEVANCE_SPEC.md's
original UIT-ViQuAD-only Stage A): pick ConstructedDocuments from any mix of
UIT-ViQuAD 2.0 + XQuAD-vi + VIMQA (Vietnamese) and four English public
sources (LongBench passage_retrieval_en, RULER niah_single_1, classic
Kamradt NIAH, InfiniteBench passkey -- see public_datasets.py's module
docstring for all four), so Stage A/B/C's outcome-supervised training and
Stage 6's sweep/eval are no longer Vietnamese-only, and every source -- both
languages -- is a public, citable dataset (vcc_bench_v2.json, an unpublished
internal benchmark, was dropped entirely; see README "Public Vietnamese test
set" section for why and what replaced it).

English sources are drawn ONLY from ttcompress.public_datasets'
build_dev_test_split() "dev" half -- their "test" half stays reserved for
PCS's own evaluate.py and, now, Stage 6's multilingual final evaluation
(evaluate_relevance.py). XQuAD-vi (ttcompress.xquad_vi) is architecturally the
same: it has no official train/dev/test split of its own (it's an eval-only
benchmark in the literature), so it's treated exactly like the English
sources here -- always its "dev" slice, test half reserved for
ttcompress.vietnamese_public_test. This module never touches either reserved
test half. There is no official train/dev split for these sources the way
UIT-ViQuAD has one; the same "dev" pool is drawn from for both Stage A's
training documents and Stage A's alpha-selection documents (with different
random masks each time, via `seed`) -- a documented simplification, not a
discipline the spec defines for these sources.

VIMQA (ttcompress.vimqa) is architecturally closer to UIT-ViQuAD: it DOES
have its own official train/validation/test split, all three directly
usable (unlike UIT-ViQuAD's official Test, see vimqa.py's module docstring).
`uit_viquad_split='train'`/`'dev'` maps onto VIMQA's own 'train'/'validation'
splits; its 'test' split is reserved exclusively for
ttcompress.vietnamese_public_test, same discipline as the other two
Vietnamese sources' reserved test portions.
"""
from __future__ import annotations

import random
from typing import Dict, List, Sequence

from .build_document import ConstructedDocument, build_documents, document_from_needle_sample
from .uit_viquad import build_paragraph_pool, filter_by_excluded_titles, load_split, split_dev_tune_test

ENGLISH_SOURCES = ('longbench', 'ruler', 'kamradt', 'infinitebench')
ALL_SOURCES = ('uit_viquad', 'xquad_vi', 'vimqa') + ENGLISH_SOURCES

# generate_labels.py/evaluate_relevance.py's short --sources names -> the
# metadata['dataset_source'] values public_datasets.py actually tags samples with.
_DATASET_SOURCE_NAME = {
    'longbench': 'longbench_passage_retrieval_en',
    'ruler': 'ruler_niah_single_1',
    'kamradt': 'kamradt_niah',
    'infinitebench': 'infinitebench_passkey',
}


def _even_counts(n: int, keys: Sequence[str]) -> Dict[str, int]:
    base, remainder = divmod(n, len(keys))
    return {k: base + (1 if i < remainder else 0) for i, k in enumerate(keys)}


def english_dev_samples_by_source(seed: int = 1234) -> Dict[str, list]:
    """dataset_source -> NeedleSample list, from public_datasets' dev half only."""
    from .public_datasets import build_dev_test_split
    dev, _test = build_dev_test_split(seed=seed)
    by_source: Dict[str, list] = {}
    for s in dev:
        by_source.setdefault(s.metadata['dataset_source'], []).append(s)
    return by_source


def select_documents(
    sources: Sequence[str], uit_viquad_split: str, n: int, seed: int,
    exclude_uit_viquad_titles: Sequence[str] = (),
) -> List[ConstructedDocument]:
    """Up to `n` ConstructedDocuments, split evenly across `sources`
    (subset of ALL_SOURCES). `uit_viquad_split` is 'train' or 'dev' (UIT-ViQuAD's
    own official split, per spec §5); English sources always draw from their
    shared "dev" pool regardless (see module docstring).
    `exclude_uit_viquad_titles`: scripts/check_uit_viquad_overlap.py's
    title_overlap output (spec §1) -- only applies to the UIT-ViQuAD portion,
    the only source that check was ever run against."""
    unknown = set(sources) - set(ALL_SOURCES)
    if unknown:
        raise ValueError(f"unknown source(s) {unknown}; choose from {ALL_SOURCES}")
    counts = _even_counts(n, list(sources))
    documents: List[ConstructedDocument] = []

    if 'uit_viquad' in counts:
        if uit_viquad_split == 'dev':
            # the official Dev split minus its reserved test portion -- see
            # uit_viquad.split_dev_tune_test / vietnamese_public_test.py.
            # 'train' still uses the full official Train split unchanged.
            samples, _reserved_test = split_dev_tune_test(seed=seed)
        else:
            samples = load_split(uit_viquad_split)
        if exclude_uit_viquad_titles:
            samples = filter_by_excluded_titles(samples, exclude_uit_viquad_titles)
        pool = build_paragraph_pool(samples)
        want = counts['uit_viquad']
        if want < len(samples):
            samples = random.Random(seed).sample(samples, want)
        documents.extend(build_documents(samples, pool, seed=seed, dataset_source='uit_viquad', doc_id_prefix='uitviquad'))

    if 'xquad_vi' in counts:
        from .xquad_vi import split_dev_test as xquad_split_dev_test
        xquad_dev, _xquad_test = xquad_split_dev_test(seed=seed)
        pool = build_paragraph_pool(xquad_dev)
        want = counts['xquad_vi']
        samples = (random.Random(seed + 100).sample(xquad_dev, want)
                   if want < len(xquad_dev) else xquad_dev)
        documents.extend(build_documents(samples, pool, seed=seed + 100, dataset_source='xquad_vi', doc_id_prefix='xquadvi'))

    if 'vimqa' in counts:
        from .vimqa import load_split as vimqa_load_split
        vimqa_split = 'train' if uit_viquad_split == 'train' else 'validation'
        vimqa_pool = vimqa_load_split(vimqa_split)
        want = counts['vimqa']
        documents.extend(random.Random(seed + 200).sample(vimqa_pool, want)
                          if want < len(vimqa_pool) else vimqa_pool)

    english_wanted = {src: counts[src] for src in counts if src in ENGLISH_SOURCES}
    if english_wanted:
        by_source = english_dev_samples_by_source(seed=seed)
        for i, (src, want) in enumerate(english_wanted.items()):
            pool_key = _DATASET_SOURCE_NAME[src]
            available = by_source.get(pool_key, [])
            samples = (random.Random(seed + i + 1).sample(available, want)
                       if want < len(available) else available)
            documents.extend(document_from_needle_sample(s) for s in samples)

    return documents
