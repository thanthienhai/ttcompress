"""The public Vietnamese test pool for Stage 6's final evaluation --
replaces vcc_bench_v2.json (an unpublished internal vncompress benchmark,
dropped entirely: it was never released anywhere citable, unlike every other
source in this pipeline). Combines the three public monolingual-Vietnamese
sources' reserved test portions:

  - UIT-ViQuAD 2.0: NOT its official 'test' split -- verified 2026-09-22 that
    split ships with no gold answers at all (all 7301 rows, standard
    withheld-answer convention for a public leaderboard set) and can't be
    scored. Instead, ttcompress.uit_viquad.split_dev_tune_test() carves a
    small reserved portion out of the *Dev* split; the rest of Dev is what
    multilingual_sources.py's sweep/alpha-selection actually tunes on, so
    this reserved slice is still genuinely untouched by anything upstream.
    Built into a needle+haystack document via ttcompress.build_document, same
    as everywhere else this source is used.
  - XQuAD-vi: its test half (ttcompress.xquad_vi.split_dev_test), the dev
    half reserved for multilingual_sources.py's sweep instead. Also built via
    ttcompress.build_document.
  - VIMQA: its official 'test' split IS directly usable (verified
    2026-09-06 -- unlike UIT-ViQuAD's), reserved exclusively for this pool;
    'train'/'validation' are what multilingual_sources.py draws from
    instead. Ships as a native needle-in-haystack document already
    (ttcompress.vimqa.load_split) -- NOT run through build_document, unlike
    the other two.

All three are genuinely public, citable datasets (see the provenance notes in
ttcompress/uit_viquad.py, ttcompress/xquad_vi.py and ttcompress/vimqa.py).
"""
from __future__ import annotations

import random
from typing import List

from .build_document import build_documents
from .data import NeedleSample
from .uit_viquad import build_paragraph_pool, split_dev_tune_test
from .vimqa import load_split as vimqa_load_split
from .xquad_vi import split_dev_test as xquad_split_dev_test


def load(n: int = 120, seed: int = 1234) -> List[NeedleSample]:
    """Up to `n` test documents (default 120, matching the scale the dropped
    vcc_bench_v2.json ran at), evenly split across all three sources.
    `n=None` for the full reserved pool."""
    _uit_tune, uit_test = split_dev_tune_test(seed=seed)
    uit_pool = build_paragraph_pool(uit_test)
    _xquad_dev, xquad_test = xquad_split_dev_test(seed=seed)
    xquad_pool = build_paragraph_pool(xquad_test)
    vimqa_test = vimqa_load_split('test')

    if n is not None:
        counts = [n // 3 + (1 if i < n % 3 else 0) for i in range(3)]
        uit_test = random.Random(seed).sample(uit_test, min(counts[0], len(uit_test)))
        xquad_test = random.Random(seed + 1).sample(xquad_test, min(counts[1], len(xquad_test)))
        vimqa_test = random.Random(seed + 2).sample(vimqa_test, min(counts[2], len(vimqa_test)))

    documents = (
        build_documents(uit_test, uit_pool, seed=seed, dataset_source='uit_viquad', doc_id_prefix='uitviquad_test')
        + build_documents(xquad_test, xquad_pool, seed=seed + 1, dataset_source='xquad_vi', doc_id_prefix='xquadvi_test')
    )
    samples = [
        NeedleSample(sample_id=d.doc_id, context=d.text(), query=d.question,
                     reference_answer=d.answer_text, doc_id=d.doc_id, metadata=d.metadata)
        for d in documents
    ]
    samples.extend(
        NeedleSample(sample_id=d.doc_id, context=d.text(), query=d.question,
                     reference_answer=d.answer_text, doc_id=d.doc_id, metadata=d.metadata, chunks=list(d.chunks))
        for d in vimqa_test
    )
    return samples
