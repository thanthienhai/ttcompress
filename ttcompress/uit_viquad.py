"""UIT-ViQuAD 2.0 loader (Nguyen et al., COLING 2020) --
OUTCOME_SUPERVISED_RELEVANCE_SPEC.md §1: replaces the internal
qa_combined.jsonl/vncompress_vi_v2 corpus as Stage A's data source.

    taidng/UIT-ViQuAD2.0 on HuggingFace -- verified 2026-09-22: plain parquet
    files (no legacy loading script), splits train=28454 / validation=3814 /
    test=7301 (the spec's own "~5.700 dev / ~3.821 test" figures are off --
    use the real HF card numbers, noted here rather than silently
    overridden). Schema: id, uit_id, title, context, question,
    answers{text[], answer_start[]}, is_impossible, plausible_answers.

The official 'test' split ships with NO gold answers -- verified 2026-09-22:
all 7301 rows have answers=None (is_impossible=False for every one of them,
so this isn't the SQuAD2.0-style "unanswerable" signal; it's the standard
withheld-answer convention for a public leaderboard test set). It cannot be
scored and load_split() below simply can't use it -- `split_dev_tune_test`
carves the *Dev* split into a tuning portion and a small reserved-once test
portion instead (see ttcompress/vietnamese_public_test.py, which is why that
module exists rather than just using the official test split directly).

License (spec §1, §8): the HF mirror carries NO license field at all -- not
MIT, not anything else. The dataset card's only relevant text is "freely
available to encourage the research community" (COLING 2020 abstract),
pointing to https://nlp.uit.edu.vn/datasets and
https://vlsp.org.vn/vlsp2021/eval/mrc for specifics. This is NOT resolved by
this code -- confirm with the UIT NLP group before any publication that uses
labels trained on this data, per spec §1/§8.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

HF_DATASET_ID = 'taidng/UIT-ViQuAD2.0'
_SPLIT_MAP = {'train': 'train', 'dev': 'validation', 'validation': 'validation', 'test': 'test'}

# Real HF card sizes (verified 2026-09-22), for reference/sanity-checking a
# fresh download against silent drift upstream. 'test' is unusable here (see
# module docstring) -- kept in the map only so load_split('test') fails with
# a clear downstream error (no answers) rather than an unhelpful KeyError.
EXPECTED_SPLIT_SIZES = {'train': 28454, 'dev': 3814, 'test': 7301}


@dataclass
class ViquadSample:
    id: str
    title: str
    context: str
    question: str
    answer_text: str
    answer_start: int


def load_split(split: str, exclude_impossible: bool = True) -> List[ViquadSample]:
    """Answerable (is_impossible=False) samples only by default -- Stage A
    needs a real reference answer to score token_f1 against (§2.4); an
    unanswerable SQuAD2.0-style question has no gold span to use as the
    needle's answer."""
    from datasets import load_dataset

    if split not in _SPLIT_MAP:
        raise ValueError(f"split must be one of {sorted(_SPLIT_MAP)}; got {split!r}")
    ds = load_dataset(HF_DATASET_ID, split=_SPLIT_MAP[split])

    samples = []
    for row in ds:
        if exclude_impossible and row['is_impossible']:
            continue
        if row['answers'] is None:  # the official 'test' split -- see module docstring
            continue
        texts = row['answers']['text']
        starts = row['answers']['answer_start']
        if not texts:
            continue
        samples.append(ViquadSample(
            id=row['id'], title=row['title'], context=row['context'],
            question=row['question'], answer_text=texts[0], answer_start=starts[0],
        ))
    return samples


def split_dev_tune_test(num_test: int = 300, seed: int = 1234) -> Tuple[List[ViquadSample], List[ViquadSample]]:
    """Carve the official Dev split (3814 answerable questions) into a
    tuning portion (used by multilingual_sources.select_documents for Stage
    A's alpha selection and Stage 6's lambda sweep) and a small reserved test
    portion (used once, by ttcompress.vietnamese_public_test) -- since the
    official 'test' split has no usable answers at all (see module
    docstring), this Dev split is the only place a genuinely held-out,
    answerable UIT-ViQuAD test portion can come from."""
    samples = load_split('dev')
    rng = random.Random(seed)
    order = list(range(len(samples)))
    rng.shuffle(order)
    test_idx = set(order[:num_test])
    tune = [samples[i] for i in range(len(samples)) if i not in test_idx]
    test = [samples[i] for i in sorted(test_idx)]
    return tune, test


def filter_by_excluded_titles(samples: Sequence[ViquadSample], excluded_titles: Sequence[str]) -> List[ViquadSample]:
    """Drop samples whose (case/whitespace-normalized) title is in
    `excluded_titles`. Use this with scripts/check_uit_viquad_overlap.py's
    `uit_train_vs_uit_test` check output -- cheap insurance against a
    UIT-ViQuAD-trained relevance model having seen text it will later be
    tested on via ttcompress/vietnamese_public_test.py's reserved portion.
    (An earlier version of this check, against vncompress's internal
    wikipedia_vi_raw.json corpus -- the source of the now-dropped
    vcc_bench_v2.json -- found real overlap on title "Hà Nội"; that finding
    is historical now that nothing reads that corpus, but the same kind of
    check on the current train/test boundary is still run for real, not
    assumed clean -- see results/train_test_overlap.json.)"""
    excluded = {t.strip().lower() for t in excluded_titles}
    return [s for s in samples if s.title.strip().lower() not in excluded]


def build_paragraph_pool(samples: Sequence[ViquadSample]) -> List[str]:
    """Deduplicated context paragraphs across all samples (many questions
    share one context, like SQuAD/UIT-ViQuAD) -- the pool
    build_document.build_base_haystack_text draws distractors from. One pool
    per split (train pool for train documents, dev pool for dev documents) so
    Stage A/B never mixes distractor paragraphs across the official split
    boundary (spec §5)."""
    seen: Dict[str, str] = {}
    for s in samples:
        seen.setdefault(s.context, s.context)
    return list(seen.values())
