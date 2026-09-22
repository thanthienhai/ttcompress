"""Four published needle_in_haystack recipes, combined into one pool.

See data/SOURCES.md for exact provenance/license of everything loaded here.
Nothing here is self-built data: `load_longbench_passage_retrieval` reads a
real published benchmark split as-is; `generate_ruler_niah` and
`generate_kamradt_niah` reproduce two established synthetic-NIAH recipes
(templates copied verbatim from their source repos) over the same public
Paul Graham essay corpus RULER itself uses as its 'essay' haystack type;
`generate_infinitebench_passkey` reuses real per-sample passkey values from
InfiniteBench (Zhang et al. 2024, ACL) but re-scales the haystack and
re-randomizes the needle's depth -- see that function's docstring for why.
"""
from __future__ import annotations

import json
import os
import random
import re
from typing import Dict, List

from .data import NeedleSample

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_HERE, '..', 'data')
ESSAYS_PATH = os.path.join(_DATA_DIR, 'paul_graham_essays.json')
LONGBENCH_PATH = os.path.join(_DATA_DIR, 'longbench_passage_retrieval_en.jsonl')

_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')


def load_essays(path: str = ESSAYS_PATH) -> Dict[str, str]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def split_essay_pool(essays: Dict[str, str], num_dev: int = 10, seed: int = 1234) -> tuple[List[str], List[str]]:
    """Disjoint dev/test essay names -- RULER and Kamradt samples generated
    from dev essays vs. test essays never share a haystack document."""
    names = sorted(essays.keys())
    random.Random(seed).shuffle(names)
    return names[:num_dev], names[num_dev:]


# ---------------------------------------------------------------------------
# 1. LongBench passage_retrieval_en (THUDM, ACL 2024) -- real data, as-is.
# ---------------------------------------------------------------------------

_LONGBENCH_PARAGRAPH_SPLIT = re.compile(r'(?=Paragraph \d+:)')


def _longbench_chunks(context: str) -> List[str]:
    """LongBench's own context format is already chunked: 'Paragraph 1: ...
    Paragraph 2: ...' back to back, verified against the real data (each of
    the 200 rows has exactly 30 paragraphs this way)."""
    return [p.strip() for p in _LONGBENCH_PARAGRAPH_SPLIT.split(context) if p.strip()]


def _longbench_needle_index(reference_answer: str) -> int:
    """reference_answer is 'Paragraph N' (1-indexed) -- parse N and convert
    to a 0-indexed position into _longbench_chunks' output."""
    m = re.search(r'\d+', reference_answer)
    if not m:
        raise ValueError(f"couldn't parse a paragraph number out of reference_answer={reference_answer!r}")
    return int(m.group()) - 1


def load_longbench_passage_retrieval(path: str = LONGBENCH_PATH, num_dev: int = 20, seed: int = 1234) -> tuple[List[NeedleSample], List[NeedleSample]]:
    with open(path, 'r', encoding='utf-8') as f:
        rows = [json.loads(line) for line in f]
    samples = [
        NeedleSample(
            sample_id=row['_id'], context=row['context'], query=row['input'],
            reference_answer=row['answers'][0], doc_id=row['_id'],
            metadata={'dataset_source': 'longbench_passage_retrieval_en',
                      'needle_index': _longbench_needle_index(row['answers'][0])},
            chunks=_longbench_chunks(row['context']),
        )
        for row in rows
    ]
    rng = random.Random(seed)
    order = list(range(len(samples)))
    rng.shuffle(order)
    dev_idx, test_idx = set(order[:num_dev]), set(order[num_dev:])
    dev = [samples[i] for i in sorted(dev_idx)]
    test = [samples[i] for i in sorted(test_idx)]
    return dev, test


# ---------------------------------------------------------------------------
# 2. RULER niah_single_1 (Hsieh et al. 2024) -- generated, templates verbatim
#    from NVIDIA/RULER scripts/data/synthetic/{niah.py,constants.py}.
# ---------------------------------------------------------------------------

_RULER_NEEDLE = "One of the special magic numbers for {key} is: {value}."
_RULER_QUESTION = "What is the special magic number for {key} mentioned in the provided text?"

# RULER's own key generator samples "{adjective}-{noun}" pairs from the
# `wonderwords` package's word lists. Substituted here with a small bundled
# list (the key is an arbitrary label, not semantically load-bearing) to
# avoid an extra dependency -- everything else matches RULER verbatim.
_ADJECTIVES = ['quiet', 'brave', 'silver', 'ancient', 'gentle', 'rapid', 'hidden', 'golden',
               'frozen', 'lucky', 'distant', 'curious', 'steady', 'bright', 'wild']
_NOUNS = ['falcon', 'harbor', 'lantern', 'canyon', 'meadow', 'compass', 'ember', 'glacier',
          'orchard', 'summit', 'thicket', 'voyage', 'beacon', 'quarry', 'tundra']


def _words_of(text: str) -> List[str]:
    return re.sub(r'\s+', ' ', text).strip().split(' ')


def generate_ruler_niah(essays: Dict[str, str], essay_names: List[str], num_samples: int,
                         target_words: int = 750, seed: int = 42) -> List[NeedleSample]:
    rng = random.Random(seed)
    samples = []
    for i in range(num_samples):
        essay_name = essay_names[i % len(essay_names)]
        words = _words_of(essays[essay_name])
        if len(words) < target_words:
            words = (words * ((target_words // len(words)) + 1))
        haystack_text = ' '.join(words[:target_words])

        key = f"{rng.choice(_ADJECTIVES)}-{rng.choice(_NOUNS)}"
        value = str(rng.randint(1000000, 9999999))
        needle = _RULER_NEEDLE.format(key=key, value=value)

        sentences = _SENT_SPLIT.split(haystack_text)
        depth = rng.random()
        insert_at = int(len(sentences) * depth)
        chunks = sentences[:insert_at] + [needle] + sentences[insert_at:]
        context = ' '.join(chunks)

        samples.append(NeedleSample(
            sample_id=f"ruler_niah_{essay_name}_{i}", context=context,
            query=_RULER_QUESTION.format(key=key), reference_answer=value,
            # underscore, NOT ':' -- generate_labels.py uses doc_id as a
            # save_mask_outcomes filename, and 'ruler:bias.jsonl' is NTFS
            # alternate-data-stream syntax on Windows (silently writes into a
            # hidden stream of a file/dir named 'ruler' instead of failing
            # loudly). Caught by an actual end-to-end run, not a review.
            doc_id=f"ruler_{essay_name}",
            metadata={'dataset_source': 'ruler_niah_single_1', 'depth': depth, 'key': key, 'needle_index': insert_at},
            chunks=chunks,
        ))
    return samples


# ---------------------------------------------------------------------------
# 3. Classic Needle-In-A-Haystack (Kamradt) -- generated, needle/question
#    copied verbatim from gkamradt/LLMTest_NeedleInAHaystack
#    needlehaystack/tasks/single_needle.py.
# ---------------------------------------------------------------------------

_KAMRADT_NEEDLE = ("The best thing to do in San Francisco is eat a sandwich "
                    "and sit in Dolores Park on a sunny day.")
_KAMRADT_ANSWER = "eat a sandwich and sit in Dolores Park"
_KAMRADT_QUESTION = "What is the best thing to do in San Francisco?"


def generate_kamradt_niah(essays: Dict[str, str], essay_names: List[str], num_samples: int,
                           target_words: int = 750, seed: int = 43) -> List[NeedleSample]:
    rng = random.Random(seed)
    samples = []
    for i in range(num_samples):
        essay_name = essay_names[i % len(essay_names)]
        words = _words_of(essays[essay_name])
        if len(words) < target_words:
            words = (words * ((target_words // len(words)) + 1))
        haystack_text = ' '.join(words[:target_words])

        sentences = _SENT_SPLIT.split(haystack_text)
        depth = rng.random()
        insert_at = int(len(sentences) * depth)
        chunks = sentences[:insert_at] + [_KAMRADT_NEEDLE] + sentences[insert_at:]
        context = ' '.join(chunks)

        samples.append(NeedleSample(
            sample_id=f"kamradt_niah_{essay_name}_{i}", context=context,
            query=_KAMRADT_QUESTION, reference_answer=_KAMRADT_ANSWER,
            doc_id=f"kamradt_{essay_name}",
            metadata={'dataset_source': 'kamradt_niah', 'depth': depth, 'needle_index': insert_at},
            chunks=chunks,
        ))
    return samples


# ---------------------------------------------------------------------------
# 4. InfiniteBench passkey (Zhang et al. 2024, ACL) -- real per-sample
#    passkey VALUES from the published data, re-scaled and re-randomized.
# ---------------------------------------------------------------------------

_INFINITEBENCH_ID = 'xinrongzhang2022/InfiniteBench'
# InfiniteBench's own "noise" filler sentences (verified verbatim against the
# real data -- every one of the 590 rows uses exactly this repeating text as
# padding). The real per-sample passkey NUMBER is extracted from each row;
# only the filler amount and the needle's position are re-generated (the
# original repeats this filler to ~470,000 chars/sample with the needle
# always in the first ~7% -- unusable at that scale for Stage A's ridge
# regression, which needs at least as many masks K as chunks C, and handing
# g(i,L) a needle that's always near position 0 by construction would be a
# trivial win baked into the source data, not a result).
_INFINITEBENCH_NOISE_FILLER = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
_INFINITEBENCH_NEEDLE_RE = re.compile(r'The pass key is \d+\. Remember it\. \d+ is the pass key\.')


def _infinitebench_passkey_rows():
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=_INFINITEBENCH_ID, repo_type='dataset', filename='passkey.jsonl')
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def generate_infinitebench_passkey(target_words: int = 750, seed: int = 44) -> List[NeedleSample]:
    rows = _infinitebench_passkey_rows()
    rng = random.Random(seed)
    filler_words = _words_of(_INFINITEBENCH_NOISE_FILLER)
    filler_words = (filler_words * ((target_words // len(filler_words)) + 1))[:target_words]
    filler_sentences = _SENT_SPLIT.split(' '.join(filler_words))

    samples = []
    for row in rows:
        m = _INFINITEBENCH_NEEDLE_RE.search(row['context'])
        needle = m.group(0)
        depth = rng.random()
        insert_at = int(len(filler_sentences) * depth)
        chunks = filler_sentences[:insert_at] + [needle] + filler_sentences[insert_at:]
        context = ' '.join(chunks)
        samples.append(NeedleSample(
            sample_id=f"infinitebench_passkey_{row['id']}", context=context,
            query=row['input'], reference_answer=row['answer'][0],
            doc_id=f"infinitebench_passkey_{row['id']}",
            metadata={'dataset_source': 'infinitebench_passkey', 'depth': depth, 'needle_index': insert_at},
            chunks=chunks,
        ))
    return samples


def split_infinitebench_passkey(num_dev: int = 20, seed: int = 1234) -> tuple[List[NeedleSample], List[NeedleSample]]:
    samples = generate_infinitebench_passkey(seed=seed + 1)
    rng = random.Random(seed)
    order = list(range(len(samples)))
    rng.shuffle(order)
    dev_idx, test_idx = set(order[:num_dev]), set(order[num_dev:])
    dev = [samples[i] for i in sorted(dev_idx)]
    test = [samples[i] for i in sorted(test_idx)]
    return dev, test


# ---------------------------------------------------------------------------
# Combined dev/test split -- the one entry point train.py/evaluate.py use.
# ---------------------------------------------------------------------------

def build_dev_test_split(
    num_dev_essays: int = 10, samples_per_essay: int = 2,
    longbench_num_dev: int = 20, infinitebench_num_dev: int = 20, seed: int = 1234,
) -> tuple[List[NeedleSample], List[NeedleSample]]:
    """dev and test pools, each a mix of all four sources, with zero
    document overlap between dev and test within every source (disjoint
    essay pools for RULER/Kamradt, disjoint sample ids for LongBench and
    InfiniteBench passkey)."""
    essays = load_essays()
    dev_essays, test_essays = split_essay_pool(essays, num_dev=num_dev_essays, seed=seed)

    lb_dev, lb_test = load_longbench_passage_retrieval(num_dev=longbench_num_dev, seed=seed)
    ib_dev, ib_test = split_infinitebench_passkey(num_dev=infinitebench_num_dev, seed=seed)

    ruler_dev = generate_ruler_niah(essays, dev_essays, len(dev_essays) * samples_per_essay, seed=seed)
    ruler_test = generate_ruler_niah(essays, test_essays, len(test_essays) * samples_per_essay, seed=seed + 1)

    kamradt_dev = generate_kamradt_niah(essays, dev_essays, len(dev_essays) * samples_per_essay, seed=seed + 2)
    kamradt_test = generate_kamradt_niah(essays, test_essays, len(test_essays) * samples_per_essay, seed=seed + 3)

    dev = lb_dev + ruler_dev + kamradt_dev + ib_dev
    test = lb_test + ruler_test + kamradt_test + ib_test
    return dev, test


# ---------------------------------------------------------------------------
# Persisting the split -- train.py writes the ACTUAL dev/test samples it used
# to a file; evaluate.py reads that file back instead of re-deriving a split
# from CLI numbers. Re-deriving independently means any mismatched flag
# between the two commands (num_dev_essays, samples_per_essay, longbench_dev,
# seed) silently changes which documents land in dev vs test -- exactly the
# dev/test contamination PCS_METHOD_SPEC.md §7 warns about, and nothing would
# catch it. Serializing the resolved samples removes the chance of drift
# entirely: evaluate.py runs on the literal rows train.py swept over (dev) or
# deliberately held out (test), never a re-computed approximation of them.
# ---------------------------------------------------------------------------

def _sample_to_dict(s: NeedleSample) -> dict:
    return {
        'sample_id': s.sample_id, 'context': s.context, 'query': s.query,
        'reference_answer': s.reference_answer, 'doc_id': s.doc_id, 'metadata': s.metadata,
        'chunks': s.chunks,
    }


def _sample_from_dict(d: dict) -> NeedleSample:
    return NeedleSample(
        sample_id=d['sample_id'], context=d['context'], query=d['query'],
        reference_answer=d['reference_answer'], doc_id=d['doc_id'], metadata=d.get('metadata', {}),
        chunks=d.get('chunks', []),
    )


def save_dev_test_split(dev: List[NeedleSample], test: List[NeedleSample], path: str) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'dev': [_sample_to_dict(s) for s in dev],
                    'test': [_sample_to_dict(s) for s in test]}, f, ensure_ascii=False)


def load_dev_test_split(path: str) -> tuple[List[NeedleSample], List[NeedleSample]]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    dev = [_sample_from_dict(d) for d in data['dev']]
    test = [_sample_from_dict(d) for d in data['test']]
    return dev, test
