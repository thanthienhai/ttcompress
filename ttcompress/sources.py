"""Dataset loaders -> QADocument, with seed-free, hash-based splits.

Sources (all public, pulled from the HuggingFace hub at runtime):

  single-hop, Vietnamese, synthetic haystack (needle paragraph + distractor paragraphs):
    uit_viquad   taidng/UIT-ViQuAD2.0      train = official train; dev/test = official
                                           validation hashed BY TITLE (the official test
                                           split ships no answers)
    xquad_vi     xquad / xquad.vi          no train (eval-only in the literature);
                                           dev/test hashed BY CONTEXT
  multi-hop, native distractor setting (10 titled paragraphs, 2+ supporting):
    vimqa        nguyenlab/vimqa           official train / validation / test (vi)
    hotpotqa     hotpotqa/hotpot_qa        train = official train; dev/test = official
                                           distractor validation hashed by id (en)
    2wiki        framolfese/2WikiMultihopQA  same as hotpotqa (its test ships no answers)

Why hash-based splits: the previous pipeline carved dev/test with
`random.Random(seed)` and two CLIs defaulted to different seeds, which
silently put 57 test documents into training (RUN_REPORT_2026-09-23.md §5
bug 4). A split here is a pure function of the example's own key -- there is
no seed to disagree about -- and the key is the unit that must not straddle
the boundary (the article title for UIT-ViQuAD, the passage for XQuAD-vi).

Why the needle position is uniform: the old haystack builder put the needle
at exactly {first, middle, last}, which hands a head+tail truncation baseline
2/3 of the needles for free and makes the position question unanswerable.
Here it is drawn uniformly over every gap; `metadata['needle_relpos']`
records it so results can still be sliced by depth.

Yes/no questions are dropped everywhere: a binary answer is matchable
without the evidence, which defeats a compression benchmark.
"""
from __future__ import annotations

import hashlib
import random
import unicodedata
from typing import Callable, Dict, Iterable, List, Optional, Sequence

from .data import QADocument

SOURCES = ('uit_viquad', 'xquad_vi', 'vimqa', 'hotpotqa', '2wiki')
LANGUAGE = {'uit_viquad': 'vi', 'xquad_vi': 'vi', 'vimqa': 'vi', 'hotpotqa': 'en', '2wiki': 'en'}
HOP = {'uit_viquad': 'single', 'xquad_vi': 'single', 'vimqa': 'multi', 'hotpotqa': 'multi', '2wiki': 'multi'}
SPLITS = ('train', 'dev', 'test')
# Which splits exist per source (xquad_vi is evaluation-only).
AVAILABLE_SPLITS = {s: SPLITS for s in SOURCES}
AVAILABLE_SPLITS['xquad_vi'] = ('dev', 'test')

YES_NO_ANSWERS = {'yes', 'no', 'đúng', 'không', 'sai', 'có'}
DEFAULT_HAYSTACK_CHARS = 30000
SPLIT_SALT = 'ttcompress-split-v1'


# ---------------------------------------------------------------------------
# hashing helpers -- the only source of "randomness" in split/sample choice
# ---------------------------------------------------------------------------

def hash_unit(key: str, salt: str = SPLIT_SALT) -> float:
    """Deterministic value in [0, 1) for `key`."""
    digest = hashlib.sha1(f'{salt}:{key}'.encode('utf-8')).hexdigest()
    return int(digest[:13], 16) / float(16 ** 13)


def hash_seed(key: str) -> int:
    return int(hashlib.sha1(key.encode('utf-8')).hexdigest()[:8], 16)


def dev_or_test(key: str, dev_fraction: float) -> str:
    return 'dev' if hash_unit(key) < dev_fraction else 'test'


def take_n(items: Sequence, n: Optional[int], key: Callable) -> List:
    """First `n` items by hash of key(item): deterministic, and nested
    (the n=100 subset is contained in the n=200 subset)."""
    ranked = sorted(items, key=lambda x: hash_unit(str(key(x)), salt='ttcompress-sample-v1'))
    return list(ranked) if n is None else list(ranked[:n])


def _norm(text: str) -> str:
    return unicodedata.normalize('NFC', text).lower().strip()


def is_yes_no(answer: str) -> bool:
    return _norm(answer).rstrip('.') in YES_NO_ANSWERS


def contains_answer(text: str, answers: Iterable[str]) -> bool:
    t = _norm(text)
    return any(_norm(a) and _norm(a) in t for a in answers)


# ---------------------------------------------------------------------------
# single-hop haystack construction
# ---------------------------------------------------------------------------

def build_haystack(
    needle: str, answers: Sequence[str], pool: Sequence[str], rng: random.Random,
    target_chars: int = DEFAULT_HAYSTACK_CHARS, same_title_pool: Sequence[str] = (),
    distractors: str = 'random', pool_norm: Optional[Dict[str, str]] = None,
):
    """(chunks, needle_index). Distractors never contain any gold answer
    string and never equal the needle. distractors='hard' draws paragraphs
    of the needle's own article first (topically close, the "hard
    negatives" ablation), then random ones; 'random' ignores topic.
    `pool_norm` (paragraph -> normalized text) is an optional cache so the
    answer-exclusion check doesn't re-normalize the pool per document."""
    if distractors not in ('random', 'hard'):
        raise ValueError(f"distractors must be 'random' or 'hard'; got {distractors!r}")
    norm_answers = [_norm(a) for a in answers if _norm(a)]
    norm_of = (lambda p: pool_norm[p]) if pool_norm is not None else _norm  # noqa: E731
    ok = lambda p: p != needle and not any(a in norm_of(p) for a in norm_answers)  # noqa: E731
    candidates = [p for p in pool if ok(p)]
    rng.shuffle(candidates)
    if distractors == 'hard':
        close = [p for p in same_title_pool if ok(p)]
        rng.shuffle(close)
        close_set = set(close)
        candidates = close + [p for p in candidates if p not in close_set]

    parts: List[str] = []
    total = len(needle)
    for p in candidates:
        if total + len(p) > target_chars and parts:
            break
        parts.append(p)
        total += len(p)
    idx = rng.randint(0, len(parts))  # uniform over every gap, both ends included
    parts.insert(idx, needle)
    return parts, idx


def _squad_rows_to_docs(rows: List[dict], chosen: List[dict], source: str, split: str, target_chars: int,
                        distractors: str) -> List[QADocument]:
    """Haystacks for the `chosen` questions; distractors come from every
    passage of the split portion (`rows`), never across the split boundary."""
    pool = list(dict.fromkeys(r['context'] for r in rows))
    pool_norm = {p: _norm(p) for p in pool}
    by_title: Dict[str, List[str]] = {}
    for r in rows:
        if r['title']:
            titled = by_title.setdefault(r['title'], [])
            if r['context'] not in titled:
                titled.append(r['context'])
    docs = []
    for r in chosen:
        doc_id = f"{source}_{split}_{r['id']}"
        rng = random.Random(hash_seed(doc_id))
        chunks, idx = build_haystack(r['context'], r['answers'], pool, rng, target_chars,
                                     by_title.get(r['title'], ()), distractors, pool_norm)
        docs.append(QADocument(
            doc_id=doc_id, source=source, language=LANGUAGE[source], hop='single', split=split,
            question=r['question'], answers=r['answers'], chunks=chunks, gold_chunks=[idx],
            cluster_id=r['cluster'],
            metadata={'source_id': r['id'], 'title': r['title'], 'needle_relpos': idx / max(1, len(chunks) - 1),
                      'distractors': distractors},
        ))
    return docs


def _uit_rows(split: str) -> List[dict]:
    from datasets import load_dataset
    official = 'train' if split == 'train' else 'validation'
    rows = []
    for row in load_dataset('taidng/UIT-ViQuAD2.0', split=official):
        if row['is_impossible'] or not row['answers'] or not row['answers']['text']:
            continue
        answers = list(dict.fromkeys(t for t in row['answers']['text'] if t.strip()))
        if not answers or is_yes_no(answers[0]):
            continue
        if split != 'train' and dev_or_test(f"uit:{row['title']}", 0.5) != split:
            continue
        rows.append({'id': row['id'], 'title': row['title'], 'context': row['context'],
                     'question': row['question'], 'answers': answers, 'cluster': f"uit:{row['title']}"})
    return rows


def _xquad_vi_rows(split: str) -> List[dict]:
    if split == 'train':
        raise ValueError("xquad_vi is evaluation-only: it has no train split")
    df = _read_parquet('xquad', 'xquad.vi/validation-00000-of-00001.parquet')
    rows = []
    for _, row in df.iterrows():
        cluster = 'xquad:' + hashlib.sha1(row['context'].encode('utf-8')).hexdigest()[:12]
        if dev_or_test(cluster, 0.2) != split:
            continue
        answers = list(dict.fromkeys(str(t) for t in row['answers']['text'] if str(t).strip()))
        if not answers or is_yes_no(answers[0]):
            continue
        rows.append({'id': str(row['id']), 'title': '', 'context': row['context'], 'question': row['question'],
                     'answers': answers, 'cluster': cluster})
    return rows


# ---------------------------------------------------------------------------
# multi-hop (native distractor setting)
# ---------------------------------------------------------------------------

def _read_parquet(repo_id: str, filename: str):
    import pandas as pd
    from huggingface_hub import hf_hub_download
    return pd.read_parquet(hf_hub_download(repo_id=repo_id, repo_type='dataset', filename=filename))


def multihop_row_to_doc(row, source: str, split: str) -> Optional[QADocument]:
    """One HotpotQA-schema row (context.title/sentences, supporting_facts.title)
    -> QADocument with one chunk per titled paragraph, or None if the row is
    unusable (yes/no or empty answer, supporting title missing from context)."""
    answer = str(row['answer']).strip()
    if not answer or is_yes_no(answer):
        return None
    titles = [str(t) for t in row['context']['title']]
    chunks = [f"{t}\n{' '.join(str(s).strip() for s in sents)}".strip()
              for t, sents in zip(titles, row['context']['sentences'])]
    gold = sorted({titles.index(str(t)) for t in row['supporting_facts']['title'] if str(t) in titles})
    if not gold:
        return None
    doc_id = f"{source}_{split}_{row['id']}"
    return QADocument(
        doc_id=doc_id, source=source, language=LANGUAGE[source], hop='multi', split=split,
        question=str(row['question']), answers=[answer], chunks=chunks, gold_chunks=gold,
        cluster_id=doc_id, metadata={'source_id': str(row['id']), 'type': str(row.get('type', ''))},
    )


_MULTIHOP_FILES = {
    'vimqa': ('nguyenlab/vimqa', {'train': ['data/train-00000-of-00001.parquet'],
                                  'dev': ['data/validation-00000-of-00001.parquet'],
                                  'test': ['data/test-00000-of-00001.parquet']}),
    'hotpotqa': ('hotpotqa/hotpot_qa', {'train': ['distractor/train-00000-of-00002.parquet',
                                                  'distractor/train-00001-of-00002.parquet'],
                                        'eval': ['distractor/validation-00000-of-00001.parquet']}),
    '2wiki': ('framolfese/2WikiMultihopQA', {'train': ['data/train-00000-of-00002.parquet',
                                                       'data/train-00001-of-00002.parquet'],
                                             'eval': ['data/validation-00000-of-00001.parquet']}),
}


def _multihop_docs(source: str, split: str) -> List[QADocument]:
    repo, files = _MULTIHOP_FILES[source]
    hashed = 'eval' in files and split != 'train'
    docs = []
    for filename in files['eval' if hashed else split]:
        df = _read_parquet(repo, filename)
        for _, row in df.iterrows():
            if hashed and dev_or_test(f"{source}:{row['id']}", 0.5) != split:
                continue
            doc = multihop_row_to_doc(row, source, split)
            if doc is not None:
                docs.append(doc)
    return docs


def pad_with_distractors(doc: QADocument, pool: Sequence[str], target_chars: int, rng: random.Random) -> QADocument:
    """Lengthen a multi-hop document with extra (easy, off-topic) paragraphs
    from other rows of the same split, inserted at random positions; gold
    indices are remapped. Distractors containing the answer are skipped."""
    chunks = list(doc.chunks)
    gold = set(doc.gold_chunks)
    total = sum(len(c) for c in chunks)
    candidates = [p for p in pool if p not in chunks and not contains_answer(p, doc.answers)]
    rng.shuffle(candidates)
    for p in candidates:
        if total + len(p) > target_chars:
            break
        pos = rng.randint(0, len(chunks))
        chunks.insert(pos, p)
        gold = {g + 1 if g >= pos else g for g in gold}
        total += len(p)
    return QADocument(**{**doc.to_dict(), 'chunks': chunks, 'gold_chunks': sorted(gold),
                         'metadata': {**doc.metadata, 'padded_to_chars': target_chars}})


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

def load_documents(
    source: str, split: str, n: Optional[int] = None, haystack_chars: int = DEFAULT_HAYSTACK_CHARS,
    distractors: str = 'random', multihop_pad_chars: int = 0,
) -> List[QADocument]:
    """Up to `n` documents of `source`/`split`, chosen by hash (no seed)."""
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}; choose from {SOURCES}")
    if split not in AVAILABLE_SPLITS[source]:
        raise ValueError(f"{source} has no {split!r} split (available: {AVAILABLE_SPLITS[source]})")

    if source in ('uit_viquad', 'xquad_vi'):
        rows = _uit_rows(split) if source == 'uit_viquad' else _xquad_vi_rows(split)
        chosen = take_n(rows, n, key=lambda r: r['id'])
        return _squad_rows_to_docs(rows, chosen, source, split, haystack_chars, distractors)

    docs = _multihop_docs(source, split)
    pool = list(dict.fromkeys(c for d in docs for c in d.chunks)) if multihop_pad_chars else []
    docs = take_n(docs, n, key=lambda d: d.doc_id)
    if multihop_pad_chars:
        docs = [pad_with_distractors(d, pool, multihop_pad_chars, random.Random(hash_seed(d.doc_id))) for d in docs]
    return docs


def parse_source_list(spec: str) -> List[str]:
    names = list(SOURCES) if spec == 'all' else [s.strip() for s in spec.split(',') if s.strip()]
    unknown = set(names) - set(SOURCES)
    if unknown:
        raise ValueError(f"unknown source(s) {sorted(unknown)}; choose from {SOURCES}")
    return names
