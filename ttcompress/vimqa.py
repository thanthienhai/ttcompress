"""VIMQA (Vietnamese multi-hop QA, `nguyenlab/vimqa` on HuggingFace, verified
2026-09-22) -- a THIRD public Vietnamese source, added after the user asked
whether any more public datasets could be added. Unlike UIT-ViQuAD/XQuAD-vi,
VIMQA is a native needle-in-haystack task already: each row ships 10
candidate Wikipedia-derived documents (HotpotQA-distractor style), only 1-3
of them containing a supporting fact for the (multi-hop) answer. No synthetic
haystack construction is needed -- build_document.py's build_document/
build_documents are deliberately NOT used here; ConstructedDocuments are
built directly from VIMQA's own context structure.

Two real, verified deviations from using the raw data as-is:
- Yes/no rows (`answer` in {'đúng', 'không'}, ~30-49% of every split,
  verified directly) are dropped: a binary answer makes token_f1 trivially
  matchable without real retrieval, defeating the point of a
  needle-in-haystack compression benchmark.
- Chunk granularity is per candidate document (title), not per sentence: all
  of a title's sentences are joined into one chunk, matching LongBench's
  paragraph-level granularity (data/SOURCES.md §2) rather than
  RULER/Kamradt's sentence-level one -- VIMQA's own supporting-fact unit is
  already "one sentence within a title", but its distractor unit (the thing
  Stage A's masking should keep/drop as one decision) is the whole title.

VIMQA's official train/validation/test split IS directly usable as three
genuinely separate pools -- verified 2026-09-22: zero null answers/contexts
in any split, `test` ships real gold answers (unlike UIT-ViQuAD's official
Test split, data/SOURCES.md §6, which does not). `load_split('test')` is
reserved exclusively for ttcompress.vietnamese_public_test; nothing upstream
of that should call it.

ConstructedDocument.needle_index is single-valued pure metadata (verified:
never read by Stage A's actual masking/outcome-measurement logic, which
operates purely on .chunks -- see build_document.py's ConstructedDocument
docstring) -- multi-hop's 1-3 supporting titles are set as
metadata['needle_indices'] (the full list); needle_index is just the first
one, for any code that expects a single value (e.g. the needle_index bounds
check every other source's tests already apply).
"""
from __future__ import annotations

from typing import List

from .build_document import ConstructedDocument

_REPO_ID = 'nguyenlab/vimqa'
_YES_NO_ANSWERS = {'đúng', 'không'}


def _load_raw(split: str):
    from huggingface_hub import hf_hub_download
    import pandas as pd
    path = hf_hub_download(repo_id=_REPO_ID, repo_type='dataset', filename=f'data/{split}-00000-of-00001.parquet')
    return pd.read_parquet(path)


def _row_to_document(row, doc_id: str) -> ConstructedDocument:
    titles = list(row['context']['title'])
    chunks = [' '.join(sents) for sents in row['context']['sentences']]

    needle_indices = sorted({titles.index(t) for t in row['supporting_facts']['title'] if t in titles})
    if not needle_indices:
        raise ValueError(f"{doc_id}: none of supporting_facts' titles found in context titles")

    return ConstructedDocument(
        doc_id=doc_id, chunks=chunks, needle_index=needle_indices[0],
        question=row['question'], answer_text=row['answer'],
        metadata={'dataset_source': 'vimqa', 'source_id': row['id'], 'question_type': row['type'],
                  'needle_indices': needle_indices},
    )


def load_split(split: str) -> List[ConstructedDocument]:
    """`split` in 'train' / 'validation' / 'test' (VIMQA's own official
    split). Yes/no-answer rows are dropped (see module docstring)."""
    df = _load_raw(split)
    df = df[~df['answer'].isin(_YES_NO_ANSWERS)]
    return [
        _row_to_document(row, doc_id=f"vimqa_{split}_{i:06d}")
        for i, (_, row) in enumerate(df.iterrows())
    ]
