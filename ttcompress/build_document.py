"""Needle+haystack document construction for Stage A
(OUTCOME_SUPERVISED_RELEVANCE_SPEC.md §2.2), replicating the algorithm the
existing internal needle_in_haystack task already uses
(vncompress/scripts/build_vcc_bench_v2.py: build_base_haystack_text /
insert_needle, confirmed by direct code read) so training documents (built
here from UIT-ViQuAD 2.0 or XQuAD-vi) and the public Vietnamese test pool
(ttcompress/vietnamese_public_test.py -- reserved portions of the same two
sources, not vncompress's internal vcc_bench_v2.json, which was dropped
entirely for not being a public dataset) are built with the matched
difficulty spec §2.2/§8 require:

  - Distractor pool = the WHOLE paragraph pool, no topic/title filtering --
    confirmed the existing task shuffles across the entire corpus, not
    same-article distractors.
  - Fill to a 30,000-char budget (TARGET_CONTEXT_LEN), greedily appending
    shuffled paragraphs; the paragraph that crosses the budget is truncated
    at a word boundary if that boundary falls past 50% of it, else kept whole.
  - Needle inserted at one of exactly 3 discrete positions -- 'beginning'
    (index 0), 'middle' (len(parts)//2), 'end' (len(parts)) -- ASSIGNED PER
    DOCUMENT (not randomly per question) via assign_needle_positions, an
    even/shuffled split across the whole batch, matching the existing task's
    pre-shuffled position list.
  - Chunk = one paragraph; parts are joined with '\\n\\n', the same separator
    the existing task's paragraph split/join uses.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from .data import NeedleSample
from .uit_viquad import ViquadSample

TARGET_CONTEXT_LEN = 30000
NEEDLE_POSITIONS = ('beginning', 'middle', 'end')


def build_base_haystack_text(pool: Sequence[str], exclude_answer: str, rng: random.Random) -> List[str]:
    """Shuffle the whole paragraph pool, drop any paragraph containing
    `exclude_answer` as a substring (so only the needle carries the current
    question's answer), then greedily fill to TARGET_CONTEXT_LEN chars.
    Returns the list of paragraph chunks (the last one possibly
    word-boundary-truncated), NOT yet joined."""
    candidates = [p for p in pool if exclude_answer not in p]
    rng.shuffle(candidates)

    parts: List[str] = []
    total_len = 0
    for p in candidates:
        if total_len >= TARGET_CONTEXT_LEN:
            break
        parts.append(p)
        total_len += len(p)

    if parts and total_len > TARGET_CONTEXT_LEN:
        overshoot = total_len - TARGET_CONTEXT_LEN
        last = parts[-1]
        cut_at = len(last) - overshoot
        if cut_at > len(last) * 0.5:
            space = last.rfind(' ', 0, cut_at)
            if space > 0:
                parts[-1] = last[:space] + '.'
    return parts


def insert_needle(parts: Sequence[str], needle_text: str, position: str) -> tuple[List[str], int]:
    """(chunks_with_needle, needle_index). `position` in NEEDLE_POSITIONS."""
    if position not in NEEDLE_POSITIONS:
        raise ValueError(f"position must be one of {NEEDLE_POSITIONS}; got {position!r}")
    chunks = list(parts)
    if position == 'beginning':
        idx = 0
    elif position == 'middle':
        idx = len(chunks) // 2
    else:
        idx = len(chunks)
    chunks.insert(idx, needle_text)
    return chunks, idx


def assign_needle_positions(n_documents: int, seed: int) -> List[str]:
    """Roughly-even split of NEEDLE_POSITIONS across n_documents, shuffled --
    one position per DOCUMENT, matching the existing task's per-haystack
    (not per-question) assignment."""
    if n_documents <= 0:
        return []
    rng = random.Random(seed)
    base, remainder = divmod(n_documents, len(NEEDLE_POSITIONS))
    counts = [base + (1 if i < remainder else 0) for i in range(len(NEEDLE_POSITIONS))]
    positions: List[str] = []
    for pos, count in zip(NEEDLE_POSITIONS, counts):
        positions.extend([pos] * count)
    rng.shuffle(positions)
    return positions


@dataclass
class ConstructedDocument:
    doc_id: str
    chunks: List[str]        # paragraph texts, in final document order (haystack + needle)
    needle_index: int        # chunks[needle_index] is the answer-bearing paragraph
    question: str
    answer_text: str
    metadata: Dict = field(default_factory=dict)

    @property
    def num_chunks(self) -> int:
        return len(self.chunks)

    def text(self) -> str:
        return '\n\n'.join(self.chunks)


def build_document(doc_id: str, sample: ViquadSample, paragraph_pool: Sequence[str], position: str, seed: int,
                    dataset_source: str = 'uit_viquad') -> ConstructedDocument:
    """One constructed needle+haystack document for one SQuAD-schema sample
    (UIT-ViQuAD 2.0 or, via the same construction, ttcompress.xquad_vi --
    `dataset_source` tags provenance for prompt-template routing and
    per-source reporting, see ttcompress/reader.py's PROMPT_TEMPLATES)."""
    rng = random.Random(seed)
    base_parts = build_base_haystack_text(paragraph_pool, sample.answer_text, rng)
    chunks, needle_index = insert_needle(base_parts, sample.context, position)
    return ConstructedDocument(
        doc_id=doc_id, chunks=chunks, needle_index=needle_index,
        question=sample.question, answer_text=sample.answer_text,
        metadata={'dataset_source': dataset_source, 'source_id': sample.id,
                  'title': sample.title, 'needle_position': position},
    )


def build_documents(samples: Sequence[ViquadSample], paragraph_pool: Sequence[str], seed: int = 0,
                     dataset_source: str = 'uit_viquad', doc_id_prefix: str = 'doc') -> List[ConstructedDocument]:
    """One document per sample (spec §7: N is now bounded by the number of
    UIT-ViQuAD questions, not a distinct source-document count)."""
    positions = assign_needle_positions(len(samples), seed=seed)
    return [
        build_document(f"{doc_id_prefix}_{i:06d}", sample, paragraph_pool, position, seed=seed + i,
                        dataset_source=dataset_source)
        for i, (sample, position) in enumerate(zip(samples, positions))
    ]


def document_from_needle_sample(sample: NeedleSample) -> ConstructedDocument:
    """Adapter for the multilingual merge: ttcompress.public_datasets'
    LongBench/RULER/Kamradt generators already populate NeedleSample.chunks
    (+ metadata['needle_index']) the same way build_document.build_document
    does for UIT-ViQuAD -- this just repackages one into the other so Stage A
    (generate_labels.py) and Stage 6 (evaluate_relevance.py) can treat every
    source uniformly as a ConstructedDocument. Requires `sample.chunks`
    populated; raises loudly rather than silently degrading if it isn't (an
    un-chunked NeedleSample means the caller picked a source before its
    generator exposed chunks, not something to guess around)."""
    if not sample.chunks:
        raise ValueError(f"{sample.sample_id}: NeedleSample.chunks is empty -- this source doesn't expose "
                          f"chunk structure, document_from_needle_sample can't build a ConstructedDocument from it")
    needle_index = sample.metadata.get('needle_index')
    if needle_index is None:
        raise ValueError(f"{sample.sample_id}: metadata['needle_index'] missing")
    return ConstructedDocument(
        doc_id=sample.doc_id, chunks=list(sample.chunks), needle_index=needle_index,
        question=sample.query, answer_text=sample.reference_answer, metadata=dict(sample.metadata),
    )
