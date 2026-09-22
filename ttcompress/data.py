"""The one sample type every needle_in_haystack source produces.

Loaders/generators for the three public sources (LongBench, RULER, classic
NIAH) live in public_datasets.py, which imports NeedleSample from here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class NeedleSample:
    sample_id: str
    context: str
    query: str
    reference_answer: str
    doc_id: str  # cluster label for the document-clustered bootstrap
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Chunk decomposition of `context`, in order (CHUNK_SEP.join(chunks) ==
    # context) -- populated by public_datasets.py's generators so
    # OUTCOME_SUPERVISED_RELEVANCE_SPEC.md's Stage A can mask/measure/fit at
    # chunk granularity on these sources too, not just UIT-ViQuAD. Empty for
    # samples nothing has decomposed (PCS's own whole-document compression
    # never needed this).
    chunks: List[str] = field(default_factory=list)
