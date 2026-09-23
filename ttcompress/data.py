"""The one document type every source produces, plus jsonl (de)serialization.

A QADocument is a question over an ordered list of chunks (paragraphs). The
unit everything downstream works on is the chunk: Stage A masks chunks, the
pruner scores chunks, selection keeps chunks under a token budget.
`gold_chunks` is the supervision the *oracle* baselines use (the needle
paragraph for single-hop haystacks, the supporting-fact titles for
multi-hop); it is never read by label generation or by the pruner at
inference time.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

CHUNK_SEP = '\n\n'


@dataclass
class QADocument:
    doc_id: str                 # unique across every source and split
    source: str                 # 'uit_viquad' | 'xquad_vi' | 'vimqa' | 'hotpotqa' | '2wiki'
    language: str               # 'vi' | 'en' -- picks the reader prompt; results are never pooled across it
    hop: str                    # 'single' | 'multi'
    split: str                  # 'train' | 'dev' | 'test'
    question: str
    answers: List[str]          # gold aliases; answers[0] is the canonical one
    chunks: List[str]
    gold_chunks: List[int]      # needle / supporting-fact chunk indices (oracle baselines only)
    cluster_id: str             # bootstrap cluster: questions sharing a passage/article resample together
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_chunks(self) -> int:
        return len(self.chunks)

    def text(self, keep: Optional[Sequence[int]] = None) -> str:
        """Kept chunks joined in ORIGINAL order (keep=None -> full context)."""
        if keep is None:
            return CHUNK_SEP.join(self.chunks)
        return CHUNK_SEP.join(self.chunks[i] for i in sorted(set(keep)))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'QADocument':
        return cls(**d)


def save_documents(docs: Iterable[QADocument], path: str) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for d in docs:
            f.write(json.dumps(d.to_dict(), ensure_ascii=False) + '\n')


def load_documents(path: str) -> List[QADocument]:
    with open(path, 'r', encoding='utf-8') as f:
        return [QADocument.from_dict(json.loads(line)) for line in f if line.strip()]
