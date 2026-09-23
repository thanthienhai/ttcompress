"""Budget-aware selection and every compression arm evaluate.py runs.

A fixed token budget (reader tokens) is the only fair comparison: every arm
gets budget = ceil(full_tokens / ratio) for the same document and ratio.
Chunk arms return per-chunk scores; `select_by_scores` keeps the best chunks
that fit (greedy by score, original order preserved). If not even the best
chunk fits, it is kept truncated to the budget, so no arm ever gets an empty
context. Text arms (LLMLingua-2) compress the string themselves at
rate = 1/ratio; their realized token count is recorded, never assumed.

Arms (name -> what it is):
  full             no compression (reference ceiling, ratio ignored)
  lead             first chunks in document order (truncation baseline)
  random           random chunk order (seeded by doc id)
  bm25             lexical BM25 of question vs chunk
  embed            dense cosine, BAAI/bge-m3 CLS embeddings
  oracle_span      chunks containing a gold answer string first (answer-span oracle, RQ2)
  oracle_support   gold chunks (needle / supporting facts) first
  oracle_beta      Stage A attribution of the test document (upper bound; needs --oracle-beta-dir)
  pruner:<path>    our distilled pruner (any number of checkpoints)
  provence:<hf_id> Provence / XProvence reranking score per chunk (naver/...)
  llmlingua2       LLMLingua-2 token-level compression
"""
from __future__ import annotations

import math
import random
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .data import QADocument
from .sources import contains_answer, hash_seed


@dataclass
class Selection:
    kept: List[int]            # kept chunk indices (empty for text arms)
    text: str                  # compressed context handed to the reader
    kept_tokens: int
    budget: int
    full_tokens: int
    seconds: float = 0.0       # compressor wall-clock for this document
    truncated: bool = False    # nothing fit: the best chunk was cut to the budget


def budget_for(full_tokens: int, ratio: float) -> int:
    return max(1, math.ceil(full_tokens / ratio))


def select_by_scores(doc: QADocument, scores: Sequence[float], lengths: Sequence[int], budget: int,
                     tokenizer) -> Selection:
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    kept, used = [], 0
    for i in order:
        if used + lengths[i] <= budget:
            kept.append(i)
            used += lengths[i]
    if kept:
        return Selection(sorted(kept), doc.text(kept), used, budget, sum(lengths))
    best = order[0]
    ids = tokenizer.encode(doc.chunks[best], add_special_tokens=False)[:budget]
    return Selection([best], tokenizer.decode(ids), len(ids), budget, sum(lengths), truncated=True)


# ---------------------------------------------------------------------------
# chunk scorers
# ---------------------------------------------------------------------------

def _terms(text: str) -> List[str]:
    text = unicodedata.normalize('NFC', text).lower()
    return re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE).split()


def bm25_scores(question: str, chunks: Sequence[str], k1: float = 1.5, b: float = 0.75) -> List[float]:
    """BM25 with IDF computed over the document's own chunks (syllable
    tokens for Vietnamese, words for English)."""
    docs = [_terms(c) for c in chunks]
    n = len(docs)
    avgdl = sum(len(d) for d in docs) / max(1, n)
    df = Counter(t for d in docs for t in set(d))
    q = _terms(question)
    scores = []
    for d in docs:
        tf = Counter(d)
        s = 0.0
        for t in q:
            if t not in tf:
                continue
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(d) / max(avgdl, 1e-9)))
        scores.append(s)
    return scores


class EmbeddingScorer:
    def __init__(self, model_name: str = 'BAAI/bge-m3', device: str = 'cuda', max_len: int = 1024):
        import torch
        from transformers import AutoModel, AutoTokenizer
        from .reader import resolve_device

        self.torch = torch
        self.device = resolve_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        from .pruner import effective_max_len
        self.max_len = effective_max_len(self.model.config, max_len)

    def _embed(self, texts: Sequence[str]):
        torch = self.torch
        out = []
        with torch.no_grad():
            for start in range(0, len(texts), 32):
                enc = self.tokenizer(list(texts[start:start + 32]), padding=True, truncation=True,
                                     max_length=self.max_len, return_tensors='pt').to(self.device)
                cls = self.model(**enc).last_hidden_state[:, 0]
                out.append(torch.nn.functional.normalize(cls.float(), dim=-1))
        return torch.cat(out)

    def score_chunks(self, question: str, chunks: Sequence[str]) -> List[float]:
        q = self._embed([question])
        return (self._embed(chunks) @ q[0]).cpu().tolist()


class ProvenceScorer:
    """Provence (naver/provence-reranker-debertav3-v1) / XProvence
    (naver/xprovence-reranker-bgem3-v1) used as a per-chunk reranker so it
    is budget-matched like every other chunk arm. Loaded with
    trust_remote_code; its `process()` API is the one documented on the
    model cards -- verify on the cluster before quoting numbers."""

    def __init__(self, model_id: str, device: str = 'cuda'):
        from transformers import AutoModel
        from .reader import resolve_device
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(resolve_device(device)).eval()

    def score_chunks(self, question: str, chunks: Sequence[str]) -> List[float]:
        out = self.model.process([question] * len(chunks), [[c] for c in chunks], always_select_title=False,
                                 enable_warnings=False)
        scores = out['reranking_score']
        return [float(s[0] if isinstance(s, (list, tuple)) else s) for s in scores]


class LLMLingua2Compressor:
    def __init__(self, model_name: str = 'microsoft/llmlingua-2-xlm-roberta-large-meetingbank', device: str = 'cuda'):
        from llmlingua import PromptCompressor
        from .reader import resolve_device
        self.compressor = PromptCompressor(model_name=model_name, use_llmlingua2=True, device_map=resolve_device(device))

    def compress(self, text: str, ratio: float) -> str:
        return self.compressor.compress_prompt(text, rate=1.0 / ratio, force_tokens=['\n'])['compressed_prompt']


# ---------------------------------------------------------------------------
# arm factory
# ---------------------------------------------------------------------------

class Arm:
    """name, kind ('chunk' | 'text' | 'full'); chunk arms implement scores(doc)."""

    def __init__(self, name: str, kind: str, scorer=None, text_compressor=None, table: Optional[Dict] = None):
        self.name, self.kind = name, kind
        self._scorer, self._text, self._table = scorer, text_compressor, table

    def scores(self, doc: QADocument) -> Optional[List[float]]:
        C = doc.num_chunks
        if self.name == 'lead':
            return [-float(i) for i in range(C)]
        if self.name == 'random':
            rng = random.Random(hash_seed(f'random-arm:{doc.doc_id}'))
            return [rng.random() for _ in range(C)]
        if self.name == 'bm25':
            return bm25_scores(doc.question, doc.chunks)
        if self.name in ('oracle_span', 'oracle_support'):
            gold = ({i for i, c in enumerate(doc.chunks) if contains_answer(c, doc.answers)}
                    if self.name == 'oracle_span' else set(doc.gold_chunks))
            # gold first; the rest in bm25 order so the oracle doesn't waste leftover budget
            rest = bm25_scores(doc.question, doc.chunks)
            top = max(rest, default=0.0) + 1.0
            return [top + 1.0 if i in gold else rest[i] for i in range(C)]
        if self.name == 'oracle_beta':
            beta = self._table.get(doc.doc_id) if self._table else None
            return list(beta) if beta is not None and len(beta) == C else None
        return self._scorer.score_chunks(doc.question, doc.chunks)

    def compress_text(self, doc: QADocument, ratio: float) -> str:
        return self._text.compress(doc.text(), ratio)


def load_beta_table(label_dir: str) -> Dict[str, List[float]]:
    from .attribution import load_chunk_labels, record_paths
    return {lab.doc['doc_id']: lab.beta for lab in (load_chunk_labels(p) for p in record_paths(label_dir))}


def make_arm(spec: str, device: str = 'cuda', oracle_beta_dir: Optional[str] = None) -> Arm:
    if spec == 'full':
        return Arm('full', 'full')
    if spec in ('lead', 'random', 'bm25', 'oracle_span', 'oracle_support'):
        return Arm(spec, 'chunk')
    if spec == 'oracle_beta':
        if not oracle_beta_dir:
            raise ValueError("oracle_beta needs --oracle-beta-dir (Stage A labels of the evaluated documents)")
        return Arm(spec, 'chunk', table=load_beta_table(oracle_beta_dir))
    if spec == 'embed' or spec.startswith('embed:'):
        model = spec.split(':', 1)[1] if ':' in spec else 'BAAI/bge-m3'
        return Arm(spec, 'chunk', scorer=EmbeddingScorer(model, device))
    if spec.startswith('pruner:'):
        from .pruner import PrunerScorer
        return Arm(spec, 'chunk', scorer=PrunerScorer(spec.split(':', 1)[1], device))
    if spec.startswith('provence:'):
        return Arm(spec, 'chunk', scorer=ProvenceScorer(spec.split(':', 1)[1], device))
    if spec == 'llmlingua2' or spec.startswith('llmlingua2:'):
        model = spec.split(':', 1)[1] if ':' in spec else 'microsoft/llmlingua-2-xlm-roberta-large-meetingbank'
        return Arm(spec, 'text', text_compressor=LLMLingua2Compressor(model, device))
    raise ValueError(f"unknown arm {spec!r}")
