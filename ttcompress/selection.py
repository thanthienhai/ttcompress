"""Budget-aware selection and every compression arm evaluate.py runs.

A fixed token budget (reader tokens) is the only fair comparison: every arm
gets budget = ceil(full_tokens / ratio) for the same document and ratio.
Chunk arms return per-chunk scores; `select_by_scores` keeps the best chunks
that fit (greedy by score, original order preserved). If not even the best
chunk fits, it is kept truncated to the budget, so no arm ever gets an empty
context. Sentence arms (RECOMP, EXIT) score sentences and are selected the
same way over sentences (`select_sentences`). Text arms (LLMLingua family)
compress the string themselves at rate = 1/ratio; their realized token count
is recorded, never assumed.

Arms (name -> what it is):
  full             no compression (reference ceiling, ratio ignored)
  lead             first chunks in document order (truncation baseline)
  random           random chunk order (seeded by doc id)
  bm25             lexical BM25 of question vs chunk
  embed            dense cosine, BAAI/bge-m3 CLS embeddings
  reranker[:<id>]  zero-shot cross-encoder, default BAAI/bge-reranker-v2-m3 (the pruner's
                   untrained backbone: what the attribution labels add over initialization)
  oracle_span      chunks containing a gold answer string first (answer-span oracle, RQ2)
  oracle_support   gold chunks (needle / supporting facts) first
  oracle_beta      Stage A attribution of the test document (upper bound; needs --oracle-beta-dir)
  pruner:<path>    our distilled pruner (any number of checkpoints)
  provence:<hf_id> Provence / XProvence reranking score per chunk (naver/...)
  recomp[:<hf_id>] RECOMP extractive compressor, sentence-level (NQ / HotpotQA checkpoint by hop)
  exit[:<adapter>] EXIT sentence classifier (Gemma-2B-it + LoRA), sentence-level
  llmlingua[:<lm>]      LLMLingua token-level compression (needs transformers<=4.47.1)
  longllmlingua[:<lm>]  LongLLMLingua, question-aware (needs transformers<=4.47.1)
  llmlingua2       LLMLingua-2 token-level compression
Not included, because their output length cannot be set to a budget: RECOMP abstractive, CompAct,
CORE-RAG (no released checkpoint). Selective Context is English/Chinese-only and superseded by the
LLMLingua family.
"""
from __future__ import annotations

import math
import random
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .data import CHUNK_SEP, QADocument
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
    """Greedy by score; a chunk costs its tokens plus the separator before it. The joined text is recounted
    (tokens can merge across a join) and the weakest kept chunk dropped until it fits, so `kept_tokens` is
    what the reader gets and never exceeds the budget."""
    count = lambda text: len(tokenizer.encode(text, add_special_tokens=False))  # noqa: E731
    sep = count(CHUNK_SEP)
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    kept, used = [], 0
    for i in order:
        cost = lengths[i] + (sep if kept else 0)
        if used + cost <= budget:
            kept.append(i)
            used += cost
    while kept:
        text = doc.text(kept)
        n = count(text)
        if n <= budget:
            return Selection(sorted(kept), text, n, budget, sum(lengths))
        kept.pop()  # `kept` is in score order: drop the weakest
    best = order[0]
    ids = tokenizer.encode(doc.chunks[best], add_special_tokens=False)[:budget]
    return Selection([best], tokenizer.decode(ids), len(ids), budget, sum(lengths), truncated=True)


# ---------------------------------------------------------------------------
# sentence units (RECOMP / EXIT select sentences, not chunks)
# ---------------------------------------------------------------------------

_SENT_END = re.compile(r'(?<=[.!?…])\s+(?=\S)')


def split_sentences(text: str) -> List[str]:
    """Rule-based splitter (terminal punctuation or newline), language-agnostic so Vietnamese and
    English documents are cut the same way; every sentence arm shares it."""
    return [s.strip() for line in text.split('\n') for s in _SENT_END.split(line) if s.strip()]


def chunk_title(doc: QADocument, i: int) -> str:
    """Multi-hop chunks are 'Title\\nparagraph' (sources.multihop_row_to_doc); the title is kept once per
    touched chunk so a kept sentence never loses the entity it is about."""
    chunk = doc.chunks[i]
    return chunk.split('\n', 1)[0].strip() if doc.hop == 'multi' and '\n' in chunk else ''


@dataclass
class SentenceUnit:
    chunk: int
    text: str


def sentence_units(doc: QADocument) -> List[SentenceUnit]:
    units = []
    for i, chunk in enumerate(doc.chunks):
        body = chunk.split('\n', 1)[1] if chunk_title(doc, i) else chunk
        units.extend(SentenceUnit(i, s) for s in split_sentences(body))
    return units


def units_text(doc: QADocument, units: Sequence[SentenceUnit], keep: Sequence[int]) -> str:
    by_chunk: Dict[int, List[str]] = {}
    for j in sorted(set(keep)):
        by_chunk.setdefault(units[j].chunk, []).append(units[j].text)
    parts = []
    for i in sorted(by_chunk):
        title = chunk_title(doc, i)
        body = ' '.join(by_chunk[i])
        parts.append(f'{title}\n{body}' if title else body)
    return '\n\n'.join(parts)


def select_sentences(doc: QADocument, units: Sequence[SentenceUnit], scores: Sequence[float], count,
                     budget: int, tokenizer) -> Selection:
    """Greedy by score under the budget, like select_by_scores, but over sentences; kept sentences are
    re-assembled in original order (title + sentences per chunk). The cost of a sentence is its own
    tokens plus its chunk's title the first time the chunk is touched; the assembled text is recounted
    and the weakest kept sentence dropped until it fits, so separators never push it over the budget.
    `kept` lists the touched chunks."""
    lengths = [count(u.text) for u in units]
    title_len = {i: count(chunk_title(doc, i)) for i in {u.chunk for u in units}}
    order = sorted(range(len(units)), key=lambda j: (-scores[j], j))
    keep, opened, used = [], set(), 0
    for j in order:
        c = units[j].chunk
        cost = lengths[j] + (0 if c in opened else title_len[c])
        if used + cost <= budget:
            keep.append(j)
            opened.add(c)
            used += cost
    full_tokens = count(doc.text())
    while keep:
        text = units_text(doc, units, keep)
        n = count(text)
        if n <= budget:
            return Selection(sorted({units[j].chunk for j in keep}), text, n, budget, full_tokens)
        keep.pop()  # `keep` is in score order: drop the weakest
    if not units:  # no sentence at all (empty document): fall back to the first chunk, cut
        units, order = [SentenceUnit(0, doc.chunks[0])], [0]
    best = order[0]
    ids = tokenizer.encode(units_text(doc, units, [best]), add_special_tokens=False)[:budget]
    return Selection([units[best].chunk], tokenizer.decode(ids), len(ids), budget, full_tokens, truncated=True)


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


class RerankerScorer:
    """Zero-shot cross-encoder (question, chunk) relevance with the model's own
    classification head -- by default the pruner's backbone
    (BAAI/bge-reranker-v2-m3) before any attribution training, so ours vs
    this arm isolates what the distilled labels add over initialization.
    Each chunk is scored on its own (no late chunking across the window)."""

    def __init__(self, model_name: str = 'BAAI/bge-reranker-v2-m3', device: str = 'cuda', max_len: int = 512,
                 batch_size: int = 32):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from .reader import resolve_device

        self.torch = torch
        self.device = resolve_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device).eval()
        if self.device != 'cpu':
            self.model.to(torch.bfloat16)
        from .pruner import effective_max_len
        self.max_len = effective_max_len(self.model.config, max_len)
        self.batch_size = batch_size

    def score_chunks(self, question: str, chunks: Sequence[str]) -> List[float]:
        torch = self.torch
        out = []
        with torch.no_grad():
            for start in range(0, len(chunks), self.batch_size):
                batch = list(chunks[start:start + self.batch_size])
                enc = self.tokenizer([question] * len(batch), batch, padding=True, truncation='only_second',
                                     max_length=self.max_len, return_tensors='pt').to(self.device)
                out.extend(self.model(**enc).logits[:, 0].float().cpu().tolist())
        return out


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


class RecompScorer:
    """RECOMP extractive compressor (Xu et al., 2024; carriex/recomp run_extractive_compressor.py): a
    Contriever-initialised encoder shared by question and sentence, mean pooling over the attention mask,
    raw dot product (no normalisation, no prefix). The released checkpoints are task-specific, so by
    default single-hop documents use the NQ one and multi-hop the HotpotQA one; `recomp:<hf_id>` forces
    one. The paper keeps the top 1-2 sentences; here sentences are ranked under the shared budget.
    English uncased vocabulary: Vietnamese is scored zero-shot."""

    DEFAULT = {'single': 'fangyuan/nq_extractive_compressor', 'multi': 'fangyuan/hotpotqa_extractive_compressor'}

    def __init__(self, model_name: Optional[str] = None, device: str = 'cuda', max_len: int = 512,
                 batch_size: int = 64):
        import torch
        from .reader import resolve_device
        self.torch, self.device = torch, resolve_device(device)
        self.model_name, self.max_len, self.batch_size = model_name, max_len, batch_size
        self._loaded: Dict[str, tuple] = {}

    def _model(self, name: str):
        if name not in self._loaded:
            from transformers import AutoModel, AutoTokenizer
            self._loaded[name] = (AutoTokenizer.from_pretrained(name),
                                  AutoModel.from_pretrained(name).to(self.device).eval())
        return self._loaded[name]

    def _embed(self, name: str, texts: Sequence[str]):
        torch = self.torch
        tok, model = self._model(name)
        out = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                enc = tok(list(texts[start:start + self.batch_size]), padding=True, truncation=True,
                          max_length=self.max_len, return_tensors='pt').to(self.device)
                hidden = model(**enc).last_hidden_state.float()
                mask = enc['attention_mask'][..., None].float()
                out.append((hidden * mask).sum(dim=1) / mask.sum(dim=1))
        return torch.cat(out)

    def score_sentences(self, doc: QADocument, units: Sequence[SentenceUnit]) -> List[float]:
        name = self.model_name or self.DEFAULT[doc.hop]
        q = self._embed(name, [doc.question])[0]
        return (self._embed(name, [u.text for u in units]) @ q).cpu().tolist()


class ExitScorer:
    """EXIT (Hwang et al., 2025; ThisIsHwang/EXIT compressors/baselines/exit/{core,compressor}.py):
    Gemma-2B-it with the released LoRA adapter reads the query, the sentence's containing document and the
    sentence, and the score is P(Yes) from a softmax over the "Yes"/"No" logits at the next position. The
    paper keeps sentences with P(Yes) >= 0.5; here they are ranked under the shared budget. bf16 instead
    of the paper's 4-bit NF4 (a quantisation choice for speed, not part of the method). Explicit position
    ids, so left padding does not shift a prompt's positions. English-only model: Vietnamese is scored
    zero-shot. The base model is gated (accept the Gemma license on huggingface.co)."""

    PROMPT = ('<start_of_turn>user\nQuery:\n{query}\nFull context:\n{context}\nSentence:\n{sentence}\n'
              'Is this sentence useful in answering the query? Answer only "Yes" or "No".<end_of_turn>\n'
              '<start_of_turn>model\n')
    DEFAULT_ADAPTER = 'doubleyyh/exit-gemma-2b'
    BASE = 'google/gemma-2b-it'

    def __init__(self, adapter: Optional[str] = DEFAULT_ADAPTER, base: str = BASE, device: str = 'cuda',
                 batch_size: int = 16):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from .reader import dtype_kwargs, resolve_device
        self.torch, self.device, self.batch_size = torch, resolve_device(device), batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(base)
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.bfloat16 if self.device != 'cpu' else torch.float32
        model = AutoModelForCausalLM.from_pretrained(base, **dtype_kwargs(dtype))
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
        self.model = model.to(self.device).eval()
        self.yes_id = self.tokenizer.encode('Yes', add_special_tokens=False)[0]
        self.no_id = self.tokenizer.encode('No', add_special_tokens=False)[0]

    def score_sentences(self, doc: QADocument, units: Sequence[SentenceUnit]) -> List[float]:
        torch = self.torch
        prompts = [self.PROMPT.format(query=doc.question, context=doc.chunks[u.chunk], sentence=u.text)
                   for u in units]
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))  # similar lengths per batch
        scores = [0.0] * len(prompts)
        with torch.no_grad():
            for start in range(0, len(order), self.batch_size):
                idx = order[start:start + self.batch_size]
                enc = self.tokenizer([prompts[i] for i in idx], padding=True, return_tensors='pt').to(self.device)
                pos = (enc['attention_mask'].cumsum(-1) - 1).clamp(min=0)
                logits = self.model(**enc, position_ids=pos).logits[:, -1, :]
                p_yes = torch.softmax(logits[:, [self.yes_id, self.no_id]].float(), dim=-1)[:, 0]
                for i, p in zip(idx, p_yes.cpu().tolist()):
                    scores[i] = p
        return scores


LLMLINGUA_MAX_TRANSFORMERS = '4.47.1'


class LLMLinguaCompressor:
    """LLMLingua (Jiang et al., 2023) and question-aware LongLLMLingua (Jiang et al., 2024) through the
    `llmlingua` package (0.2.2), LongLLMLingua with the settings its README recommends. The chunks are
    passed as the package's document list; the question conditions LongLLMLingua but is not part of the
    returned text. Both need `transformers<=4.47.1`: their perplexity loop feeds past_key_values back as
    legacy tuples, which later versions reject (microsoft/LLMLingua#210). run_pipeline.sh runs these arms
    with such a transformers on PYTHONPATH. The small LM defaults to Qwen2.5-7B-Instruct instead of the
    package's English Llama-2-7B, so the Vietnamese sources get a multilingual compressor."""

    DEFAULT_MODEL = 'Qwen/Qwen2.5-7B-Instruct'

    def __init__(self, model_name: str = DEFAULT_MODEL, long: bool = False, device: str = 'cuda'):
        import transformers
        from packaging.version import Version
        if Version(transformers.__version__) > Version(LLMLINGUA_MAX_TRANSFORMERS):
            raise RuntimeError(f"llmlingua/longllmlingua arms need transformers<={LLMLINGUA_MAX_TRANSFORMERS} "
                               f"(found {transformers.__version__}); run_pipeline.sh installs one on PYTHONPATH "
                               f"for these arms (see LLMLINGUA_SITE)")
        from llmlingua import PromptCompressor
        from .reader import resolve_device
        self.long = long
        self.compressor = PromptCompressor(model_name=model_name, device_map=resolve_device(device))

    def compress(self, doc: QADocument, ratio: float) -> str:
        kwargs = dict(instruction='', question='', rate=1.0 / ratio)
        if self.long:
            kwargs.update(question=doc.question, concate_question=False, rank_method='longllmlingua',
                          condition_in_question='after_condition', reorder_context='sort',
                          dynamic_context_compression_ratio=0.3, condition_compare=True, context_budget='+100')
        return self.compressor.compress_prompt(list(doc.chunks), **kwargs)['compressed_prompt']


LLMLINGUA2_DEFAULT = 'microsoft/llmlingua-2-xlm-roberta-large-meetingbank'


class LLMLingua2Compressor:
    def __init__(self, model_name: str = LLMLINGUA2_DEFAULT, device: str = 'cuda'):
        from llmlingua import PromptCompressor
        from .reader import resolve_device
        self.compressor = PromptCompressor(model_name=model_name, use_llmlingua2=True, device_map=resolve_device(device))

    def compress(self, doc: QADocument, ratio: float) -> str:
        return self.compressor.compress_prompt(doc.text(), rate=1.0 / ratio, force_tokens=['\n'])['compressed_prompt']


# ---------------------------------------------------------------------------
# arm factory
# ---------------------------------------------------------------------------

class Arm:
    """name, kind ('chunk' | 'sentence' | 'text' | 'full'); chunk arms implement scores(doc), sentence
    arms sentence_scores(doc), text arms compress_text(doc, ratio)."""

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

    def sentence_scores(self, doc: QADocument):
        units = sentence_units(doc)
        return units, (self._scorer.score_sentences(doc, units) if units else [])

    def compress_text(self, doc: QADocument, ratio: float) -> str:
        return self._text.compress(doc, ratio)


def load_beta_table(label_dirs: str) -> Dict[str, List[float]]:
    """doc_id -> beta over one fit dir or a comma list of them (one per source)."""
    from .attribution import load_chunk_labels, record_paths
    paths = [p for d in label_dirs.split(',') if d for p in record_paths(d)]
    return {lab.doc['doc_id']: lab.beta for lab in (load_chunk_labels(p) for p in paths)}


def arm_models(spec: str) -> List[str]:
    """Hub ids an arm downloads, so scripts/prefetch.py can fetch them once before the shards start
    (and a gated one fails in the first minute)."""
    name, _, arg = spec.partition(':')
    defaults = {'embed': ['BAAI/bge-m3'], 'reranker': ['BAAI/bge-reranker-v2-m3'], 'llmlingua2': [LLMLINGUA2_DEFAULT],
                'llmlingua': [LLMLinguaCompressor.DEFAULT_MODEL], 'longllmlingua': [LLMLinguaCompressor.DEFAULT_MODEL],
                'recomp': sorted(RecompScorer.DEFAULT.values()), 'provence': []}
    if name == 'exit':
        return [ExitScorer.BASE, arg or ExitScorer.DEFAULT_ADAPTER]
    if name in defaults:
        return [arg] if arg else defaults[name]
    return []


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
    if spec == 'reranker' or spec.startswith('reranker:'):
        model = spec.split(':', 1)[1] if ':' in spec else 'BAAI/bge-reranker-v2-m3'
        return Arm(spec, 'chunk', scorer=RerankerScorer(model, device))
    if spec.startswith('pruner:'):
        from .pruner import PrunerScorer
        return Arm(spec, 'chunk', scorer=PrunerScorer(spec.split(':', 1)[1], device))
    if spec.startswith('provence:'):
        return Arm(spec, 'chunk', scorer=ProvenceScorer(spec.split(':', 1)[1], device))
    if spec == 'llmlingua2' or spec.startswith('llmlingua2:'):
        model = spec.split(':', 1)[1] if ':' in spec else LLMLINGUA2_DEFAULT
        return Arm(spec, 'text', text_compressor=LLMLingua2Compressor(model, device))
    for name, long in (('llmlingua', False), ('longllmlingua', True)):
        if spec == name or spec.startswith(name + ':'):
            model = spec.split(':', 1)[1] if ':' in spec else LLMLinguaCompressor.DEFAULT_MODEL
            return Arm(spec, 'text', text_compressor=LLMLinguaCompressor(model, long, device))
    if spec == 'recomp' or spec.startswith('recomp:'):
        return Arm(spec, 'sentence', scorer=RecompScorer(spec.split(':', 1)[1] if ':' in spec else None, device))
    if spec == 'exit' or spec.startswith('exit:'):
        adapter = spec.split(':', 1)[1] if ':' in spec else ExitScorer.DEFAULT_ADAPTER
        return Arm(spec, 'sentence', scorer=ExitScorer(adapter, device=device))
    raise ValueError(f"unknown arm {spec!r}")
