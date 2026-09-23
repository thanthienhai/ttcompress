"""Readers: (context, question) -> answer, batched; plus the teacher-forced
log-probability of the gold answer (the AT2/ContextCite-style target, used
only for the F1-vs-logprob surrogate ablation).

Two backends behind one interface:
  - HFReader:   transformers, left-padded batched greedy decoding. Used by
                tests (tiny models) and for small runs.
  - VLLMReader: vLLM, for label generation at scale (tens of thousands of
                documents x ~64 masks). Imported lazily.

Lessons carried over from RUN_REPORT_2026-09-23.md, fixed here by design:
  - device strings: `resolve_device` accepts 'cuda', 'cuda:N', 'cpu'; the old
    `device == 'cuda'` check silently left the reader on CPU for 'cuda:N'.
    The recommended multi-GPU launch is one process per GPU (group) with
    CUDA_VISIBLE_DEVICES, so every process just uses 'cuda'.
  - generation budget: chat-tuned readers spend tokens on a preamble;
    budgets are per hop type and generous (32 / 48), never 6.
  - Qwen3 thinking mode is switched off through the chat template.
  - reader failures propagate instead of being scored as F1 = 0: a silently
    empty answer would be written into the labels as "this context is useless".
"""
from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple

PROMPTS = {
    'vi': ("Dựa vào ngữ cảnh dưới đây, hãy trả lời câu hỏi bằng tiếng Việt. Chỉ ghi đáp án ngắn gọn "
           "(một cụm từ), không giải thích.\n\n### Ngữ cảnh:\n{context}\n\n### Câu hỏi:\n{question}\n\n### Đáp án:"),
    'en': ("Answer the question based on the context below. Reply with the short answer only "
           "(a phrase), no explanation.\n\n### Context:\n{context}\n\n### Question:\n{question}\n\n### Answer:"),
}
MAX_NEW_TOKENS = {'single': 32, 'multi': 48}
_THINK_RE = re.compile(r'<think>.*?</think>', flags=re.DOTALL)


def reader_tag(model_name: str) -> str:
    """Filesystem-safe name for a reader, used as a directory level."""
    return model_name.strip('/').replace('/', '--')


def resolve_device(device: str) -> str:
    import torch
    if device.startswith('cuda'):
        return device if torch.cuda.is_available() else 'cpu'
    return device


# A base (non-chat) reader has no end-of-turn: after the answer it keeps writing the prompt's pattern
# ("### Câu hỏi: ..."). Its answer is the text up to the first of these markers.
BASE_STOP = ('\n', '###')


def clean_answer(text: Optional[str], first_line: bool = False) -> str:
    if not text:
        return ''
    text = _THINK_RE.sub('', text).replace('<think>', '').replace('</think>', '').strip()
    if first_line:
        for stop in BASE_STOP:
            text = text.split(stop, 1)[0]
    return text.strip()


def fit_ids(ids: List[int], limit: Optional[int]) -> Tuple[List[int], bool]:
    """(ids, truncated). An over-long prompt loses tokens from the MIDDLE, so
    the chat header + instruction (head) and the question + answer cue (tail)
    survive. Documents measure <= ~9k Qwen tokens, so with --max-model-len
    16384 this should never fire; Reader.n_truncated makes it visible if it does."""
    if not limit or len(ids) <= limit:
        return list(ids), False
    head = limit // 2
    return list(ids[:head]) + list(ids[len(ids) - (limit - head):]), True


class Reader:
    """Common prompt construction; subclasses implement generate/answer_logprob."""

    name: str
    tokenizer = None
    n_truncated = 0

    @property
    def is_chat(self) -> bool:
        """False for a base model (e.g. aisingapore/Llama-SEA-LION-v3-8B): plain
        prompt, answer cut at the first line (BASE_STOP)."""
        return bool(getattr(self.tokenizer, 'chat_template', None))

    def build_prompt(self, context: str, question: str, language: str) -> str:
        user = PROMPTS[language].format(context=context, question=question)
        tok = self.tokenizer
        if getattr(tok, 'chat_template', None):
            messages = [{'role': 'user', 'content': user}]
            try:
                return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                               enable_thinking=False)
            except TypeError:
                return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # base model: plain completion prompt; the BOS token is added here because prompts are
        # tokenized with add_special_tokens=False (chat templates already carry it)
        return (tok.bos_token or '') + user + ' '

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def generate(self, prompts: Sequence[str], max_new_tokens: int) -> List[str]:
        raise NotImplementedError

    def answer_logprob(self, prompts: Sequence[str], answers: Sequence[str]) -> List[float]:
        """Mean per-token log p(answer | prompt), teacher-forced."""
        raise NotImplementedError


class HFReader(Reader):
    def __init__(self, model_name: str, device: str = 'cuda', dtype: str = 'bfloat16', batch_size: int = 8,
                 max_prompt_tokens: Optional[int] = None):
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = model_name
        self.batch_size = batch_size
        self.max_prompt_tokens = max_prompt_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = resolve_device(device)
        torch_dtype = getattr(torch, dtype) if self.device != 'cpu' else torch.float32
        # `dtype=` replaced `torch_dtype=` in transformers 4.56; an older version would ignore the new
        # name and silently load an 8B reader in fp32
        major, minor = (int(re.match(r'\d+', x).group()) for x in transformers.__version__.split('.')[:2])
        dtype_kw = 'dtype' if (major, minor) >= (4, 56) else 'torch_dtype'
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True,
                                                          **{dtype_kw: torch_dtype})
        self.model.to(self.device).eval()

    def _encode(self, text: str, reserve: int = 0) -> List[int]:
        """`max_prompt_tokens` is the total context (prompt + `reserve`
        generated tokens), the same meaning as vLLM's max_model_len."""
        limit = self.max_prompt_tokens - reserve if self.max_prompt_tokens else None
        ids, truncated = fit_ids(self.tokenizer.encode(text, add_special_tokens=False), limit)
        self.n_truncated += truncated
        return ids

    def _left_pad(self, seqs: List[List[int]]):
        import torch
        width = max(len(s) for s in seqs)
        pad = self.tokenizer.pad_token_id
        ids = torch.full((len(seqs), width), pad, dtype=torch.long)
        mask = torch.zeros((len(seqs), width), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, width - len(s):] = torch.tensor(s, dtype=torch.long)
            mask[i, width - len(s):] = 1
        return ids.to(self.device), mask.to(self.device)

    def generate(self, prompts: Sequence[str], max_new_tokens: int) -> List[str]:
        """Greedy, batched. Prompts are sorted by length inside so a batch
        pads to similar lengths; output order matches input order."""
        import torch
        encoded = [self._encode(p, reserve=max_new_tokens) for p in prompts]
        order = sorted(range(len(prompts)), key=lambda i: len(encoded[i]))
        out: List[str] = [''] * len(prompts)
        for start in range(0, len(order), self.batch_size):
            idx = order[start:start + self.batch_size]
            ids, mask = self._left_pad([encoded[i] for i in idx])
            with torch.no_grad():
                gen = self.model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=max_new_tokens,
                                          do_sample=False, pad_token_id=self.tokenizer.pad_token_id)
            for row, i in enumerate(idx):
                out[i] = clean_answer(self.tokenizer.decode(gen[row, ids.shape[1]:], skip_special_tokens=True),
                                      first_line=not self.is_chat)
        return out

    def _tail_logits(self, ids, mask, position_ids, keep: int):
        """Logits of the last `keep` positions only; full logits would be
        seq x vocab (10k x 152k) per row, tens of GB for a batch."""
        try:
            return self.model(input_ids=ids, attention_mask=mask, position_ids=position_ids,
                              logits_to_keep=keep).logits
        except TypeError:  # transformers without logits_to_keep: run the body, project only the tail
            hidden = self.model.get_decoder()(input_ids=ids, attention_mask=mask,
                                              position_ids=position_ids).last_hidden_state
            return self.model.get_output_embeddings()(hidden[:, -keep:])

    def answer_logprob(self, prompts: Sequence[str], answers: Sequence[str]) -> List[float]:
        import torch
        pairs = []
        for p, a in zip(prompts, answers):
            ans = self.tokenizer.encode(a, add_special_tokens=False) or [self.tokenizer.eos_token_id]
            pairs.append((self._encode(p, reserve=len(ans)), ans))
        order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))
        out: List[float] = [float('nan')] * len(pairs)
        for start in range(0, len(order), self.batch_size):
            idx = order[start:start + self.batch_size]
            ids, mask = self._left_pad([pairs[i][0] + pairs[i][1] for i in idx])
            keep = max(len(pairs[i][1]) for i in idx) + 1
            position_ids = (mask.cumsum(-1) - 1).clamp_min(0)
            with torch.no_grad():
                logp = torch.log_softmax(self._tail_logits(ids, mask, position_ids, keep).float(), dim=-1)
            for row, i in enumerate(idx):
                ans = pairs[i][1]
                # logits at t predict token t+1; the answer occupies the last len(ans) positions
                preds = logp[row, keep - len(ans) - 1:keep - 1]
                tgt = torch.tensor(ans, device=preds.device)
                out[i] = float(preds.gather(-1, tgt.unsqueeze(-1)).mean())
        return out


class VLLMReader(Reader):
    """vLLM backend. One process per GPU group (CUDA_VISIBLE_DEVICES), with
    tensor_parallel_size = GPUs in the group (e.g. 2 for a 32B reader).
    Prompts are passed as token ids so truncation is exact."""

    def __init__(self, model_name: str, dtype: str = 'bfloat16', max_model_len: Optional[int] = None,
                 gpu_memory_utilization: float = 0.9, tensor_parallel_size: int = 1, seed: int = 0):
        from vllm import LLM

        self.name = model_name
        kwargs = {'model': model_name, 'dtype': dtype, 'trust_remote_code': True, 'seed': seed,
                  'gpu_memory_utilization': gpu_memory_utilization, 'tensor_parallel_size': tensor_parallel_size,
                  # older vLLM rejects prompt_logprobs together with prefix caching; masked contexts share
                  # little prefix anyway
                  'enable_prefix_caching': False}
        if max_model_len:
            kwargs['max_model_len'] = max_model_len
        self.llm = LLM(**kwargs)
        self.tokenizer = self.llm.get_tokenizer()
        config = getattr(self.llm, 'model_config', None) or self.llm.llm_engine.model_config
        self.max_model_len = config.max_model_len

    def _fit(self, prompt: str, reserve: int) -> List[int]:
        ids, truncated = fit_ids(self.tokenizer.encode(prompt, add_special_tokens=False),
                                 self.max_model_len - reserve)
        self.n_truncated += truncated
        return ids

    def generate(self, prompts: Sequence[str], max_new_tokens: int) -> List[str]:
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        # a base reader stops at the next prompt section; clean_answer then keeps its first line
        params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens,
                                stop=None if self.is_chat else ["###"])
        batch = [TokensPrompt(prompt_token_ids=self._fit(p, max_new_tokens)) for p in prompts]
        outs = self.llm.generate(batch, params, use_tqdm=False)
        return [clean_answer(o.outputs[0].text, first_line=not self.is_chat) for o in outs]

    def answer_logprob(self, prompts: Sequence[str], answers: Sequence[str]) -> List[float]:
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
        batch, spans = [], []
        for p, a in zip(prompts, answers):
            ans = self.tokenizer.encode(a, add_special_tokens=False) or [self.tokenizer.eos_token_id]
            ids = self._fit(p, len(ans) + 1)
            batch.append(TokensPrompt(prompt_token_ids=ids + ans))
            spans.append((len(ids), ans))
        outs = self.llm.generate(batch, params, use_tqdm=False)
        result = []
        for o, (start, ans) in zip(outs, spans):
            lps = [o.prompt_logprobs[start + j][tok].logprob for j, tok in enumerate(ans)]
            result.append(sum(lps) / len(lps))
        return result


def load_reader(model_name: str, backend: str = 'hf', device: str = 'cuda', dtype: str = 'bfloat16',
                batch_size: int = 8, max_model_len: Optional[int] = None, tensor_parallel_size: int = 1,
                gpu_memory_utilization: float = 0.9) -> Reader:
    if backend == 'hf':
        return HFReader(model_name, device=device, dtype=dtype, batch_size=batch_size, max_prompt_tokens=max_model_len)
    if backend == 'vllm':
        return VLLMReader(model_name, dtype=dtype, max_model_len=max_model_len,
                          gpu_memory_utilization=gpu_memory_utilization, tensor_parallel_size=tensor_parallel_size)
    raise ValueError(f"backend must be 'hf' or 'vllm'; got {backend!r}")
