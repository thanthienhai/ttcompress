"""Reader (generation) call: compressed context + query -> answer text.

Trimmed from vncompress/vncompress/evaluation.py::VCCBench._build_prompt_ids /
_default_generate -- one sample at a time (no batching), which is enough for
an academic-scope bench and keeps this file short. Batching is a pure
performance optimization the original left as an opt-in
(generation_batch_size), not something PCS's correctness depends on.

PROMPT_TEMPLATES is keyed by `sample.metadata['dataset_source']` (see
public_datasets.py) so each source gets the instruction its own paper uses --
critically, LongBench's official template asks for the answer in the exact
"Paragraph N" format its gold answers use
(THUDM/LongBench, LongBench/config/dataset2prompt.json, copied verbatim). Not
using it would make token_f1 near-zero for every arm on that source alike,
drowning out the compression signal for a third of the data.

Most of this file's data sources (LongBench, RULER, classic NIAH,
InfiniteBench passkey -- see data/SOURCES.md) are English text, hence
PROMPT_TEMPLATES defaulting to English instructions. The 'uit_viquad' entry
is the exception: Stage A of OUTCOME_SUPERVISED_RELEVANCE_SPEC.md builds
documents from UIT-ViQuAD 2.0 (Vietnamese) and needs a Vietnamese instruction
so the reader answers in Vietnamese rather than switching languages
mid-generation.
"""
from __future__ import annotations

from typing import List, Optional

import torch

# Fallback for any source without its own entry below (currently: Kamradt --
# its needle/question are short and generic phrasing is enough).
PROMPT_TEMPLATES = {
    'generic': (
        "Based on the context below, answer the question concisely and precisely.\n\n"
        "### Context:\n{context}\n\n### Question:\n{query}\n\n### Answer:"
    ),
    # Copied verbatim from THUDM/LongBench, LongBench/config/dataset2prompt.json
    # ["passage_retrieval_en"]. {query} here is LongBench's own {input} field
    # (the abstract to match to a paragraph).
    'longbench_passage_retrieval_en': (
        "Here are 30 paragraphs from Wikipedia, along with an abstract. Please "
        "determine which paragraph the abstract is from.\n\n{context}\n\n"
        "The following is an abstract.\n\n{query}\n\n"
        "Please enter the number of the paragraph that the abstract is from. "
        "The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\n"
        "The answer is:"
    ),
    # Framing sentence copied verbatim from NVIDIA/RULER
    # scripts/data/synthetic/constants.py TASKS['niah']['template'], singular
    # form (niah_single_1: one needle). {query} is the question text
    # public_datasets.generate_ruler_niah already built.
    'ruler_niah_single_1': (
        "A special magic number is hidden within the following text. Make sure "
        "to memorize it. I will quiz you about the number afterwards.\n"
        "{context}\n{query}"
    ),
    # Copied verbatim from OpenBMB/InfiniteBench, src/prompt.py
    # gpt4_templates['passkey'] ("{context}\n\n{input}" with this instruction
    # prefix). {query} is InfiniteBench's own {input} field ("What is the
    # pass key?"), unmodified -- public_datasets.generate_infinitebench_passkey
    # re-scales/re-randomizes the context but not this question.
    'infinitebench_passkey': (
        "There is an important info hidden inside a lot of irrelevant text. "
        "Find it and memorize them. I will quiz you about the important "
        "information there.\n\n{context}\n\n{query}"
    ),
    # OUTCOME_SUPERVISED_RELEVANCE_SPEC.md Stage A (§2.4): context is a
    # constructed Vietnamese needle+haystack document from UIT-ViQuAD 2.0,
    # query is the original UIT-ViQuAD question.
    'uit_viquad': (
        "Dựa vào ngữ cảnh dưới đây, hãy trả lời câu hỏi ngắn gọn và chính xác "
        "bằng tiếng Việt.\n\n### Ngữ cảnh:\n{context}\n\n### Câu hỏi:\n{query}\n\n### Đáp án:"
    ),
}
# xquad_vi (ttcompress/xquad_vi.py) and vimqa (ttcompress/vimqa.py) are also
# monolingual Vietnamese content -- same template as uit_viquad, not the
# (English) 'generic' fallback. A dataset_source with no PROMPT_TEMPLATES
# entry silently gets PROMPT_TEMPLATES['generic'] (English) instead of
# erroring -- this exact class of bug was caught TWICE already (first on a
# since-removed source, then again on xquad_vi when it was added); adding
# every new Vietnamese source's alias here, and to VIETNAMESE_SOURCES in
# tests/test_reader_prompts.py, is what prevents a third repeat.
PROMPT_TEMPLATES['xquad_vi'] = PROMPT_TEMPLATES['uit_viquad']
PROMPT_TEMPLATES['vimqa'] = PROMPT_TEMPLATES['uit_viquad']

# Copied from THUDM/LongBench LongBench/config/dataset2maxlen.json
# ["passage_retrieval_en"] = 32; RULER's constants.py TASKS['niah']
# ['tokens_to_generate'] = 128; Kamradt's expected answer is a short phrase.
# uit_viquad/xquad_vi answers are short extractive spans (SQuAD-style) -- same
# ballpark. infinitebench_passkey = 6, copied verbatim from
# OpenBMB/InfiniteBench src/eval_utils.py DATA_NAME_TO_MAX_NEW_TOKENS['passkey'].
# vimqa = 48, not 32: measured on the real validation split's word counts
# (multi-hop bridge answers run longer than SQuAD-style spans -- p95 is 20
# words, max 59), not guessed.
DEFAULT_MAX_NEW_TOKENS = {
    'longbench_passage_retrieval_en': 32,
    'ruler_niah_single_1': 128,
    'kamradt_niah': 32,
    'uit_viquad': 32,
    'xquad_vi': 32,
    'infinitebench_passkey': 6,
    'vimqa': 48,
}


def default_max_new_tokens(dataset_source: Optional[str]) -> int:
    return DEFAULT_MAX_NEW_TOKENS.get(dataset_source, 64)


def build_prompt_ids(tokenizer, compressed_ids: List[int], query: str, prompt_kind: str = 'generic') -> List[int]:
    """'chat' style when the tokenizer has a chat template (falls back to a
    bare concatenation otherwise, so non-instruct readers still run)."""
    if getattr(tokenizer, 'chat_template', None):
        context = tokenizer.decode(compressed_ids, skip_special_tokens=True)
        template = PROMPT_TEMPLATES.get(prompt_kind, PROMPT_TEMPLATES['generic'])
        messages = [{'role': 'user', 'content': template.format(context=context, query=query)}]
        try:
            encoded = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        return encoded['input_ids'] if hasattr(encoded, 'keys') else encoded
    return compressed_ids + tokenizer.encode(query, add_special_tokens=False)


def generate_answer(
    model, tokenizer, compressed_ids: List[int], query: str,
    max_new_tokens: int = 64, prompt_kind: str = 'generic',
) -> Optional[str]:
    """Greedy decode. Returns None (not raise) on failure -- a broken
    generation must score 0 downstream, not crash the whole sweep."""
    try:
        prompt_ids = build_prompt_ids(tokenizer, compressed_ids, query, prompt_kind=prompt_kind)
        input_tensor = torch.tensor([prompt_ids]).to(model.device)
        with torch.no_grad():
            out = model.generate(
                input_tensor, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        return tokenizer.decode(out[0][len(prompt_ids):], skip_special_tokens=True)
    except Exception as exc:
        print(f"    [WARN] generation failed for query={query!r}: {type(exc).__name__}: {exc}")
        return None


def load_reader(model_name: str, device: str = 'cuda', dtype: str = 'float16', attn_implementation: Optional[str] = None):
    """Copied from vncompress/vncompress/models.py::load_model -- no
    device_map, load on CPU then .to(device).

    attn_implementation='eager' is required for the h2o/snapkv arms
    (compression.SnapKVCompressor reads real attention weights via
    output_attentions=True) -- sdpa/flash_attention_2, which many models pick
    by default, DO NOT populate outputs.attentions and that arm will raise
    RuntimeError. evaluate.py sets this automatically when h2o/snapkv are in
    --arms; pass it explicitly for any other direct use of SnapKVCompressor.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    torch_dtype = getattr(torch, dtype)
    kwargs = {'trust_remote_code': True, 'torch_dtype': torch_dtype}
    if attn_implementation:
        kwargs['attn_implementation'] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if device == 'cuda' and torch.cuda.is_available():
        model = model.to('cuda')
    model.eval()
    return model, tokenizer
