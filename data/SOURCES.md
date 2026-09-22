# Data sources

Every dataset this project uses is public and citable -- not self-built data,
and (since the OUTCOME_SUPERVISED_RELEVANCE_SPEC.md pipeline dropped
vcc_bench_v2.json, an unpublished internal vncompress benchmark, in favor of
three more public Vietnamese/English sources) that now holds across both
pipelines, not just PCS_METHOD_SPEC.md's. §§1-5 below are the English
sources (`metadata['dataset_source']` tags each sample); §§6-8 are the
outcome-supervised pipeline's three Vietnamese sources.

**Neither generated file under §§1-4 is committed to git** — `data/*.json`
and `data/*.jsonl` are gitignored. Run `scripts/fetch_data.sh` to
(re)download them; this file is the only thing in `data/` that's tracked.
§5 and §§6-8 aren't local files at all -- `ttcompress/public_datasets.py`,
`ttcompress/uit_viquad.py`, `ttcompress/xquad_vi.py` and `ttcompress/vimqa.py`
all pull them straight from HuggingFace at runtime (cached under the
standard HF cache dir, not under `data/`).

## 1. `paul_graham_essays.json` — haystack corpus for RULER + classic NIAH

49 plain-text Paul Graham essays, fetched 2026-09-22 directly from the exact
URL list RULER's own data generator uses
(`NVIDIA/RULER, scripts/data/synthetic/json/PaulGrahamEssays_URLs.txt`, the
non-HTML subset, which RULER itself sources from
`gkamradt/LLMTest_NeedleInAHaystack` — i.e. this is the same corpus both
RULER's `niah_single_1` and the original "Needle In A Haystack" test use).
Paul Graham's site: "You're welcome to do whatever you want with this text."
`LLMTest_NeedleInAHaystack` is MIT-licensed.

Used by `ttcompress/public_datasets.py::generate_ruler_niah` and
`::generate_kamradt_niah` as the haystack text.

## 2. `longbench_passage_retrieval_en.jsonl` — LongBench (THUDM, ACL 2024)

200 samples, downloaded 2026-09-22 from `data.zip` in the
`THUDM/LongBench` HuggingFace dataset repo (MIT license). Real needle-style
retrieval task: a long context of ~30 Wikipedia paragraphs, a query that
paraphrases ONE paragraph, and the gold answer is which paragraph it is
(`"Paragraph N"`). This is the exact split LongLLMLingua/LLMLingua-2 and
related compression papers report on.

Loaded as-is by `ttcompress/public_datasets.py::load_longbench_passage_retrieval`.

## 3. RULER `niah_single_1` (generated) — Hsieh et al. 2024, NeurIPS

Reproduces RULER's needle template and query template verbatim from
`NVIDIA/RULER, scripts/data/synthetic/{niah.py,constants.py}` (Apache-2.0):
a "magic number" needle for a random key, inserted into the essay haystack,
one question per sample. Simplification vs. the original: haystack size is
picked by word count to hit a target context length, not RULER's exact
tokenizer-driven binary search — everything else (needle/query strings,
essay source) matches.

## 4. Classic Needle-In-A-Haystack (generated) — Kamradt

Reproduces the original single-needle recipe verbatim from
`gkamradt/LLMTest_NeedleInAHaystack, needlehaystack/tasks/single_needle.py`
(MIT): the fixed "best thing to do in San Francisco... eat a sandwich and
sit in Dolores Park" sentence, inserted into the essay haystack at a given
depth percent, one fixed question.

## 5. InfiniteBench `passkey` (Zhang et al. 2024, ACL) — real needle values, rescaled

`xinrongzhang2022/InfiniteBench` on HuggingFace (`passkey.jsonl`, 590 rows,
verified 2026-09-22, Apache-2.0-style "freely available" per the paper). Each
row's real passkey needle sentence (`"The pass key is N. Remember it. N is
the pass key."`) is extracted from the original context via regex (verified
590/590 matches, extracted N always equals the row's `answer[0]`) and
re-inserted into a freshly built ~750-word haystack of InfiniteBench's own
verbatim "noise" filler text, at a random depth (`random.Random` per sample) --
**not used as published**. Two real deviations, both because the original is
unusable at this pipeline's pilot scale:
- The original context is a constant ~470,000 chars (~120K+ tokens) per
  sample, meant for extreme-length retrieval. Stage A's ridge regression
  (`estimate_chunk_contributions`) requires at least as many masks K as
  chunks C; at the pilot scale (K=15) that context would need thousands of
  chunks, infeasible.
- The needle in the original is always placed within the first ~7% of the
  text (verified: relative depth 0.0-0.069 across 50 samples checked) --
  using it as-is would hand `g(i,L)` (PCS's position score, maximal near
  position 0) a trivial built-in win baked into the source data, not a
  result. Re-randomizing depth matches RULER/Kamradt's own random-depth
  convention (§§3-4).

`tokens_to_generate`/prompt instruction copied verbatim from
`OpenBMB/InfiniteBench, src/eval_utils.py::DATA_NAME_TO_MAX_NEW_TOKENS['passkey']`
(= 6) and `src/prompt.py::gpt4_templates['passkey']`.

Loaded by `ttcompress/public_datasets.py::generate_infinitebench_passkey` /
`::split_infinitebench_passkey`.

## 6. UIT-ViQuAD 2.0 — Nguyen et al., COLING 2020

`taidng/UIT-ViQuAD2.0` on HuggingFace (plain parquet, no loading script,
verified 2026-09-22). Train=28454 / Dev(validation)=3814 / Test=7301 --
the official Test split ships with **no gold answers at all** (verified: all
7301 rows have `answers=None`, the standard withheld-answer convention for a
public leaderboard test set) and is unusable here; see
`ttcompress/uit_viquad.py::split_dev_tune_test` for how a genuinely held-out,
answerable test portion is instead carved out of the Dev split.
**License unresolved**: the HF mirror carries no license field at all, only
"freely available to encourage the research community" (the paper's own
abstract) — confirm with the UIT NLP group (`nlp.uit.edu.vn/datasets`,
`vlsp.org.vn/vlsp2021/eval/mrc`) before any publication.

Loaded by `ttcompress/uit_viquad.py`; documents built from it via
`ttcompress/build_document.py`, replicating the same needle+haystack
construction vncompress's own internal `needle_in_haystack` task uses
(confirmed by reading `vncompress/scripts/build_vcc_bench_v2.py`).

## 7. XQuAD Vietnamese (`xquad.vi`) — Artetxe, Ruder & Yogatama, ACL 2020

`xquad` on HuggingFace, config `xquad.vi` (plain parquet, verified
2026-09-22): the SQuAD 1.1 dev set (240 paragraphs, 1190 questions)
professionally translated into Vietnamese, one of 11 XQuAD languages. Used
purely as an evaluation benchmark in the literature (it *is* a translated
SQuAD dev set) -- no official train split, so
`ttcompress/xquad_vi.py::split_dev_test` carves its own small dev/test
division, mirroring `load_longbench_passage_retrieval`'s pattern (§2).
Same SQuAD-style schema as UIT-ViQuAD (reuses its `ViquadSample`), so it goes
through the exact same document construction.

## 8. VIMQA — Le et al. 2022, multi-hop Vietnamese QA

`nguyenlab/vimqa` on HuggingFace (plain parquet, verified 2026-09-22).
Train=8041 / Validation=1003 / Test=1003. Unlike UIT-ViQuAD, VIMQA's official
Test split IS directly usable (verified: zero null answers/contexts, real
gold answers, not withheld) -- reserved exclusively for
`ttcompress/vietnamese_public_test.py`; Train/Validation are what
`ttcompress/multilingual_sources.py` draws from instead.

A native needle-in-haystack task already: every row ships 10 candidate
Wikipedia-derived documents (HotpotQA-distractor style, `context.title`/
`context.sentences`), only 1-3 of them containing a `supporting_facts` entry
for the (multi-hop, `type='bridge'`) answer -- **no synthetic haystack
construction needed**, unlike UIT-ViQuAD/XQuAD-vi (§§6-7). Two real,
verified deviations from the raw data:
- Yes/no rows (`answer` in `{'đúng', 'không'}`) are dropped -- 300/1003
  (30%) of Validation/Test, 3932/8041 (49%) of Train -- since a binary
  answer would make `token_f1` trivially matchable without real retrieval.
- Chunk granularity is per candidate document (title), not per sentence --
  all of a title's sentences are joined into one chunk (10 chunks/document),
  matching LongBench's paragraph-level granularity (§2) rather than
  RULER/Kamradt's sentence-level one.

License unresolved, same caveat as UIT-ViQuAD (§6): the HF mirror carries no
license field.

Loaded by `ttcompress/vimqa.py::load_split`, built directly into
`ConstructedDocument` (bypassing `ttcompress/build_document.py` entirely).
