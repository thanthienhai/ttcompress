# ttcompress — Position-Calibrated Selection (PCS)

Standalone, academic-scope implementation of [`PCS_METHOD_SPEC.md`](PCS_METHOD_SPEC.md).
Code copied/adapted from the sibling `vncompress` project where it matters for
correctness (see per-file docstrings); everything else is written fresh and
kept intentionally small — this is a research/eval codebase, not production.

## Data: four published needle_in_haystack sources, not self-built data

- **LongBench `passage_retrieval_en`** (THUDM, ACL 2024) — real published split, used as-is.
- **RULER `niah_single_1`** (Hsieh et al. 2024, NeurIPS) — generated, needle/query templates copied verbatim from `NVIDIA/RULER`.
- **Classic Needle-In-A-Haystack** (Kamradt) — generated, needle/question copied verbatim from `gkamradt/LLMTest_NeedleInAHaystack`.
- **InfiniteBench `passkey`** (Zhang et al. 2024, ACL) — real per-sample passkey values, re-scaled from the original's ~470K-char context and re-randomized to a non-trivial needle depth (see `data/SOURCES.md` §5 for why the original is unusable as published).

The first three share one haystack corpus (49 public-domain Paul Graham
essays — the same corpus RULER's own "essay" haystack type uses);
InfiniteBench uses its own verbatim filler text instead. Full provenance/
license per file: [`data/SOURCES.md`](data/SOURCES.md).
`ttcompress/public_datasets.py::build_dev_test_split` pools all four into one
`needle_in_haystack` task and splits dev/test with **zero shared source
document** between the two halves (disjoint essay pools for RULER/Kamradt,
disjoint sample ids for LongBench/InfiniteBench) — see spec §7 on dev/test contamination.

**`data/*.json`/`data/*.jsonl` are not committed** (`.gitignore`) — they're
downloaded/assembled by `scripts/fetch_data.sh`, not pushed to git:

```bash
scripts/fetch_data.sh            # fetches into data/ (skips files that already exist)
scripts/fetch_data.sh --force    # re-downloads everything
```

It pulls the plaintext Paul Graham essays from the same URL list RULER's own
generator uses, assembles them into `data/paul_graham_essays.json` (via
`scripts/build_paul_graham_essays.py`), and extracts
`data/longbench_passage_retrieval_en.jsonl` from LongBench's published
`data.zip` — only `curl`, `unzip`, and Python's standard library, no extra
dependencies. `data/SOURCES.md` (this doc) stays tracked either way.
InfiniteBench `passkey` isn't in this script -- like UIT-ViQuAD/XQuAD-vi
below, `ttcompress/public_datasets.py::_infinitebench_passkey_rows` pulls it
straight from HuggingFace at runtime via `huggingface_hub.hf_hub_download`
(cached under the standard HF cache dir, not under `data/`).

## Two things resolved by reading vncompress's code (spec's open questions)

- **§4 (token vs. chunk):** E6 (`EncoderClassifierCompressor` in
  `vncompress/vncompress/encoder_compression.py`) pools its keep-probability
  onto **generation-tokenizer token spans**, one score per token — not
  sentence/chunk-level. `g(i, L)` applies directly, no unit conversion. See
  `ttcompress/pcs.py` docstring.
- **§10 (H2O/SnapKV):** real eviction logic exists
  (`SnapKVCompressor` in `vncompress/vncompress/compression.py`, modes
  `snapkv`/`h2o`/`streamingllm`, attention-based, `output_attentions=True`),
  copied verbatim into `ttcompress/compression.py`. It is registered but has
  never been run (matches the spec's own note) — this build does not change
  that, it just makes the arm runnable.

**Still open:** no trained E6 checkpoint exists yet anywhere under
`vncompress/models/` (only `scripts/train_encoder_compressor.py`, unrun).
`ttcompress/relevance.py::E6RelevanceProvider` is a real, unexercised code
path — point `--encoder-path` at a checkpoint once one exists. Training that
checkpoint is out of scope here by design (hard constraint #1: bolt-on only).

## A spec inconsistency, resolved: the dev sweep costs real GPU time

§9 describes the λ sweep as "no GPU needed, CPU only" (just mixing
already-computed E6 scores with `g(i,L)`). But §1 defines the sweep's curve
in terms of `token_f1`, which only exists after the **reader actually
generates an answer** for each compressed variant. Those two claims conflict.
`train.py` follows §1 literally: it calls `reader.generate_answer` for every
(λ, ratio, sample) triple, so a full sweep is GPU-bound like the official
bench, not a free CPU pass — budget for it accordingly (`--num-dev-essays`/
`--samples-per-essay`/`--longbench-dev` control how many dev samples that is).

## Three gaps found and fixed while reviewing the pipeline

- **LongBench answers need LongBench's own prompt.** Its gold answers are
  formatted `"Paragraph N"`; a generic instruction gets a free-text answer
  back and `token_f1` against `"Paragraph N"` reads ~0 regardless of
  compression quality, drowning out a third of the data. `reader.py` now
  keys prompt templates (and `max_new_tokens`) by `dataset_source`, with
  LongBench's real template copied verbatim from
  `THUDM/LongBench/LongBench/config/dataset2prompt.json`.
- **h2o/snapkv need real attention weights.** `SnapKVCompressor` requires
  `output_attentions=True` to actually return something — sdpa/
  flash-attention, which many models default to, silently don't. `evaluate.py`
  now forces `attn_implementation='eager'` on the reader automatically
  whenever `--arms` includes `h2o`/`snapkv`.
- **Dev/test split reproducibility was flag-dependent.** `evaluate.py` used
  to re-derive its test set from CLI numbers that had to match `train.py`'s
  by hand — a silent mismatch would leak dev documents into the test run.
  `train.py` now writes the actual resolved dev+test samples to
  `results/dev_test_split.json`; `evaluate.py` loads that file by default,
  so there is nothing left to keep in sync (see `save_dev_test_split`/
  `load_dev_test_split` in `public_datasets.py`).

## A real bug, found by actually running the "unexercised" h2o/snapkv arms

`SnapKVCompressor`'s docstring always said h2o/snapkv were "NOT exercised on
this dev machine" — asked to estimate `run_pipeline.sh`'s wall-clock on real
hardware, that claim got checked instead of trusted, and it broke
immediately: `_compute_attention_importance` called the 8B reader with
`output_attentions=True` on the FULL, uncompressed context. HF hands back
attention weights for every layer at once (it can't return a subset), and at
LongBench's real context length (~15,000 tokens) that's
`36 layers × 32 heads × 15,000² × 2 bytes ≈ 518GB` just for the attention
tensors — a guaranteed CUDA OOM on any real GPU, not a slow arm, a crashing
one, and it would have hit within the first minute of `evaluate.py`'s
default `--arms` (LongBench sorts first in the test pool).

**Fix, not a workaround**: SnapKV/H2O's own published algorithm only ever
needs attention from a small trailing "observation window" (`window_size`
query positions, already a constructor parameter — the intent was always
there, just not implemented) against the full KV cache — not every query
position's attention, which the old single-pass call retained regardless of
ever reading it. `_compute_attention_importance` now does a prefill pass
with `output_attentions=False` (builds the KV cache, same cost as ordinary
generation prefill) followed by a small windowed pass with
`output_attentions=True` over just the last `window_size` tokens. This is
mathematically IDENTICAL to the original single pass — verified in
`tests/test_snapkv_windowed_attention.py` against a real (if tiny)
multi-layer attention reference implementation in float64 (max diff ~1e-10,
pure floating-point noise; a real logic bug would show up 8+ orders of
magnitude larger, confirmed by first observing exactly that at float32) —
just far cheaper: at LongBench's scale, ~3GB instead of ~518GB.

A second, smaller bug turned up alongside it: `registry.make_compressor`
never passed `--device` through to `SnapKVCompressor`, which defaulted to
`device='cuda'` unconditionally — invisible on the documented single-GPU
`cuda:0` default, but a real device mismatch (or an outright crash) on
`--device cpu` smoke tests, on `cuda:N` for N>0, or on
`run_pipeline_multi_gpu.sh`'s per-shard GPU pinning below. Fixed by deriving
the device from the model actually doing the forward pass instead
(`tests/test_registry_device.py`).

## Multi-GPU sharding: `run_pipeline_multi_gpu.sh`

Neither `train.py`'s λ sweep nor `evaluate.py`'s arm/ratio bench
parallelizes internally — each is a single process pinned to one `--device`,
so a machine with N GPUs, run through plain `run_pipeline.sh`, only ever
uses one of them. `run_pipeline_multi_gpu.sh` is the sharded alternative:
same six steps, but steps 4/5 split their work across `NUM_GPUS` GPUs
(`.env`/`--num-gpus`, `cuda:0..cuda:N-1`):

- **Step 4** splits the 11-point λ grid round-robin across GPUs
  (`train.py --lambdas`, new) — every shard resolves the identical dev/test
  split (same seed), so it's written once upfront
  (`train.py --skip-split-write`, new) instead of N processes racing to
  write the same file.
- **Step 5** splits the (arm, ratio) cells — `encoder_pcs@4x`, `encoder_pcs@8x`,
  `h2o@8x`, `snapkv@8x` by default, a natural fit for 4 GPUs
  (`evaluate.py --ratios`, new, restricts an arm's ratios to the given set).
- `scripts/shard_pipeline.py` computes each shard's slice
  (`lambdas-for-shard`/`cells-for-shard`) and merges every shard's output
  JSON back into the exact single-process file shape
  (`merge-train`/`merge-eval`) — `results/lambda_sweep_dev.json` and
  `results/official_bench.json` look identical either way, so nothing
  downstream (step 6's push, `evaluate.py` reading `chosen_lambda`) needs to
  know sharding happened. Partitioning is round-robin, not contiguous
  blocks, and correctly leaves a shard idle (not an error) when `NUM_GPUS`
  exceeds the work-unit count — tested in `tests/test_shard_pipeline.py`.

```bash
./run_pipeline_multi_gpu.sh --reader-model Qwen/Qwen3-8B --encoder-path models/encoder_compressor
./run_pipeline_multi_gpu.sh --num-gpus 4 --relevance synthetic --skip-install   # smoke-test the sharding itself
```

## Layout

```
ttcompress/
  compression.py     BaseCompressor, TruncationCompressor, SnapKVCompressor
  pcs.py              g(i,L), score(i) = (1-λ)·relevance + λ·g, PCSCompressor
  relevance.py        E6RelevanceProvider (real, bolt-on) + SyntheticRelevanceProvider (smoke-test only)
  metrics.py          token_f1, needle_recall, document-clustered bootstrap CI
  data.py             NeedleSample — the one row type every source produces
  public_datasets.py  LongBench / RULER / Kamradt loaders + build_dev_test_split
  reader.py           prompt building + greedy generation
  registry.py         arm name -> compressor factory
train.py              "training" flow = the λ sweep on the dev split (no gradients, ever)
                      --lambdas/--skip-split-write: shard_pipeline.py's multi-GPU sweep support
evaluate.py           official bench on the test split + CI (pooled and per-source)
                      --ratios: shard_pipeline.py's multi-GPU sweep support (one arm/ratio cell per shard)
tests/                mandatory λ=1 ≡ TruncationCompressor equivalence test, metrics sanity,
                      SnapKV windowed-attention equivalence, shard partitioning/merging
run_pipeline.sh        one command: install -> fetch data -> sanity test -> train -> evaluate
run_pipeline_multi_gpu.sh  same, but steps 4/5 sharded across NUM_GPUS GPUs -- see below
scripts/
  fetch_data.sh                    downloads/assembles everything under data/ (not committed)
  build_paul_graham_essays.py      helper fetch_data.sh calls to assemble the essay corpus JSON
  shard_pipeline.py                run_pipeline_multi_gpu.sh's work partitioning + shard-output merging
data/
  paul_graham_essays.json               haystack corpus (RULER + Kamradt) -- gitignored, run fetch_data.sh
  longbench_passage_retrieval_en.jsonl  LongBench split, as published -- gitignored, run fetch_data.sh
  SOURCES.md                            exact provenance/license per file -- tracked
```

## Run

**All in one** (`run_pipeline.sh`): installs `requirements.txt`, runs
`scripts/fetch_data.sh`, runs the mandatory tests as a gate, sweeps λ on dev,
evaluates on test, optionally pushes results to HF — each step individually
skippable so a re-run doesn't redo expensive ones:

```bash
./run_pipeline.sh --reader-model Qwen/Qwen3-8B --encoder-path <path-to-trained-E6>

# smoke-test the whole wiring with no checkpoint/GPU/real download cost:
./run_pipeline.sh --relevance synthetic --device cpu

# already know lambda*, or re-running evaluate.py alone after a change:
./run_pipeline.sh --lam 0.4 --skip-train --reader-model Qwen/Qwen3-8B --encoder-path <path-to-trained-E6>

# same run, plus push results/*.json to a private HF Dataset repo at the end:
./run_pipeline.sh --reader-model Qwen/Qwen3-8B --encoder-path <path-to-trained-E6> \
    --push-to-hub --hf-repo-id you/ttcompress-pcs-results

./run_pipeline.sh --help   # every flag, incl. --venv, --ratios, --arms, --force-fetch
```

`cp .env.example .env` and fill it in to set the same flags via environment
instead of typing them every run (`READER_MODEL`, `ENCODER_PATH`,
`RELEVANCE`, `DEVICE`, `RATIOS`, `ARMS`, `PUSH_TO_HUB`/`HF_REPO_ID`/
`HF_PRIVATE`, plus `HF_TOKEN`/`HF_HOME`/`CUDA_VISIBLE_DEVICES` for gated
models, pushing, or GPU selection). `run_pipeline.sh` auto-loads `.env` if
present; a CLI flag always overrides it. `.env` is gitignored — only
`.env.example` is tracked. `train.py`/`evaluate.py` run directly (without
`run_pipeline.sh`) don't read `.env`, only their own flags.

### Push results to Hugging Face

PCS never trains a model — bolt-on only (hard constraint #1), so `train.py`
produces a chosen λ\* and a sweep curve, not a checkpoint. `--push-to-hub`
therefore pushes **run artifacts only** (`results/*.json`: the lambda curve,
the exact dev/test split used, the official bench's point estimate + 95% CI)
to a **Dataset** repo, not a Model repo — there is nothing to push to a Model
repo unless you separately have a trained E6 checkpoint, which lives in
`vncompress` and isn't something this build produces or pushes. Private by
default (`--hf-public` to make it public); needs `HF_TOKEN` with write access.
Standalone: `python scripts/push_results.py --repo-id you/ttcompress-pcs-results`.

**Or step by step**, same thing `run_pipeline.sh` automates:

```bash
pip install -r requirements.txt
scripts/fetch_data.sh               # populates data/ (not committed, see .gitignore)
python -m pytest tests/ -q          # no GPU needed — the mandatory equivalence test lives here

# 1. "training" = pick λ* on the dev split (needs a reader model; E6 checkpoint optional until you have one)
#    writes results/dev_test_split.json (dev used here, test held out for step 2)
python train.py --reader-model Qwen/Qwen3-8B --encoder-path <path-to-trained-E6> \
    # or, to smoke-test the pipeline without a checkpoint/GPU:
    # --relevance synthetic

# 2. official bench: encoder_pcs@{4x,8x} + h2o/snapkv@8x on the test split, pooled + per-source CI
#    loads its test set from results/dev_test_split.json by default -- no flags to keep in sync
python evaluate.py --reader-model Qwen/Qwen3-8B --encoder-path <path-to-trained-E6> --lam <λ* from step 1>
```

`train.py` never touches the test set; `evaluate.py` never sweeps λ — this
mirrors the spec's dev/test separation (§7) and the "single global λ" hard
constraint (#3).

---

# Outcome-Supervised Relevance — a second pipeline, now merged with PCS's data

Implementation of [`OUTCOME_SUPERVISED_RELEVANCE_SPEC.md`](OUTCOME_SUPERVISED_RELEVANCE_SPEC.md):
train a *new* relevance model (unlike PCS above, this one **does** run real
gradient training) by measuring the reader's actual `token_f1` outcome under
random chunk-keep masks, fitting each chunk's marginal contribution via ridge
regression, separating out the position-signal PCS already captures, then
training two regression checkpoints on what's left. Reconnects to the
existing PCS infrastructure (`registry.make_compressor`, `bootstrap_mean_ci`,
the λ=1 equivalence guarantee) for the final numbers — nothing there is
reimplemented.

**Multilingual, public-only upgrade:** the spec's original scope was
UIT-ViQuAD 2.0 (Vietnamese) only, tested on `vcc_bench_v2.json` — an
**unpublished internal vncompress benchmark**, not a public dataset (verified:
no HF page, no citation, no public download anywhere). `--sources` (default
`all`) now mixes in two more public Vietnamese sources (XQuAD-vi, VIMQA) and
four English sources (LongBench passage_retrieval_en, RULER niah_single_1,
classic Kamradt NIAH, InfiniteBench passkey) for Stage A's label generation
and Stage 6's λ sweep -- one multilingual `relevance_v2`/`relevance_v3`
checkpoint, trained on chunks from all seven sources, then evaluated on
**both** the public Vietnamese test pool (`ttcompress/vietnamese_public_test.py`
-- replaces `vcc_bench_v2.json` entirely) and the English test pool PCS's own
`evaluate.py` reserves. `ttcompress/multilingual_sources.py` is the one place
this mixing happens; everything downstream (masks, ridge regression, Stage C
training, the PCS reconnection) is unaware which language or source a chunk
came from. Every non-UIT-ViQuAD-train, non-VIMQA-train source is drawn only
from a reserved "dev"/"tune" pool, never a reserved test pool -- see that
file's module docstring.

Prerequisites this needed and didn't have:
- RULER/LongBench/Kamradt didn't expose chunk structure before (PCS only
  ever compressed them as one whole string). `ttcompress/public_datasets.py`'s
  generators now also populate `NeedleSample.chunks`
  (+ `metadata['needle_index']`) -- RULER/Kamradt/InfiniteBench at sentence
  granularity, LongBench by splitting on its own `"Paragraph N:"` markers
  (verified against the real data, not guessed).
- A second public monolingual-Vietnamese source, since UIT-ViQuAD's own
  official Train/Dev/Test split turned out not to have a usable held-out test
  portion at all (see "A second real finding" below) --
  `ttcompress/xquad_vi.py` (XQuAD Vietnamese, Artetxe et al. 2020),
  independently sourced, same SQuAD-style schema.
- A fourth English source, InfiniteBench `passkey` (Zhang et al. 2024, ACL) --
  added after the user asked whether any more public sources could be added.
  **Not used as published**: the real context is a constant ~470,000
  chars/sample (extreme-length retrieval), infeasible for Stage A's ridge
  regression at pilot scale (needs masks K ≥ chunks C), and the needle is
  always placed in the first ~7% of the text (verified across 50 samples),
  which would hand `g(i,L)` a trivial built-in win. The real per-sample
  passkey *value* is extracted and re-inserted at a random depth into a
  freshly built haystack of InfiniteBench's own verbatim filler text --
  see `data/SOURCES.md` §5 for the full rationale.
- A third public Vietnamese source, VIMQA (Le et al. 2022, multi-hop) --
  chosen alongside InfiniteBench in the same round. Unlike UIT-ViQuAD/XQuAD-vi
  it's a **native** needle-in-haystack task already (10 candidate documents
  per row, HotpotQA-distractor style, 1-3 hold a supporting fact for the
  answer), so it bypasses `ttcompress/build_document.py`'s synthetic
  construction entirely -- `ttcompress/vimqa.py` builds `ConstructedDocument`
  directly from VIMQA's own structure. Its official Test split is directly
  usable (verified, unlike UIT-ViQuAD's), so it's the first Vietnamese source
  that doesn't need a carved-out reserved portion. Yes/no-answer rows
  (`'đúng'`/`'không'`, 30-49% of every split) are dropped -- see
  `data/SOURCES.md` §8.

The base encoder for Stage C changed accordingly: `xlm-roberta-base`
(multilingual) is now the default, not `vinai/phobert-base`
(Vietnamese-only, still available via `--base-checkpoint` for a
`--sources uit_viquad`-only run).

## Three things resolved by reading vncompress's code (spec's open questions)

- **§2.1 (chunk vs. token):** already resolved by `PCS_METHOD_SPEC.md` §4 —
  E6 scores per token. Chunk (one paragraph) is only the unit Stage A
  estimates labels at; Stage C broadcasts a chunk's label to every token
  inside it (`ttcompress/relevance_training.py`).
- **§2.2 (haystack construction):** read `vncompress/scripts/build_vcc_bench_v2.py`
  directly. Distractors are drawn from the **entire** paragraph pool (no
  topic/title filtering), filled to a 30,000-char budget, needle inserted at
  one of exactly 3 positions (beginning/middle/end) assigned **per document**.
  Replicated exactly in `ttcompress/build_document.py`.
- **§4 (E6 architecture reuse):** read `vncompress/scripts/train_encoder_compressor.py`
  in full. Its classifier head (`num_labels=2`, cross-entropy) can't consume
  continuous regression targets — Stage C reuses the same backbone +
  `AutoModelForTokenClassification` class with `num_labels=1` instead, and
  computes MSE manually rather than relying on the model's built-in loss
  (which is hardcoded to cross-entropy regardless of `num_labels`). Same
  `save_pretrained`/tokenizer/`*_meta.json` convention, so the result loads
  the same way. See `ttcompress/relevance_training.py`'s module docstring.

## A real finding, not just a check: train/test corpus overlap

§1 requires checking UIT-ViQuAD 2.0's content for leakage before doing
anything else. The check originally compared it against vncompress's
internal `wikipedia_vi_raw.json` (the corpus the now-dropped
`vcc_bench_v2.json` was built from) and, run for real, found the Dev split
clean but the **Train split not**: 410/19238 samples (86 distinct paragraphs)
shared the Wikipedia article "Hà Nội" with that corpus — confirmed via both
title match and duplicated paragraph text, not a false positive.

That finding is now historical (nothing reads `wikipedia_vi_raw.json`
anymore), but `scripts/check_uit_viquad_overlap.py` wasn't deleted — it was
repointed at the boundary that actually matters for the current pipeline:
UIT-ViQuAD TRAIN vs. its own reserved test portion, and vs. XQuAD-vi. Run for
real again (`results/train_test_overlap.json`): **both come back EMPTY**
(official splits + an independently-sourced second dataset, as expected, but
checked rather than assumed — same WAVE4_REPORT.md §4.3 discipline). The
mitigation mechanism from the first finding is kept as a live safety net,
not removed just because it currently has nothing to filter:
`ttcompress/uit_viquad.py::filter_by_excluded_titles`, wired into
`generate_labels.py measure --overlap-report <path>`, drops any train sample
whose title shows up in a future non-empty overlap report.

## `vcc_bench_v2.json` was dropped — it was never a public dataset

Checked directly: the HF mirror/paper-citation trail for `vcc_bench_v2.json`
doesn't exist — it's an unpublished internal vncompress benchmark (its own
Wikipedia-sourced raw material is CC-BY-SA 4.0, but the curated benchmark
file itself was never released anywhere citable). Every other source in this
project is a real public, citable dataset; keeping this one would have been
the odd one out. Replaced entirely by `ttcompress/vietnamese_public_test.py`
(UIT-ViQuAD 2.0 + XQuAD-vi reserved test portions) — see that module and
`data/SOURCES.md` §§5-6.

## A second real finding: UIT-ViQuAD's official Test split has no answers

Verified directly (not assumed from the dataset card): all 7301 rows of
`taidng/UIT-ViQuAD2.0`'s `test` split have `answers=None` — the standard
withheld-answer convention for a public leaderboard test set, not the
SQuAD2.0 "unanswerable" signal (`is_impossible=False` for every one of them).
It cannot be scored, so it's unusable as this pipeline's Vietnamese test set
despite being the obviously-intended candidate. `load_split('test')` now
returns an empty list rather than crashing on `row['answers']['text']`
(a real crash this caused, not a hypothetical). `ttcompress/uit_viquad.py::
split_dev_tune_test` instead carves a small reserved portion out of the
*Dev* split (default 300 of 3814) — the only place a genuinely held-out,
answerable UIT-ViQuAD test portion can come from; the rest of Dev is what
`multilingual_sources.py`'s sweep/alpha-selection tunes on.

## Two more real bugs, found by actually running the merged pipeline end to end

None of these would've been caught by inspection — all three only showed up
running `generate_labels.py measure --sources all` and `evaluate_relevance.py
evaluate` against real data with a fake model, which is why every stage of
this project gets an actual smoke-test run, not just a syntax check:

- **Windows treats `:` in a filename as an NTFS alternate-data-stream
  separator.** RULER/Kamradt's `doc_id` used to be `f"ruler:{essay_name}"` --
  harmless as a bootstrap-CI cluster label (PCS's original use), but
  `generate_labels.py` now also uses `doc_id` as a `save_mask_outcomes`
  filename. `ruler:bias.jsonl` silently became a hidden stream on a file/dir
  named `ruler`, not an error -- 8 documents measured, only 4 files landed on
  disk, no exception anywhere. Fixed by switching to `ruler_{essay_name}`
  (`ttcompress/public_datasets.py`).
- **`doc_id` is a cluster label, not a unique id — collisions are
  intentional, and that broke file saving a second way.** RULER/Kamradt
  legitimately reuse one `doc_id` per essay across several *different*
  constructed documents (correct for PCS's own bootstrap-CI clustering), but
  with the default `samples_per_essay=2` that means `generate_labels.py`
  could silently overwrite one document's measured `MaskOutcomeRecord` with
  another's under the same filename — caught by a test asserting file count,
  not by asserting `doc_id` uniqueness (which would have been the wrong
  assertion; see `tests/test_multilingual_sources.py`'s note on this).
  Fixed by disambiguating the saved filename with the loop index
  (`generate_labels.py`), keeping the original as `metadata['cluster_id']`.
- **A `dataset_source` with no `PROMPT_TEMPLATES` entry silently gets the
  English `'generic'` template.** Caught twice: first on the now-removed
  `vcc_bench_v2_needle_in_haystack` source, then again on `xquad_vi` when it
  was added — both all-Vietnamese content that would have silently gotten an
  English instruction while being fed Vietnamese context and questions,
  corrupting every number from that source the same way the original
  LongBench prompt-language bug would have. Fixed by aliasing each new
  Vietnamese source to the `'uit_viquad'` template in `ttcompress/reader.py`,
  with a regression test (`tests/test_reader_prompts.py`) that would have
  caught the second occurrence automatically had it been written after the
  first instead of the second.

## Still unresolved (not a code problem — needs a human)

**License.** UIT-ViQuAD 2.0's HuggingFace mirror (`taidng/UIT-ViQuAD2.0`)
carries **no license field at all**. The dataset card only says "freely
available to encourage the research community" (COLING 2020 abstract),
pointing to `nlp.uit.edu.vn/datasets` and `vlsp.org.vn/vlsp2021/eval/mrc` for
specifics. Confirm with the UIT NLP group before publishing anything trained
on this data (spec §1/§8) — this build does not resolve it either way. XQuAD
(Artetxe et al. 2020) is CC-BY-SA 4.0, no ambiguity there.

## Layout

```
ttcompress/
  build_document.py        needle+haystack document construction (§2.2, replicated exactly) + ConstructedDocument,
                            document_from_needle_sample (adapter from the English sources' NeedleSample)
  uit_viquad.py             taidng/UIT-ViQuAD2.0 loader, split_dev_tune_test, overlap-exclusion filter
  xquad_vi.py               xquad.vi loader + its own dev/test split (no official one of its own)
  vimqa.py                  nguyenlab/vimqa loader -- native needle-in-haystack, no build_document.py needed
  multilingual_sources.py   the merge point: mixes uit_viquad/xquad_vi/vimqa/longbench/ruler/kamradt/infinitebench into one document pool
  vietnamese_public_test.py  the public Vietnamese test pool -- replaces vcc_bench_v2.json entirely
  outcome_labels.py         masks, ridge regression, bootstrap CI, measure/fit artifact types (§2.3-2.6)
  position_decompose.py     beta_c = gamma_0 + gamma_1*g(i_c,L) + residual_c (§3), TrainingLabels artifact
  relevance_training.py     chunk-label -> per-E6-token windowing for Stage C (GPU-free, unit-tested)
  relevance.py              + RegressionRelevanceProvider (sigmoid(logit), alongside the existing E6RelevanceProvider)
generate_labels.py          Stage A CLI: measure (GPU) / fit (CPU) subcommands, --sources for the merge
decompose_labels.py         Stage B CLI (CPU only)
train_relevance.py           Stage C CLI: real gradient training -> relevance_v2 / relevance_v3
                              (default base checkpoint: xlm-roberta-base, multilingual)
evaluate_relevance.py        Stage 6 CLI: sweep (--sources) / evaluate (public Vietnamese pool + English test pool)
run_outcome_pipeline.sh      all of the above, pilot-scale by default (spec §7: N=10, K=15), --sources all by default
scripts/
  check_uit_viquad_overlap.py   §1's mandatory pre-check -- now checks UIT-ViQuAD train vs. its own reserved
                                 test portion and vs. XQuAD-vi (both come back EMPTY), real output in results/
tests/
  test_build_document.py, test_uit_viquad.py, test_xquad_vi.py, test_vimqa.py, test_outcome_labels.py,
  test_multilingual_sources.py, test_vietnamese_public_test.py, test_document_from_needle_sample.py,
  test_position_decompose.py, test_stage_b_cli.py, test_relevance_training.py, test_reader_prompts.py,
  test_generate_labels.py (the doc_id-collision regression test), test_relevance.py (classification vs.
  regression scoring, locked in with a fake encoder)
```

## Run

```bash
./run_outcome_pipeline.sh --reader-model Qwen/Qwen3-8B   # multilingual, pilot scale (N=10, K=15) by default
./run_outcome_pipeline.sh --reader-model Qwen/Qwen3-8B --n 200 --k 20   # past the pilot, once timing looks right
./run_outcome_pipeline.sh --reader-model Qwen/Qwen3-8B \
    --sources uit_viquad --base-checkpoint vinai/phobert-base   # original Vietnamese-only scope
./run_outcome_pipeline.sh --help
```

Every stage's math (masking, ridge regression, bootstrap CI, position
decomposition, chunk→token label windowing, the training loop's masked-MSE
step, the multi-source document mixing) is unit-tested and was also
smoke-tested end-to-end against real `hf-internal-testing/tiny-random-bert`
weights, real documents built from all seven sources at once (UIT-ViQuAD,
XQuAD-vi, VIMQA, LongBench, RULER, Kamradt, InfiniteBench passkey), and the real public Vietnamese test
pool + English test pool (fake reader, so `token_f1` reads ~0 — the point was
proving the wiring reaches real scoring code with the right per-source
prompt language, not producing a real number). What is **not** exercised at
scale in this build: Stage A's `measure` step against a real 8B reader
(that's real GPU-hours no dev machine has), and Stage C training on
`xlm-roberta-base` at more than pilot size. Spec §7's two missing cost
numbers (ms/mask, chunks/document) print directly from a real pilot run —
read them before scaling `--n`/`--k`.
