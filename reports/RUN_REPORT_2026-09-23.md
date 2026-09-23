# Outcome-Supervised Relevance — pilot run report

**Date:** 2026-09-23 · **Cluster:** `stg` (`fke-ncp-modas-stg-qc8ifaxe`), ns `llms-dev`
**Node:** `...-workers-z1-f9f69-wknqc`, 4× H100 80GB
**Pipeline:** `OUTCOME_SUPERVISED_RELEVANCE_SPEC.md`, all 7 stages, exit code 0
**Artifacts:** `/mnt/hps/anhm-paper/ttcompress/results/outcome_mgpu/` (persisted; survives pod deletion)
**Code:** commit `154a617` + the contamination fix (see §6)

---

## 1. Headline

The pipeline now runs end to end — it could not before — and produced two trained
relevance checkpoints plus a one-time evaluation on two held-out public test pools.

**The trained method does not beat its trivial baseline, and the result contradicts
the λ chosen on dev.** Both findings are reported as measured; neither is a crash.

Two things must be said before any number below is read:

1. **Five bugs had to be fixed to get here** (§5), three of which silently corrupted
   results rather than crashing. One of them (train/test contamination) invalidated a
   first, apparently-complete set of numbers, which were discarded and re-measured.
2. **This is pilot scale**: 28 documents per split, 484 training windows, 3 epochs.
   That is far too small to conclude the *method* fails. It is enough to conclude the
   pipeline is now trustworthy and to size a real run.

---

## 2. Final numbers (one-time, held-out)

`token_f1`, 95% CI from document-clustered bootstrap. λ* = 0.1 for both checkpoints.

### Vietnamese public test pool — 120 samples (UIT-ViQuAD + XQuAD-vi + VIMQA reserved portions)

| arm | ratio | relevance_v2 (raw) | relevance_v3 (residual) |
|---|---|---|---|
| **truncation** (baseline) | 4× | **0.4416** [0.3705, 0.5139] | **0.4416** [0.3705, 0.5139] |
| **truncation** (baseline) | 8× | **0.4375** [0.3654, 0.5105] | **0.4375** [0.3654, 0.5105] |
| encoder_pcs (λ=0.1) | 4× | 0.3836 [0.3162, 0.4532] | 0.4061 [0.3393, 0.4754] |
| encoder_pcs (λ=0.1) | 8× | 0.3480 [0.2810, 0.4158] | 0.3105 [0.2447, 0.3789] |
| encoder (λ=0) | 4× | 0.2358 [0.1825, 0.2918] | 0.2079 [0.1610, 0.2558] |
| encoder (λ=0) | 8× | 0.1898 [0.1423, 0.2399] | 0.1715 [0.1265, 0.2211] |

`truncation` is identical across checkpoints by construction (it ignores the relevance
model) — a useful sanity check that the two runs are comparable.

**Truncation is the best arm at both ratios.** The CIs overlap heavily
(truncation@4× [0.3705, 0.5139] vs v3 encoder_pcs@4× [0.3393, 0.4754]), so this is
*not* a significant loss — but it is certainly not a win, and the ordering is
consistent at 4× and 8×.

### English public test pool — 906 samples (LongBench / RULER / Kamradt / InfiniteBench reserved halves)

| arm | ratio | relevance_v2 (raw) | relevance_v3 (residual) |
|---|---|---|---|
| **encoder** (λ=0) | 4× | **0.4006** [0.3852, 0.4158] | **0.4124** [0.3962, 0.4286] |
| **encoder** (λ=0) | 8× | 0.3747 [0.3621, 0.3874] | 0.3791 [0.3655, 0.3926] |
| encoder_pcs (λ=0.1) | 4× | 0.3953 [0.3800, 0.4104] | 0.3926 [0.3778, 0.4074] |
| encoder_pcs (λ=0.1) | 8× | 0.3728 [0.3601, 0.3854] | 0.3735 [0.3601, 0.3868] |
| truncation (baseline) | 4× | 0.1552 [0.1391, 0.1717] | 0.1552 [0.1391, 0.1717] |
| truncation (baseline) | 8× | 0.1156 [0.1020, 0.1297] | 0.1157 [0.1022, 0.1298] |

**Here the learned relevance model wins enormously** — 0.40 vs truncation's 0.16, a
~2.6× margin with non-overlapping CIs. But the *position calibration adds nothing*:
`encoder` (λ=0, pure learned relevance) matches or slightly beats `encoder_pcs`
(λ=0.1) at both ratios.

### The two findings worth carrying forward

**(a) λ* did not transfer from dev to test.** The dev sweep picked λ=0.1 because it
clearly beat λ=0 there (0.4374 vs 0.3795, §3). On the English test pool the ordering
reverses — λ=0 is as good or better. Either the dev pool (28 mixed-source documents)
is too small to select λ on, or the gain at λ=0.1 is specific to the dev mixture.
**λ selection is the weakest link in this run**, not the relevance model.

**(b) The baseline ordering flips by language.** Truncation is the best arm on
Vietnamese and by far the worst on English. This is consistent with how the documents
are built: the Vietnamese pools place the needle at one of three positions
(beginning/middle/end) per `build_document.py`, so keeping a contiguous prefix is a
strong strategy; the English needle-in-haystack sources randomize depth, which
punishes truncation. **Any claim about compression quality must therefore be reported
per language, never pooled** — a pooled average here would hide a sign flip.

This also explains γ₁ (§3): the U-shaped position prior carries almost no weight, so
raw and residual labels are nearly the same, which is why v2 ≈ v3 everywhere.

---

## 3. Stage-by-stage results

| stage | result |
|---|---|
| 1 — overlap check (§1) | **EMPTY** — UIT-ViQuAD train vs. its reserved test: 0 content, 0 title; vs. XQuAD-vi: 0 |
| 2 — Stage A measure | 28 docs/split, **3022 train + 3034 dev reader calls**; chunks/doc min 10, mean 53.0, max 199 |
| 3 — Stage A fit | α = **1.0** (CV over 28 dev docs); **2/28 (7.1%)** docs have ≥1 low-confidence chunk (CI width > 0.5) |
| 4 — Stage B decompose | mean **γ₁ = 0.0144** — the position prior `g(i,L)` explains ~1.4% of β_c |
| 5 — Stage C train | v2 MSE 1.1392 → **0.9755**; v3 1.1372 → **1.0100** (3 epochs, 484 windows, xlm-roberta-base) |
| 6 — Stage 6 sweep | both checkpoints: **λ\* = 0.1**; monotonically decreasing after 0.1 |
| 7 — Stage 6 evaluate | §2 above |

**λ sweep curve** (dev, mean over 4×/8×):

| λ | 0.0 | **0.1** | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v2 | 0.3795 | **0.4374** | 0.4203 | 0.3929 | 0.3624 | 0.3182 | 0.2992 | 0.2908 | 0.2796 | 0.2781 | 0.2768 |
| v3 | 0.3368 | **0.3990** | 0.3825 | 0.3492 | 0.3553 | 0.3208 | 0.3090 | 0.2924 | 0.2937 | 0.2704 | 0.2768 |

λ=1.0 (pure position, ≡ truncation) is the worst point on dev — note this *disagrees*
with the Vietnamese test pool, where truncation wins. Another symptom of (b) above.

### Per-source Stage A health (`scripts/diagnose_zero_f1.py`)

Masks keep only C/4 or C/8 of chunks, so the needle is usually dropped and a low
overall mean is expected. The diagnostic column that matters is the score on masks
that **kept** the needle:

| source | overall | needle **kept** | needle dropped | keep rate |
|---|---|---|---|---|
| longbench_passage_retrieval_en | 0.407 | 0.791 | 0.326 | 0.17 |
| kamradt_niah | 0.114 | 0.565 | 0.012 | 0.19 |
| infinitebench_passkey | 0.063 | 0.333 | 0.000 | 0.19 |
| ruler_niah_single_1 | 0.026 | 0.133 | 0.000 | 0.20 |
| uit_viquad | 0.274 | n/a (no needle_index) | | |
| vimqa | 0.250 | n/a | | |
| xquad_vi | 0.142 | n/a | | |

`ruler_niah_single_1` at 0.133-when-kept is weak and worth a look in a follow-up: the
other needle sources are 3–6× higher. It is signal, not noise, so it was not excluded.

---

## 4. Performance: 4 GPUs + batching

The pipeline was single-process/single-GPU. Now:

| | before | after |
|---|---|---|
| ms per reader call | 679–707 | **226–249** |
| GPUs used | 1 (at ~49%) | **4** (46–98%) |
| speedup | — | **~2.9× batching × 4 sharding ≈ 11×** |

- **Stage A** shards by document across GPUs; **Stage 6 sweep** shards by λ;
  **Stage C**'s two trainings and the two final evaluations run concurrently.
- Both are **throughput-only, and that is tested, not asserted**: batched output is
  token-identical to sequential decoding on a real HF model across batch sizes
  1/2/3/5/8 and ragged lengths (`tests/test_reader_batching.py`), and a 4-shard run
  yields byte-identical masks/outcomes to an unsharded one
  (`tests/test_generate_labels.py`), because each shard keeps documents' *global*
  indices for seeding and naming.
- Batching is applied only **within one prompt-template group**, since template and
  `max_new_tokens` are per source; batching across sources would score answers under
  the wrong prompt.

---

## 5. Bugs found and fixed

All five were found by running the pipeline, not by reading it. Three were silent.

**1. Flat K under-determines the ridge fit — run-blocking.**
Stage A used one global mask count, but documents carry 10–199 chunks (mean 53) and
`estimate_chunk_contributions` requires K ≥ C+1. The spec §7 pilot default K=15
aborted Stage A's `fit` on **8 of 10** documents — the original run died at step 3/7
with exit 1. `masks_for_document` now scales K per document
(`K = max(--k, ⌈2·(C+1)⌉)`), which is also why the throughput work was necessary: a
valid K is ~5× more reader calls.

**2. `device == 'cuda'` never matches `cuda:N` — silent, expensive.**
Every multi-GPU path pins shards with an indexed device, so the equality check fell
through and left an 8B reader **on the CPU**: 31 CPU-minutes burned in 4 wall-minutes,
GPUs at 0%, 1.1GB memory. Nothing raised — it looked like a slow GPU. Fixed in all
three sites via `resolve_device()`. **This also affected the pre-existing
`run_pipeline_multi_gpu.sh`** (the PCS pipeline), which has always run on CPU.

**3. `max_new_tokens=6` truncates the answer away — silent, total signal loss.**
Copied verbatim from upstream InfiniteBench, that budget assumes a reader emitting
bare digits. Qwen3-8B is chat-tuned and spends it all on preamble:

| budget | answer | token_f1 |
|---|---|---|
| 6 | `'The pass key is **2'` | **0.000** |
| 16 | `'The pass key is **28024**.'` | 0.333 |

The source scored **exactly 0.000 on every mask, including masks that kept the
needle** — its entire contribution was constant-zero noise, while being the most
expensive source to measure (199 chunks → 400 masks/doc). Raised to 32. Found via
`scripts/diagnose_zero_f1.py`, which separates "needle was dropped" (by design) from
"wrong even with the needle present" (broken).

**4. Train/test contamination from a seed mismatch — silent, invalidated results.**
Stage A held its pools out with `seed=0`; `evaluate` defaulted to `seed=1234`. Each
split helper guarantees dev/test disjointness only *within one call* — two seeds are
unrelated partitions. Measured by `scripts/check_all_pool_leakage.py`:

| source | affects | leaked |
|---|---|---|
| infinitebench_passkey | Stage A **training labels** | 19 |
| longbench_passage_retrieval_en | Stage A **training labels** | 19 |
| xquad_vi | Stage A **training labels** | 19 |
| uit_viquad | alpha selection only | 265 / 300 |
| kamradt, ruler, vimqa | — | 0 |

**57 test documents were also training documents.** With one consistent seed: zero
leakage everywhere. A first, apparently-successful set of numbers was produced under
the mismatch and **discarded**; it is retained as
`relevance_v{2,3}_official_LEAKED_seed1234.json` for comparison. The English pool
barely moved (4.2% leaked), but the Vietnamese numbers shifted materially because the
reserved 300 questions are a different sample under a different seed
(e.g. truncation@4× 0.3736 → 0.4416). One `--seed` is now threaded through Stage A,
the sweep and the evaluation.

**5. `--skip-overlap` handed `generate_labels.py` a report path that does not exist.**

### Tests added
`237 passing`. New: `test_reader_batching.py` (batched ≡ sequential on a real HF
model), `test_outcome_sharding.py` (adaptive K is always fittable; λ partition/merge),
`test_resolve_device.py` (a module moved to `cuda:1` reports index 1),
`test_pool_disjointness.py` (the two CLIs' seed defaults agree; Stage A docs absent
from the test pool), plus sharded-≡-unsharded and generation-budget regressions.

---

## 6. Caveats — read before quoting any number

1. **Pilot scale.** 28 docs/split, 484 windows, 3 epochs. Stage C's MSE was still
   falling at epoch 3 (0.9755) — the model is under-trained, so §2 is **not** evidence
   the method fails.
2. **λ selection is unreliable at this scale** — it demonstrably did not transfer (§2a).
   Fix this before scaling anything else.
3. **Never pool across languages** — the baseline ordering flips (§2b).
4. **InfiniteBench's token_f1 ceiling is ~0.333**, not 1.0, because the reference is a
   bare passkey while the reader answers in a sentence. Uniform across arms, so
   arm-to-arm comparison is valid; absolute values are not comparable to other sources.
5. **`ruler_niah_single_1` is weak** (0.133 when the needle is kept) and unexplained.
6. **Stage C truncates to 256-token windows** while documents are far longer
   (a `8532 > 512` tokenizer warning appears); windowing handles it, but the
   interaction with `max_encoder_len` was not investigated.
7. The **λ=1 ≡ truncation equivalence test** passes, so the PCS wiring itself is sound.

## 7. Recommended next steps

1. **Fix λ selection first** — larger dev pool, or select λ per language, or report the
   whole curve on test instead of a single dev-chosen point.
2. **Scale Stage A** — with ~235 ms/call on 4 GPUs, N=200 docs/split is ≈2–3 h. Stage C
   needs far more than 484 windows.
3. **Report per language**, always.
4. **Investigate RULER's low needle-kept score.**
5. **Re-run `run_pipeline_multi_gpu.sh`** (the PCS pipeline) — its prior results were
   produced on CPU via bug 2; any wall-clock figures from it are wrong.

---

## 8. Reproduce

```bash
# 4×H100, ~35 min at this scale
./run_outcome_pipeline_multi_gpu.sh \
    --num-gpus 4 --batch-size 8 --n 28 --seed 0 \
    --reader-model Qwen/Qwen3-8B --run-dir results/outcome_mgpu

# audits used in this report
python scripts/diagnose_zero_f1.py --in-dir results/outcome_mgpu/outcome_raw/train
python scripts/check_all_pool_leakage.py --train-seed 0 --eval-seed 0
python scripts/probe_passkey_answer.py --device cuda:0
```

**Config:** Qwen3-8B reader (fp16), xlm-roberta-base encoder, 7 sources, N=28/split,
K adaptive (2 masks per ridge parameter, floor 15), batch 8, α grid
{0.01,0.1,1,10,100}, λ grid 11 points, 3 epochs, lr 2e-5, seed 0.
