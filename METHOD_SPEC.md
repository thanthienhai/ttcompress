# Amortized Utility Attribution for Reader-Agnostic Context Compression — Method Spec

> **Status:** v1, replaces `PCS_METHOD_SPEC.md` and `OUTCOME_SUPERVISED_RELEVANCE_SPEC.md` (removed; see git
> history). Source: the COLING 2027 proposal (`compass_artifact_…md`, direction A + C) and the review of
> `reports/RUN_REPORT_2026-09-23.md`.
> **Target:** ARR October 2026 cycle (deadline 2026-10-12), COLING 2027 main areas (IR / efficient methods /
> QA / multilinguality), not the "NLP for Linguistics" theme track.

---

## 0. What changed from the previous direction, and why

| Previous (PCS / outcome-supervised relevance) | Now |
|---|---|
| Query-**agnostic** token scorer (E6) + a global position mix-in λ | Query-**aware** cross-encoder that scores **chunks** in one pass |
| β framed as "relevance"; overlaps ContextCite/AT2 if framed as attribution | β framed as **utility attribution to be amortized** (distilled), cost of labels reported |
| λ chosen on 28 dev docs, did not transfer | No λ. Pruner model selection is reader-free, on dev labels |
| Needle at {first, middle, last} → truncation won by construction | Needle depth **uniform**; results sliced by depth |
| Per-document regression of β on a U-shaped prior (γ₁ ≈ 0 by construction) | Pooled position prior; its explained variance (R²) is reported; position adjustment is an ablation |
| dev/test via `Random(seed)` in two CLIs → 57 leaked docs | Splits are a **hash of the example key** (title / passage / id); no seed exists |
| One reader | ≥ 3 label readers + 1 held-out eval reader; ensemble labels; upgrade retention |
| Compared overlapping marginal CIs | **Paired** cluster bootstrap of per-document differences |
| NIAH synthetic English sources | Real multi-hop QA (HotpotQA, 2Wiki, VIMQA) + Vietnamese single-hop haystacks |

## 1. Research questions and hypotheses

- **RQ1 (amortization).** A pruner distilled from F1-utility attribution keeps downstream F1 at fixed token
  budgets (1/4, 1/8) while costing one encoder pass instead of K≈64 reader calls.
  *H1:* `ours_beta` beats `bm25`, `embed`, `lead`, `random` at both ratios on every source (paired Δ F1 > 0,
  CI excludes 0), and gets within a small margin of `oracle_beta`.
- **RQ2 (survival test).** Utility labels beat answer-span supervision.
  *H2a:* on multi-hop (VIMQA, HotpotQA, 2Wiki), `ours_beta` > `span_sup` and > `oracle_span` (the answer-span
  oracle misses bridge paragraphs). *H2b:* on single-hop, `ours_beta` ≈ `span_sup` is the expected outcome
  (β collapses onto the needle — measured directly by `gold_recall_at_g` in Stage A summaries); a win there
  is a bonus, not a requirement.
  **Decision rule (from the proposal):** if H2a fails, move the paper's weight to RQ3.
- **RQ3 (reader-agnostic).** *H3a:* labels from different readers agree (cross-reader Spearman of β in
  `ensemble/summary.json`) but not perfectly. *H3b:* `ours_ens` (ensemble labels) retains more of the
  weak→strong reader upgrade than `ours_beta` (single reader), including for the **held-out** reader
  (Qwen3-32B by default) that produced no labels. Upgrade retention per arXiv 2606.21807.
- **RQ4 (multilingual / Vietnamese).** *H4:* on Vietnamese (syllable-level token F1) `ours_*` beats
  XProvence zero-shot and LLMLingua-2 at matched budgets. If XProvence is already strong, the contribution
  shifts to the benchmark + cross-reader analysis (proposal §Recommendation).

Everything is reported **per source** (never pooled across sources or languages): the previous run showed a
sign flip between pools.

## 2. Data (`ttcompress/sources.py`)

| source | lang | hop | chunks | train | dev / test |
|---|---|---|---|---|---|
| `uit_viquad` | vi | single | needle paragraph + distractor paragraphs to 30k chars, needle depth uniform | official train | official validation hashed **by title** 50/50 |
| `xquad_vi` | vi | single | same construction | — (eval only) | hashed **by passage** 20/80 |
| `vimqa` | vi | multi | 10 titled paragraphs (native) | official train | official validation / test |
| `hotpotqa` | en | multi | 10 titled paragraphs (distractor setting) | official train | official validation hashed by id 50/50 |
| `2wiki` | en | multi | 10 titled paragraphs | (not used for training) | official validation hashed by id 50/50 |

Yes/no questions are dropped. `gold_chunks` = needle / supporting-fact paragraphs; only oracle arms read it.
`xquad_vi` and `2wiki` are never trained on (cross-dataset transfer). Options for ablations:
`--distractors hard` (same-article distractors for single-hop), `--multihop-pad-chars` (lengthen multi-hop
documents with easy distractors).

Licenses to confirm before submission: UIT-ViQuAD (research-only terms from UIT NLP group), VIMQA (user
agreement for the full release).

## 3. Stage A — utility attribution (`generate_labels.py`, `ttcompress/attribution.py`)

For each (document, reader): K masks, each chunk kept i.i.d. with p ∈ {0.5, 0.25} (cycled),
K = clamp(⌈C+1⌉, 64, 256); masks are seeded by the document id, so **every reader sees identical masks**.
The reader answers each masked context (plus the full context, stored as `full_f1`); outcomes are token F1
(the method) and the teacher-forced answer log-prob (ablation). Ridge regression (intercept unpenalized,
one global α chosen by 5-fold CV MSE on **dev** measurements) gives β; the pseudo-label is the
within-document z-score of β.

Label quality is measured, not assumed (`summary.json` per fit dir):
`mean_cv_r2` (held-out R² of the additive surrogate; low on multi-hop = interaction effects),
`gold_recall_at_g` / `gold_mrr` (does β find the evidence?), `n_informative` (documents whose outcome
varied at all), `position_r2` (share of label variance a pooled position prior explains), and for ensembles
`cross_reader_spearman`.

## 4. Stage B — distillation (`train_pruner.py`, `ttcompress/pruner.py`)

Cross-encoder over `[CLS] question [SEP] chunk_1 … chunk_m [SEP]` (chunks tokenized separately, exact
spans), mean-pool each chunk's contextualized tokens → linear head → one logit per chunk. Documents
longer than `--max-len` (default 4096) are packed into windows of whole chunks, each repeating the
question. Backbone `BAAI/bge-reranker-v2-m3` (same initialization family as XProvence, so the comparison
isolates the supervision). Loss: ListNet over the document's chunks (softmax(z/τ) target) + 0.5·MSE; the
span control uses BCE on gold chunks. Model selection: best epoch by reader-free dev `ndcg@3_beta` (span:
`gold_recall@25%`).

Pruners trained by `run_pipeline.sh`: `pruner_beta_primary` (ours, one reader), `pruner_beta_ensemble`
(ours, reader-agnostic), `pruner_span` (RQ2 control), `pruner_logprob_primary` (F1 vs log-prob, the AT2
distinction), `pruner_beta_primary_posadj` (position adjustment).

## 5. Evaluation (`evaluate.py`, `ttcompress/selection.py`)

Budget = ⌈full_tokens / ratio⌉ in a fixed reference tokenizer; every chunk arm greedily keeps its best
chunks that fit, in original order. Arms: `full`, `lead`, `random`, `bm25`, `embed` (bge-m3),
`oracle_span`, `oracle_support`, `oracle_beta` (optional upper bound), `pruner:<ckpt>` (any number),
`provence:<hf id>` (Provence / XProvence as budget-matched rerankers), `llmlingua2`.
Selections are computed once and answered by every reader (fixed compressor ⇒ upgrade retention is
meaningful).

Reported per reader × source × ratio × arm: token F1, EM, answer recall, gold-chunk recall, realized
compression, selection latency; cluster-bootstrap CIs (cluster = article for UIT, passage for XQuAD);
paired Δ F1 of our arms vs every other arm with CI and p; upgrade retention for every reader pair; F1 and
gold recall by needle-depth quintile for single-hop.

## 6. Ablations (map to proposal)

| ablation | how |
|---|---|
| F1 vs log-prob surrogate | `pruner_logprob_primary` vs `pruner_beta_primary` |
| additivity | `mean_cv_r2` by hop in Stage A summaries; (pairwise-interaction surrogate: future work) |
| position adjustment | `pruner_beta_primary_posadj`; `position_r2`; depth slices |
| 1 reader vs ensemble | `ours_beta` vs `ours_ens`, incl. held-out reader |
| single- vs multi-hop | per-source tables |
| hard vs easy distractors | `--distractors hard` / `--multihop-pad-chars` at label and eval time |
| label cost | `seconds` per record, reader calls = Σ(K+1) |

## 7. Compute plan (4×H100)

Label generation dominates: per reader ≈ Σ_docs (K+1) ≈ 3 sources × 3000 docs × 65 ≈ 0.6M generations
(3 readers ≈ 1.8M). Throughput must be measured on the cluster with a pilot
(`N_TRAIN=50 N_DEV=20 N_TEST=30 ./run_pipeline.sh`) before committing to N. Pruner training: five runs,
one GPU each, a few hours. Evaluation: 5 sources × 500 docs × (arms × ratios) × 4 readers.

## 8. Known gaps / to verify on the cluster

- `provence:` and `llmlingua2` arms use the APIs documented on their model cards; not executed locally.
- vLLM backend (`VLLMReader`) not executed locally (no GPU); HF backend is tested (batched ≡ sequential).
- Baselines not implemented: RECOMP, EXIT, LongLLMLingua, CORE-RAG, ContextCite-at-inference (the
  "unamortized" reference: `oracle_beta` on test documents is its budget-matched equivalent, at K reader
  calls per document).
- Novelty re-check right before submission (keywords from the proposal: "F1 surrogate pseudo-label
  pruner", "amortized attribution compression").
