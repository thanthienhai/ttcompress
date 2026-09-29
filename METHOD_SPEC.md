# Amortized Utility Attribution for Reader-Agnostic Context Compression — Method Spec

> **Status:** v2. Merges the COLING 2027 proposal (`compass_artifact_…md`) into the v1 spec, keeping only what
> the code, the paper draft and the submission use. Dropped from the proposal: the week-by-week timeline, the
> venue-ranking evidence, datasets that are not used (VCC-Bench, NQ, MKQA, LongBench, ViQA-COVID, ViHERMES,
> VMLU), extensions that were not implemented (multi-action KEEP/DROP/TRUNCATE/SUMMARIZE, chunk-boundary
> ablation) and its compute estimate (superseded by §7). The proposal and the earlier specs
> (`PCS_METHOD_SPEC.md`, `OUTCOME_SUPERVISED_RELEVANCE_SPEC.md`) are in git history. Other source: the review
> of `reports/RUN_REPORT_2026-09-23.md`.

---

## 0. Positioning

### 0.1 Venue

- **COLING 2027**, Macau, 9–14 May 2027 (CORE B / CCF B). Submission only through **ARR**, October 2026
  cycle: submission **Mon 2026-10-12**, commitment after meta-reviews Wed 2026-12-23, notification Wed
  2027-02-10. The October cycle is shared with NAACL 2027; the venue is chosen at commitment.
- Long paper: 8 pages of body; the **Limitations** section is mandatory and not counted; references and
  appendices are unlimited.
- Area: a main area (IR / efficient methods / QA / multilinguality), **not** the "NLP for Linguistics" theme
  track.

### 0.2 Closest work

The recipe — random chunk masks → ridge surrogate of the downstream **F1** per chunk → coefficients distilled
as pseudo-labels into a **separate** cross-encoder pruner that runs in one forward pass — had not been
published as of the proposal. β on its own is not new (framed as "attribution" it is ContextCite): the
contribution is the **amortization** and the **reader-agnostic** labels.

| work | what it does | difference / role here |
|---|---|---|
| ContextCite (NeurIPS 2024, arXiv 2409.00729) | random context ablations + sparse linear surrogate of the answer log-prob; applied to pruning | not amortized: 32 ablations (reader calls) per example at inference match the baselines, 64–256 do better. `oracle_beta` is its budget-matched equivalent |
| AT2 (arXiv 2504.13752) | learns weights over the reader's attention heads to reproduce ablation attributions; one pass; tried pruning on HotpotQA | **closest idea**. Amortizes into the reader itself and targets log-prob; ours is a separate pruner trained on F1 → ablation `pruner_logprob_primary` |
| LooComp (arXiv 2603.09222) | encoder-only (ModernBERT) cross-encoder pruner | **closest architecture**. Binary labels from deterministic leave-one-out, ranking/BCE loss; ours: random masks + ridge F1 surrogate, graded labels |
| CORE-RAG / "Less Is More" (arXiv 2508.19282, ICML 2026) | 1.5B generative compressor trained with GRPO, reward = downstream EM (+3.3 EM over full documents at 3% compression) | optimizes the downstream metric through an RL reward, not a distilled surrogate; no released checkpoint (not run) |
| ECoRAG (arXiv 2506.05167) | compressor trained on evidentiality (does the sentence help produce the gold answer) | uses the answer signal, no random-mask surrogate |
| PoC (arXiv 2603.19733) | predicts performance to choose the compression ratio | no per-chunk attribution |
| Provence (ICLR 2025, arXiv 2501.16214) / XProvence (ECIR 2026, arXiv 2601.18886) | cross-encoder reranker + sequence-labeling pruning; XProvence: `bge-reranker-v2-m3`, trained on 16 languages, Vietnamese zero-shot | mandatory multilingual baseline; same backbone family as ours, so the comparison isolates the supervision |
| EXIT (ACL Findings 2025, arXiv 2412.12559) | adaptive sentence-level extraction, trained on HotpotQA supporting facts | strong multi-hop baseline |
| "Fixed RAG Compression Collapses Measured Reader Scaling" (arXiv 2606.21807) | a fixed compressor hides up to 80% of a reader upgrade (Qwen 7B → GPT-4.1-mini, HotpotQA) and flips 31% of model rankings (LongMemEval-S); toolkit `ragscale` | motivates RQ3; source of the upgrade-retention metric |

No benchmark targets context compression in Vietnamese, and XProvence already covers Vietnamese zero-shot.
That weakens a "Vietnamese only" angle, so Vietnamese is an evaluation dimension (§1), not a contribution of
its own.

Directions weighed in the proposal: **A** (this method) + **C** (reader-agnostic labels, RQ3) are the paper.
**B**, a non-additive / set-level surrogate, is reduced to the additivity diagnostic (§6, risk 3 in §8):
submodular and DPP selection for RAG is crowded (AdaGReS arXiv 2512.25052, GeoRAG, ScalDPP, "What Survives
Into Context" arXiv 2607.00725). **D**, soft-prompt / KV-cache compression (gist tokens, ICAE, xRAG,
500xCompressor, CompLLM, MiniKV), is out of scope: crowded, and too heavy as a main axis on 4×H100.

### 0.3 What changed from the previous direction, and why

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

Three research questions, one hypothesis each. Vietnamese is an evaluation dimension (every result is per
source, three of the five sources are Vietnamese), not a research question of its own.

- **RQ1 (amortization).** A pruner distilled from F1-utility attribution keeps downstream F1 at fixed token
  budgets (1/4, 1/8) while costing one encoder pass instead of K≈64 reader calls.
  *H1:* `ours_beta` beats every non-oracle compressor — `bm25`, `embed`, `reranker` (its untrained backbone),
  `lead`, `random` and every published compressor that was run — at both ratios on every source, and stays
  within 0.05 F1 of `oracle_beta` (the unamortized attribution).
- **RQ2 (survival test).** Utility labels beat answer-span supervision.
  *H2:* on multi-hop (VIMQA, HotpotQA, 2Wiki), `ours_beta` > `span_sup` and > `oracle_span` (the answer-span
  oracle misses bridge paragraphs); on single-hop, `ours_beta` ≈ `span_sup` is the expected outcome
  (β collapses onto the needle — measured directly by `gold_recall_at_g` in Stage A summaries); a win there
  is a bonus, not a requirement.
  **Decision rule (§8, risk 2):** if the multi-hop part fails, move the paper's weight to RQ3.
- **RQ3 (reader-agnostic).** *H3:* `ours_ens` (ensemble labels) retains more of the weak→strong reader
  upgrade than `ours_beta` (single reader), including for the **held-out** reader (Qwen3-32B by default)
  that produced no labels. Upgrade retention per arXiv 2606.21807. Cross-reader agreement of β (Spearman in
  `ensemble/summary.json`) is reported descriptively.

On the Vietnamese sources the comparison with XProvence and LLMLingua-2 is part of H1. If XProvence is
already strong there, the contribution shifts to the benchmark + cross-reader analysis (§8, risk 5).

Everything is reported **per source** (never pooled across sources or languages): the previous run showed a
sign flip between pools.

**Confirmatory tests** (`evaluate.py report`, "Hypotheses" section; fixed before the full run). Each row is
its own family of one-sided paired cluster-bootstrap tests on the primary (label) reader, Holm-corrected
within the family at α = 0.05; the full paired table is exploratory.

| hypothesis | family | test | per source × ratio |
|---|---|---|---|
| H1 | H1 | superiority, `ours_beta` − Y > 0 for Y in {`bm25`, `embed`, `reranker`, `lead`, `random`} and every published compressor that was run (`provence`, `xprovence`, `recomp`, `exit`, `llmlingua`, `longllmlingua`, `llmlingua2`) | all sources |
| H1 | H1-oracle | non-inferiority, `ours_beta` − `oracle_beta` > −0.05 (`ORACLE_MARGIN`), on the `ORACLE_N` labeled test docs | all sources |
| H2 | H2a | superiority vs `span_sup`, `oracle_span` | multi-hop |
| H2 | H2b | equivalence (TOST) vs `span_sup`, margin ±0.02 (`EQUIV_MARGIN`) | single-hop |
| H3 | H3 / H3-heldout | superiority of retention(`ours_ens`) − retention(`ours_beta`), same docs, reader pairs with full-context gap ≥ 0.05 (`MIN_UPGRADE_GAP`); pairs with the held-out reader are a separate family | all sources |

H1's cheap baselines and published compressors are one family on purpose: they are the same claim ("beats
every non-oracle compressor"), and Holm over all of them together is the stricter test.

`reranker` is in H1 because it is the pruner's own backbone: beating it is what shows the attribution labels
add something. A test whose two arms differ by > 10% in realized tokens is flagged `budget ≠` (text arms
compress to a rate, not a hard budget). RQ1's cost claim is
the "Cost" table (label reader calls and seconds per document vs selection ms per document).

## 2. Data (`ttcompress/sources.py`)

| source | lang | hop | chunks | train | dev / test |
|---|---|---|---|---|---|
| `uit_viquad` | vi | single | needle paragraph + distractor paragraphs to 30k chars, needle depth uniform | official train minus the dev titles | dev = 10% of the official **train** titles (hashed by title); test = the whole official validation (19 articles). v1 split the 19 validation titles 11/8, leaving an 8-article test set |
| `xquad_vi` | vi | single | same construction | — (eval only) | hashed **by passage** 20/80 |
| `vimqa` | vi | multi | 10 titled paragraphs (native) | official train | official validation / test |
| `hotpotqa` | en | multi | 10 titled paragraphs (distractor setting) | official train | official validation hashed by id 50/50 |
| `2wiki` | en | multi | 10 titled paragraphs | — (eval only) | official validation hashed by id 50/50 |

`uit_viquad` is UIT-ViQuAD **2.0** (`taidng/UIT-ViQuAD2.0`; v1: Nguyen et al. 2020) with the unanswerable
questions (`is_impossible`) dropped. VIMQA (LREC 2022): 10k+ Wikipedia multi-hop QA pairs with sentence-level
supporting facts.

Yes/no questions are dropped. `gold_chunks` = needle / supporting-fact paragraphs. Our labels, loss and
model selection never use it; it is read by the oracle arms (`oracle_span`, `oracle_support`), by the span
control (`pruner_span` → `span_sup`, §4) and by diagnostics only (gold-chunk recall in evaluation,
`gold_recall_at_g` in Stage A, `gold_recall@25%` logged on dev).
`xquad_vi` and `2wiki` are never trained on (cross-dataset transfer; `sources.py` exposes no train split). Options for ablations:
`--distractors hard` (same-article distractors for single-hop), `--multihop-pad-chars` (lengthen multi-hop
documents with easy distractors).

Licenses to confirm before submission: UIT-ViQuAD (research-only terms from UIT NLP group), VIMQA (user
agreement for the full release).

## 3. Stage A — utility attribution (`generate_labels.py`, `ttcompress/attribution.py`)

For each (document, reader): K masks, each chunk kept independently with p ∈ {0.5, 0.25} (cycled),
K = clamp(⌈C+1⌉, 64, 256); masks are seeded by the document id, so **every reader sees identical masks**.
No mask is the empty or the full context (the full context is answered on its own, below): a draw that keeps
no chunk gets one random chunk switched on and, for C > 1, a draw that keeps every chunk gets one switched off.
The masks are therefore independent Bernoulli draws only up to this correction, which touches a mask with
probability (1−p)^C + p^C: nothing for the ~30-chunk single-hop haystacks, 5.6% at C = 10 and p = 0.25
(mean kept 2.55 instead of 2.50), 32% at C = 4. At C = 2 every mask keeps exactly one chunk, so β_1 + β_2 is
absorbed by the intercept and the label only says which chunk scores higher (all the z-score keeps anyway).
The reader answers each masked context (plus the full context, stored as `full_f1`); outcomes are token F1
(the method) and, for the primary reader's train/dev labels only, the teacher-forced answer log-prob
(ablation; timed apart as `seconds_logprob`, so the cost table's `seconds` is the F1 labels' cost). Answer
budgets: 64 new tokens single-hop, 48 multi-hop. `measure_config.json` records every setting that changes a
record (masks, outcomes, budgets, prompt version, data version, backend) and a resumed run must match it. Ridge regression (intercept unpenalized,
one global α chosen by 5-fold CV MSE on **dev** measurements) gives β; the pseudo-label is the
within-document z-score of β.

Label quality is measured, not assumed (`summary.json` per fit dir):
`mean_cv_r2` (held-out R² of the additive surrogate; low on multi-hop = interaction effects),
`gold_recall_at_g` / `gold_mrr` (does β find the evidence?), `n_informative` (documents whose outcome
varied at all), `position_r2` (share of label variance a pooled position prior explains), and for ensembles
`cross_reader_spearman`.

## 4. Stage B — distillation (`train_pruner.py`, `ttcompress/pruner.py`)

Cross-encoder over the backbone's (query, passage) pair frame `<s> question </s></s> chunk_1 … chunk_m </s>`,
the reranker baseline's input (checkpoints trained before 2026-09-28 used `<s> question </s> chunks </s>`;
`pair_format` in `pruner_config.json` records which). The question is cut to 96 tokens and each chunk to 512;
chunks are tokenized separately, so their spans are exact. Mean-pool each chunk's contextualized tokens →
linear head → one logit per chunk. Documents
longer than `--max-len` (default 4096) are packed into windows of whole chunks, each repeating the
question. Backbone `BAAI/bge-reranker-v2-m3` (same initialization family as XProvence, so the comparison
isolates the supervision). Loss: ListNet over the document's chunks (softmax(z/τ) target) + 0.5·MSE; the
span control uses BCE on gold chunks, with `pos_weight` = #non-gold / #gold (at least 1). ListNet is the
cross-entropy summed over chunks, MSE is averaged over chunks: the ListNet value grows like log C only
through the label entropy (no gradient), and per chunk both gradients are ≈ z_i / C at initialization, so
the two terms keep the same weight at any chunk count C. Uninformative documents (the reader's outcome
never varied) are dropped for every pruner, so the span control trains on exactly `pruner_beta_primary`'s
documents. Model selection: best epoch by reader-free dev `ndcg@3_beta` (span:
`gold_recall@25%`).

Pruners trained by `run_pipeline.sh`: `pruner_beta_primary` (ours, one reader), `pruner_beta_ensemble`
(ours, reader-agnostic), `pruner_span` (RQ2 control), `pruner_logprob_primary` (F1 vs log-prob, the AT2
distinction), `pruner_beta_primary_posadj` (position adjustment).

## 5. Evaluation (`evaluate.py`, `ttcompress/selection.py`)

Budget = ⌈full_tokens / ratio⌉ in a fixed reference tokenizer; every chunk arm greedily keeps its best
chunks that fit, in original order, counting the separators between them; the joined text is recounted and
the weakest kept chunk dropped until it fits, so realized tokens never exceed the budget. Arms: `full`, `lead`, `random`, `bm25`, `embed` (bge-m3),
`reranker` (bge-reranker-v2-m3 zero-shot: the pruner's backbone before attribution training),
`oracle_span`, `oracle_support`, `oracle_beta` (upper bound: Stage A labels of the first `ORACLE_N`=100 test
documents per source, primary reader; `ORACLE_N=0` disables it), `pruner:<ckpt>` (any number).
Published compressors (`EXTRA_ARMS`), all held to the same budget:

| arm | method | unit | notes |
|---|---|---|---|
| `provence:<hf id>` | Provence (EN) / XProvence (multilingual) | chunk | reranking score of the released model |
| `recomp` | RECOMP extractive | sentence | NQ checkpoint for single-hop, HotpotQA for multi-hop; dot product, mean pooling |
| `exit` | EXIT | sentence | Gemma-2B-it + released LoRA, P(Yes) given query + containing paragraph; bf16 |
| `llmlingua` | LLMLingua | token | `llmlingua` 0.2.2, small LM Qwen2.5-7B-Instruct (multilingual) |
| `longllmlingua` | LongLLMLingua | token | same LM, README settings (question-aware ranking, `reorder_context=sort`) |
| `llmlingua2` | LLMLingua-2 | token | XLM-R large, MeetingBank |

Sentence arms rank sentences by their score and keep them greedily under the budget (a chunk's title is
kept once per touched chunk); the papers' own cut-offs (top-1/2 sentences, P(Yes) ≥ 0.5) are replaced by the
shared budget. Text arms compress to a rate and are tightened / cut to the budget. EXIT, RECOMP and Provence
are English models: on Vietnamese they are zero-shot. `llmlingua`/`longllmlingua` need transformers ≤ 4.47.1
(microsoft/LLMLingua#210): `run_pipeline.sh` installs transformers 4.46.3 + llmlingua into `LLMLINGUA_SITE`
and runs these two arms in a second select pass with it on PYTHONPATH (the pass reuses the written
documents, so it never imports `datasets`). Not included: RECOMP abstractive, CompAct (abstractive, output
length not controllable), CORE-RAG (no released checkpoint), Selective Context (English/Chinese only).
Selections are computed once and answered by every reader (fixed compressor ⇒ upgrade retention is
meaningful).

Reported per reader × source × ratio × arm: token F1, EM, answer recall, gold-chunk recall, realized
compression, selection latency; cluster-bootstrap CIs (cluster = article for UIT, passage for XQuAD);
paired Δ F1 of our arms vs every other arm with CI, raw p, BH q and Holm p (the whole paired table is
one family; quote q or Holm p); upgrade retention for every reader pair; F1 and
gold recall by needle-depth quintile for single-hop.

## 6. Ablations

| ablation | how |
|---|---|
| F1 vs log-prob surrogate | `pruner_logprob_primary` vs `pruner_beta_primary` |
| additivity | `mean_cv_r2` by hop in Stage A summaries; (pairwise-interaction surrogate: future work) |
| position adjustment | `pruner_beta_primary_posadj`; `position_r2`; depth slices |
| 1 reader vs ensemble | `ours_beta` vs `ours_ens`, incl. held-out reader |
| single- vs multi-hop | per-source tables |
| hard vs easy distractors | `DISTRACTORS=hard` / `MULTIHOP_PAD_CHARS=<n>` (label and eval time, own suffixed dirs; only the sources an ablation changes are re-measured, the others' raw labels are read from the main run); `scripts/compare_runs.py`. `xquad_vi` has no titles, so `hard` equals `random` there |
| training seed | `EXTRA_SEEDS` (default 1 2): `ours_beta_s<k>`, `ours_ens_s<k>`; per-hypothesis "every seed agrees" count, seed SD table |
| label cost | `seconds` per record, reader calls = Σ(K+1) |

## 7. Compute plan (4×H100)

Label generation dominates: per reader ≈ Σ_docs (K+1) ≈ 3 sources × 3000 docs × 65 ≈ 0.6M generations
(3 readers ≈ 1.8M), plus `oracle_beta` test labels: 5 sources × `ORACLE_N` × 65 ≈ 33k (primary reader). Throughput must be measured on the cluster with a pilot
(`N_TRAIN=50 N_DEV=20 N_TEST=30 RUN_ROOT=runs/pilot ./run_pipeline.sh`) before committing to N (`SMOKE=1` first
on a new pod/image). Every configuration needs its own `RUN_ROOT`: `run_pipeline.sh` records the sizes, train
sources and label readers in `$RUN_ROOT/run_config.txt` and refuses to run fit / train / select with other
ones (train skips finished pruners, so a pilot in the full run's `RUN_ROOT` would otherwise leak its pruners
into the full evaluation). The campaign job (`scripts/cluster_campaign.sh`, README) runs smoke → probes → a
300/30/50 pilot → the full run, and projects the full run's hours from the pilot into `STATUS.md`. Pruner training:
five runs + 2 × `EXTRA_SEEDS` seed reruns (nine by default), one GPU each, a few hours per batch of four. Evaluation: 5 sources × 500 docs × (arms × ratios) × 4 readers.

## 8. Risks and decision rules

1. **Idea collision.** AT2 is the closest idea, LooComp the closest architecture (§0.2). Mitigation: state
   the differences (F1 vs log-prob; separate cross-encoder pruner vs the reader's attention heads; random
   masks + ridge vs deterministic leave-one-out), the F1 vs log-prob ablation, the novelty re-check (§9).
   *Decision rule:* if a preprint with exactly this recipe appears, the novelty moves to reader-agnostic
   transfer (RQ3), which is less contested.
2. **β collapses onto the answer span on single-hop** (β ≈ "the chunk containing the gold span").
   Mitigation: weight on multi-hop, where one span is not enough; `span_sup` and `oracle_span` are
   mandatory arms; the collapse is measured by `gold_recall_at_g`. *Decision rule:* if H2 fails on
   multi-hop, the paper's weight moves to RQ3.
3. **Additivity.** The ridge surrogate ignores interactions between chunks, which multi-hop needs. Measured
   by `mean_cv_r2` by hop (§6); a pairwise-interaction surrogate is future work and a Limitations item.
4. **Labels tied to one reader** (the failure mode of arXiv 2606.21807). Turned into RQ3: ≥ 3 label readers,
   ensemble labels, one held-out reader, upgrade retention.
5. **XProvence already strong on Vietnamese.** *Decision rule:* the contribution shifts to the Vietnamese
   evaluation suite + cross-reader analysis instead of a state-of-the-art claim.
6. **Fast-moving landscape.** Several works in §0.2 are 2026 preprints with few citations; their
   `paper/references.bib` entries are marked TODO.

## 9. Known gaps / to verify (cluster, before submission)

Status 2026-09-29. What ran on the cluster so far: the pilot (2026-09-26, vLLM 0.26, 4×H100) and the smoke
test (2026-09-28, commit `80975e9`, vLLM 0.28, 2×H100), every stage, **without** the published compressors
(`EXTRA_ARMS` empty). The current code has not run there yet: its first run is the campaign job
(`scripts/cluster_campaign.sh`), whose `probe` phase runs each published compressor alone on the smoke run and
records `ok` / `failed` with the error under "Published compressors" in `STATUS.md`. Fill in the outcome
here once it has run; a compressor that failed is left out of H1 (report.md names it) until a re-submission
makes it work.

- `provence:` (Provence, XProvence) and `llmlingua2`: APIs checked against the released code (XProvence
  `process()` returns nested `reranking_score` lists and needs spaCy `xx_sent_ud_sm`, downloaded from GitHub;
  LLMLingua-2's `rate` is counted in XLM-R tokens, so `select` tightens the rate and finally cuts the tail to
  the budget, flagged `truncated`). Never executed.
- `recomp`, `exit`, `llmlingua`, `longllmlingua`: checked locally on CPU (real RECOMP checkpoint under
  transformers 5; llmlingua arms with a tiny Qwen2 LM through the pinned site dir, 5.6k-token document);
  not on the cluster. EXIT needs the HF token's account to have accepted the Gemma license
  (`google/gemma-2b-it`); its GPU speed is unknown.
- vLLM backend (`VLLMReader`): ran in the pilot and the smoke test; the prompt and log-prob target changes of
  `b879b37` (no trailing space after the prompt, `answer_target`) run there first in the campaign's smoke.
  No unit test (Linux + GPU only); the HF backend is tested (batched ≡ sequential).
- Not implemented: CORE-RAG (no checkpoint), CompAct / RECOMP abstractive (no length control),
  ContextCite-at-inference (the "unamortized" reference: `oracle_beta` on test documents is its
  budget-matched equivalent, at K reader calls per document).
- Novelty re-check right before submission (keywords: "F1 surrogate pseudo-label pruner", "amortized
  attribution compression").
- Confirm page limits and the area list on 2027.coling-iccl.org: some area details came from the CFP on the
  mailing list, not the full track page.
