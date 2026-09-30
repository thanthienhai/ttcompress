# ttcompress — Amortized Utility Attribution for Reader-Agnostic Context Compression

Research code for [`METHOD_SPEC.md`](METHOD_SPEC.md): measure how much each chunk of a context is worth to
a reader (random chunk ablation → ridge surrogate of the downstream F1), distill that attribution into a
query-aware cross-encoder that prunes a document in one forward pass, and evaluate it at fixed token
budgets against the answer-span oracle and published compressors — per source, per reader, with
upgrade retention across readers.

The previous direction (Position-Calibrated Selection + query-agnostic outcome relevance) was retired
after `reports/RUN_REPORT_2026-09-23.md`; its code and specs are in git history (commit `3af893f`).

## Layout

```
ttcompress/
  data.py              QADocument (question, ordered chunks, gold chunks, cluster id) + jsonl io
  sources.py           UIT-ViQuAD / XQuAD-vi haystacks, VIMQA / HotpotQA / 2Wiki multi-hop; hash-based splits
  reader.py            HF (batched) and vLLM readers; answer generation + teacher-forced answer log-prob
  attribution.py       Stage A: masks, ridge surrogate, CV alpha, label diagnostics, ensembles
  pruner.py            chunk-scoring cross-encoder, window packing, inference scorer
  pruner_training.py   Stage B: label sources (beta / ensemble / span), ListNet + MSE / BCE, dev metrics
  selection.py         budget-aware selection + every evaluation arm
  metrics.py           EM / token F1 / recall, cluster + paired bootstrap, upgrade retention
generate_labels.py     measure (GPU) / fit (CPU) / ensemble (CPU)
train_pruner.py        distill labels into a pruner
evaluate.py            select (compress once) / answer (per reader) / report (CPU)
run_pipeline.sh        the whole thing on N GPUs, resumable
scripts/followup.sh    the follow-up experiments of 2026-09-30 on an existing run
scripts/bench_latency.py  selection latency, one arm at a time on one GPU
```

## Run

```bash
pip install -r requirements.txt
cp .env.example .env           # all settings + HF_TOKEN (a WRITE token) in one file; .env is gitignored

# smoke test on a new pod/image first: every stage on a few documents (-> runs/smoke, ~1 h of model
# start-ups, never uploads) -- catches package installs, vLLM, XProvence / LLMLingua-2 and path problems
SMOKE=1 ./run_pipeline.sh

# pilot (~1/10 of the data, every stage): command-line variables override .env
N_TRAIN=300 N_DEV=30 N_TEST=50 RUN_ROOT=/mnt/hps/anhm-paper/ttcompress/runs/pilot HF_RUN_NAME=pilot ./run_pipeline.sh

# mid-scale go/no-go before the full run: does the gap to `embed` / `reranker` (the untrained backbone)
# grow with training data? Documents are nested across n, so pilot labels are reused.
N_TRAIN=800 N_DEV=150 N_TEST=200 RUN_ROOT=/mnt/hps/anhm-paper/ttcompress/runs/mid HF_RUN_NAME=mid ./run_pipeline.sh

# full run: .env.example is the full configuration (3000 / 300 / 500; labels measured by earlier runs are reused)
./run_pipeline.sh
# re-evaluate without the published baselines (e.g. if XProvence's remote code fails on the cluster)
STAGES="select answer report" EXTRA_ARMS= ./run_pipeline.sh
# distractor ablation (labels + evaluation): its own suffixed LABELS / RUN_ROOT, e.g. runs/main_distractors-hard
DISTRACTORS=hard ./run_pipeline.sh
MULTIHOP_PAD_CHARS=20000 ./run_pipeline.sh           # multi-hop docs padded with easy distractors

# side by side (descriptive): distractor ablation, or the data-size curve mid -> full
python scripts/compare_runs.py --primary-reader Qwen--Qwen3-8B --out-dir results/compare \
  --run random=runs/main/results/eval_test/report.json --run hard=runs/main_distractors-hard/results/eval_test/report.json
```

### Follow-up to the full run (2026-09-30)

The per-document analysis of the full run (METHOD_SPEC.md §9) found that paragraph selection, not the labels,
limits multi-hop F1 at 8x, that `oracle_beta`'s single-hop lead exists only on its labeling reader, and that
the single-hop sources are saturated at 4x / 8x. `scripts/followup.sh` runs the experiments that follow from
it on the existing `RUN_ROOT`, each resumable, cheapest first:

```bash
bash scripts/followup.sh                 # bench arms oracle units hard compare
bash scripts/followup.sh bench arms      # or some of them
```

`bench`: selection latency one arm at a time (`STAGES=bench`); `arms`: `sent+` / `fill+` arms, LLMLingua with a
token target, 16x / 32x on single-hop (`FOLLOWUP_ARMS=1 EXTRA_RATIOS_SINGLE=16,32`); `oracle`: `ORACLE_N=500`;
`units`: `MULTIHOP_UNITS=sentence` ablation; `hard`: `DISTRACTORS=hard` ablation. The confirmatory families stay
on the pre-registered 4x / 8x (`CONFIRM_RATIOS`); the report's first version is kept as `first_report.*`, and
`report.md` gains exploratory diagnostics (answer coverage, bridge effect, selection overlap, oracle gap by reader).

### The whole campaign as one job

When the cluster only accepts a submitted script (no shell on the pod), build one self-contained file and
submit it as the job's command (`bash ttcompress_job.sh`):

```bash
python scripts/build_cluster_job.py        # -> dist/ttcompress_job.sh (the repository embedded, no secrets)
python scripts/build_cluster_job.py --set PHASES="main abl_hard" --set MAIN_SIZES="2000 200 500"   # variants
```

The job unpacks the code to `$BASE/code/<build>` and runs `scripts/cluster_campaign.sh`: `smoke` → `probe`
(each published compressor alone on the smoke run; one that cannot run there is dropped, not fatal) → `pilot`
→ `main` → `abl_hard` / `abl_pad` → `compare` → `upload`. Re-submitting it resumes: finished phases are
skipped, a compressor that failed is probed again and, once it works, added to the finished runs. Progress
and every report go to stdout and to the private dataset repo `<token user>/ttcompress-campaign-<CAMPAIGN>`
(`STATUS.md`, updated after each phase and every 30 min, with a projection of the full run from the pilot).
Settings and their defaults: the top of `scripts/cluster_campaign.sh`; the HF token is read on the cluster
(`HF_TOKEN` in the job, else `HF_TOKEN_SOURCES`).

Training-seed robustness: `EXTRA_SEEDS="1 2"` (default) also trains `ours_beta` / `ours_ens` with seeds 1 and 2
(4 more training runs); the report lists F1 per seed and, per hypothesis, how many supported tests hold for
every seed. `EXTRA_SEEDS=none` turns it off.

Stages: `preflight` (GPU count, imports, baseline packages) → `prefetch` (every dataset and model downloaded
once, in one process; gated models fail here, not hours later; then, when `preflight` is in STAGES too, the
whole CPU unit-test suite, `logs/tests.log`) → `labels` → `fit` → `ensemble` → `train`
→ `select` → `answer` → `report` → `upload` (every pruner to its own model repo
`<namespace>/ttcompress-<run>-<pruner>`, evaluation outputs and labels to the dataset repo
`<namespace>/ttcompress-<run>-eval`; private by default, namespace defaults to the token's user;
`python scripts/upload_hf.py --dry-run ...` lists what would be uploaded). Readers matching `LARGE_READER_PATTERN` (default 30–79B) run with
tensor parallelism over `TP_LARGE` GPUs; the answer stage works with any process count, independent of
how many shards `select` used. A scheduler-provided `CUDA_VISIBLE_DEVICES` is respected. A failed
shard prints the tail of its log (`$RUN_ROOT/logs/<stage>_shard<k>.log`) and stops the run; re-running
continues from what is on disk. Label and selection dirs record their settings and refuse to be resumed
with different ones; a `RUN_ROOT` records its sizes, train sources and label readers (`run_config.txt`) and
refuses fit / train / select with other ones, so every configuration (pilot, mid, full) needs its own.

Outputs: `labels/raw/<reader>/<source>_<split>/` (masks + outcomes per document; `LABELS` is shared by runs,
documents are nested across N), `$RUN_ROOT/labels_fit/<reader|ensemble>/<target>/<source>_<split>/` (this
run's labels, restricted to its N_TRAIN / N_DEV / ORACLE_N documents, + `summary.json` with label-quality
diagnostics), `models/<pruner>/` (+ `train_log.json`), `results/eval_test/report.{json,md}` (hypothesis
verdicts, RQ1 cost table, label quality, per-cell tables, paired differences, upgrade retention) and
`results/eval_test/paper/` (LaTeX tables + CSVs for plots, from `scripts/paper_tables.py`).

Every stage skips work already on disk, so a crashed or pre-empted job is simply re-run.

## Discipline built into the code

- **Splits cannot leak through a seed** — there is none: dev/test membership is a hash of the article
  title (UIT-ViQuAD), the passage (XQuAD-vi) or the example id; `tests/test_sources.py` checks
  disjointness on the real data.
- **Test is only touched by `evaluate.py select --split test`**, and by the `oracle_beta` upper bound: Stage A
  labels of the first `ORACLE_N` test documents (`labels/fit/<primary>/f1/<source>_test`), which training never
  lists (`label_dirs` in `run_pipeline.sh` reads train/dev only). α is chosen on dev measurements and the
  pruner's best epoch on dev labels, both reader-free.
- **Confirmatory tests are fixed in code**: `report.md` opens with one family per hypothesis (METHOD_SPEC.md
  §1), one-sided paired tests Holm-corrected within the family; the big paired table is exploratory.
- **Fixed compressor across readers**: budgets use one reference tokenizer and selections are computed
  once, so reader-to-reader differences are attributable to the reader.
- **Paired statistics**: arms are compared on the same documents with a paired cluster bootstrap.
- **Never pooled across sources/languages**.

## Tests

```bash
python -m pytest -q
```

Unit tests run on tiny cached HF models (`hf-internal-testing/tiny-random-*`); `tests/test_sources.py`
reads the real datasets from the HF cache (downloads on first run).
