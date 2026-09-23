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
```

## Run

```bash
pip install -r requirements.txt
cp .env.example .env           # all settings + HF_TOKEN (a WRITE token) in one file; .env is gitignored

# pilot first (~1/10 of the data, every stage): command-line variables override .env
N_TRAIN=300 N_DEV=30 N_TEST=50 RUN_ROOT=/mnt/hps/anhm-paper/ttcompress/runs/pilot HF_RUN_NAME=pilot ./run_pipeline.sh

# full run: .env.example is the full configuration (3000 / 300 / 500; labels measured by the pilot are reused)
./run_pipeline.sh
STAGES="select answer report" \
  EXTRA_ARMS="xprovence=provence:naver/xprovence-reranker-bgem3-v1,llmlingua2" ./run_pipeline.sh
# ablation with a different label/selection setting -> its own RUN_ROOT
MEASURE_ARGS="--distractors hard" SELECT_ARGS="--distractors hard" RUN_ROOT=runs/hard ./run_pipeline.sh
```

Stages: `preflight` (GPU count, imports, CPU unit tests) → `prefetch` (every dataset and model downloaded
once, in one process; gated models fail here, not hours later) → `labels` → `fit` → `ensemble` → `train`
→ `select` → `answer` → `report` → `upload` (every pruner to its own model repo
`<namespace>/ttcompress-<run>-<pruner>`, evaluation outputs and labels to the dataset repo
`<namespace>/ttcompress-<run>-eval`; private by default, namespace defaults to the token's user;
`python scripts/upload_hf.py --dry-run ...` lists what would be uploaded). Readers matching `LARGE_READER_PATTERN` (default 30–79B) run with
tensor parallelism over `TP_LARGE` GPUs; the answer stage works with any process count, independent of
how many shards `select` used. A scheduler-provided `CUDA_VISIBLE_DEVICES` is respected. A failed
shard prints the tail of its log (`$RUN_ROOT/logs/<stage>_shard<k>.log`) and stops the run; re-running
continues from what is on disk. Label and selection dirs record their settings and refuse to be resumed
with different ones.

Outputs: `labels/raw/<reader>/<source>_<split>/` (masks + outcomes per document),
`labels/fit/<reader|ensemble>/<target>/<source>_<split>/` (labels + `summary.json` with label-quality
diagnostics), `models/<pruner>/` (+ `train_log.json`), `results/eval_test/report.{json,md}`.

Every stage skips work already on disk, so a crashed or pre-empted job is simply re-run.

## Discipline built into the code

- **Splits cannot leak through a seed** — there is none: dev/test membership is a hash of the article
  title (UIT-ViQuAD), the passage (XQuAD-vi) or the example id; `tests/test_sources.py` checks
  disjointness on the real data.
- **Test is only touched by `evaluate.py select --split test`**; α is chosen on dev measurements and the
  pruner's best epoch on dev labels, both reader-free.
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
