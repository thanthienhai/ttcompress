#!/usr/bin/env bash
# Round 3, part 1 (docs/OPS_ROUND3.md): the remaining no-retraining runs for the paper, plus one evaluation
# reader from a family other than Qwen. Every GPU step goes through run_pipeline.sh with the run's own .env,
# reuses the main run's pruners and selections, and is resumable: finished selections / answers are skipped,
# so re-running a step after a crash continues it.
#
#   bash scripts/round3.sh                  # every GPU step, in order (~3.5-4.5 h on 4xH100)
#   bash scripts/round3.sh select answer    # some of them
#   bash scripts/round3.sh upload           # from a CPU pod, after `report`
#
# Steps:
#   check        environment, GPUs idle, inputs on disk, access to the new reader (~1 min)
#   bench        selection latency of ours_sent / fuse_sent, one GPU, nothing else may run (~15-20 min)
#   select       missing follow-up arms, seeds, llmlingua_tt / longllmlingua_tt (~45-60 min); downloads the
#                new reader in the background meanwhile
#   answer       the four existing readers answer every new selection (~1-1.5 h)
#   checkreader  the new reader answers 30 full-context documents per source; stops the script when a source
#                has empty answers or a very low F1 (~10 min)
#   newreader    the new reader answers every selection (~1.25-1.75 h)
#   report       report over the five readers (~5 min)
#   upload       results to the HF dataset (CPU pod)
#
# Knobs: NEW_READER (default google/gemma-3-27b-it), NEW_READER_TP (2), NEW_READER_PATTERN (regex that makes
# run_pipeline.sh give NEW_READER tensor parallelism; default '27b'), FORCE_NEW_READER=1 (run newreader even
# though checkreader failed), HF_EVAL_REPO (default thanthienhai/ttcompress-main-eval).
set -euo pipefail
cd "$(dirname "$0")/.."

ALL_GPU_STEPS=(check bench select answer checkreader newreader report)
STEPS=("$@")
(( ${#STEPS[@]} )) || STEPS=("${ALL_GPU_STEPS[@]}")

setting() {  # setting <KEY> <default>: the environment's value, else .env's, else the default (as run_pipeline.sh)
  local key=$1 value
  if [[ -n "${!key:-}" ]]; then echo "${!key}"; return; fi
  value=$( { grep -E "^[[:space:]]*(export[[:space:]]+)?$key=" .env 2>/dev/null || true; } | tail -n 1 | sed -E "s/^[^=]*=//; s/\r$//; s/^[\"']//; s/[\"']$//")
  echo "${value:-$2}"
}

RUN_ROOT=$(setting RUN_ROOT .)
MODELS=$(setting MODELS "$RUN_ROOT/models")
EVAL_DIR=$(setting EVAL_DIR "$RUN_ROOT/results/eval_test")
LOGS=$(setting LOGS "$RUN_ROOT/logs")
LABEL_READERS=$(setting LABEL_READERS "Qwen/Qwen3-8B Qwen/Qwen3-1.7B aisingapore/Llama-SEA-LION-v3-8B")
EVAL_READERS=$(setting EVAL_READERS "$LABEL_READERS Qwen/Qwen3-32B")
NEW_READER=${NEW_READER:-google/gemma-3-27b-it}
NEW_READER_TP=${NEW_READER_TP:-2}
NEW_READER_PATTERN=${NEW_READER_PATTERN:-27b}
HF_EVAL_REPO=${HF_EVAL_REPO:-thanthienhai/ttcompress-main-eval}
HF_TOKEN=$(setting HF_TOKEN "")
export HF_TOKEN
BENCH_OUT="$EVAL_DIR/bench_latency_sent.json"
PREFETCH_DONE="$LOGS/round3_prefetch_newreader.done"
PREFETCH_FAILED="$LOGS/round3_prefetch_newreader.failed"
PREFETCH_PID="$LOGS/round3_prefetch_newreader.pid"
CHECK_JSON="$LOGS/round3_check_reader.json"
TIMINGS="$LOGS/round3_timings.tsv"
mkdir -p "$LOGS"

# Every select / answer call lists the same arm switches, so a later step never selects fewer arms than an
# earlier one on the same documents.
export FOLLOWUP_ARMS=1 ROUND2_ARMS=1 EXTRA_RATIOS_SINGLE=${EXTRA_RATIOS_SINGLE:-16,32}
export REPORT_OURS=${REPORT_OURS:-ours_beta,ours_ens,ours_fill,ours_sent,fuse_sent}

die() { echo "!! $*" >&2; exit 1; }
gpu_busy() { command -v nvidia-smi >/dev/null && nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; }

run_step() {
  local step=$1 start end
  start=$(date +%s)
  echo "================ round3 step: $step ($(date -u +%FT%TZ)) ================"
  "step_$step"
  end=$(date +%s)
  printf '%s\t%s\t%d min\n' "$step" "$(date -u +%FT%TZ)" $(( (end - start + 59) / 60 )) | tee -a "$TIMINGS"
}

step_check() {
  echo "RUN_ROOT=$RUN_ROOT  MODELS=$MODELS  EVAL_DIR=$EVAL_DIR  LOGS=$LOGS"
  echo "EVAL_READERS=$EVAL_READERS  NEW_READER=$NEW_READER (tp=$NEW_READER_TP)"
  git log --oneline -1
  [[ -f scripts/check_reader.py ]] || die "scripts/check_reader.py missing: git pull"
  for p in "$MODELS/pruner_beta_primary" "$MODELS/pruner_span" "$EVAL_DIR/documents_shard0.jsonl"; do
    [[ -e "$p" ]] || die "missing $p (wrong RUN_ROOT in .env?)"
  done
  python - <<'EOF'
import transformers, vllm
print(f"vllm {vllm.__version__}  transformers {transformers.__version__}")
def ver(v): return tuple(int(x) for x in v.split('.')[:2] if x.isdigit())
if ver(vllm.__version__) < (0, 8) or ver(transformers.__version__) < (4, 50):
    raise SystemExit("!! Gemma 3 needs vllm>=0.8 and transformers>=4.50")
EOF
  python - "$NEW_READER" <<'EOF'
import sys
from huggingface_hub import hf_hub_download
repo = sys.argv[1]
try:
    hf_hub_download(repo, 'config.json')
except Exception as e:  # gated repo without an accepted license, bad token, no network
    raise SystemExit(f"!! cannot read {repo}: {type(e).__name__}: {e}\n   accept the license on huggingface.co/{repo} "
                     "with the account of HF_TOKEN")
print(f"access to {repo}: ok")
EOF
  command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
  if gpu_busy; then echo "!! warning: a process is using a GPU (bench needs the GPU to itself)"; fi
}

step_bench() {
  if [[ -f "$BENCH_OUT" ]]; then echo "bench: $BENCH_OUT exists, skipped"; return; fi
  gpu_busy && die "bench: a process is using a GPU; stop it first (latency would be wrong)"
  EXTRA_ARMS= STAGES=bench BENCH_OUT="$BENCH_OUT" \
  BENCH_ARMS="ours_beta=pruner:$MODELS/pruner_beta_primary,ours_sent=sent+pruner:$MODELS/pruner_beta_primary,fuse_sent=sent+rrf:pruner:$MODELS/pruner_beta_primary|pruner:$MODELS/pruner_span,reranker=reranker:BAAI/bge-reranker-v2-m3" \
    bash run_pipeline.sh
}

start_prefetch() {  # the new reader's weights (~55 GB) download while select runs; network and disk only
  [[ -f "$PREFETCH_DONE" ]] && return
  rm -f "$PREFETCH_FAILED"
  echo "prefetch of $NEW_READER in the background -> $LOGS/round3_prefetch_newreader.log"
  ( if EVAL_READERS="$NEW_READER" EXTRA_ARMS= STAGES=prefetch bash run_pipeline.sh; then touch "$PREFETCH_DONE"
    else touch "$PREFETCH_FAILED"; fi ) > "$LOGS/round3_prefetch_newreader.log" 2>&1 &
  echo $! > "$PREFETCH_PID"
}

prefetch_running() { [[ -f "$PREFETCH_PID" ]] && kill -0 "$(cat "$PREFETCH_PID")" 2>/dev/null; }

wait_prefetch() {
  [[ -f "$PREFETCH_DONE" ]] && return
  [[ -f "$PREFETCH_FAILED" ]] && die "prefetch of $NEW_READER failed: see $LOGS/round3_prefetch_newreader.log"
  prefetch_running || start_prefetch   # select was skipped, or its invocation (and the download) ended
  echo "waiting for the prefetch of $NEW_READER ..."
  while [[ ! -f "$PREFETCH_DONE" ]]; do
    [[ -f "$PREFETCH_FAILED" ]] && die "prefetch of $NEW_READER failed: see $LOGS/round3_prefetch_newreader.log"
    sleep 30
  done
}

step_select() {
  start_prefetch
  EXTRA_ARMS=llmlingua,longllmlingua STAGES=select bash run_pipeline.sh
}

step_answer() {
  EVAL_READERS="$EVAL_READERS" STAGES=answer bash run_pipeline.sh
}

step_checkreader() {
  wait_prefetch
  local status
  set +e
  python scripts/check_reader.py --eval-dir "$EVAL_DIR" --reader-model "$NEW_READER" --tp "$NEW_READER_TP" \
    --n 30 --out "$CHECK_JSON" 2>&1 | tee "$LOGS/round3_check_reader.log"
  status=${PIPESTATUS[0]}
  set -e
  if (( status == 3 )) && [[ "${FORCE_NEW_READER:-0}" != 1 ]]; then
    die "checkreader: gate failed ($CHECK_JSON). Report the output above before running newreader " \
        "(FORCE_NEW_READER=1 bash scripts/round3.sh newreader report overrides)"
  fi
  (( status == 0 || status == 3 )) || die "checkreader: exit status $status (see $LOGS/round3_check_reader.log)"
}

step_newreader() {
  if [[ "${FORCE_NEW_READER:-0}" != 1 ]]; then
    [[ -f "$CHECK_JSON" ]] || die "newreader: run checkreader first"
    python -c "import json, sys; sys.exit(1 if json.load(open(sys.argv[1]))['failed'] else 0)" "$CHECK_JSON" \
      || die "newreader: checkreader failed for $NEW_READER ($CHECK_JSON); FORCE_NEW_READER=1 overrides"
  fi
  wait_prefetch
  EVAL_READERS="$NEW_READER" LARGE_READER_PATTERN="$NEW_READER_PATTERN" TP_LARGE="$NEW_READER_TP" \
    STAGES=answer bash run_pipeline.sh
}

step_report() {
  EVAL_READERS="$EVAL_READERS $NEW_READER" STAGES=report bash run_pipeline.sh
}

hf_up() {  # hf_up <local path> <path in repo> [include patterns...]
  local src=$1 dst=$2; shift 2
  if [[ ! -e "$src" ]]; then echo "   skip (missing): $src"; return; fi
  local inc=()
  (( $# )) && inc=(--include "$@")
  "$HF_CLI" upload "$HF_EVAL_REPO" "$src" "$dst" --repo-type dataset ${inc[@]+"${inc[@]}"}
}

step_upload() {
  HF_CLI=$(command -v hf || command -v huggingface-cli || true)   # huggingface_hub >= 1.0 ships only `hf`
  [[ -n "$HF_CLI" ]] || die "neither hf nor huggingface-cli found (pip install -U huggingface_hub)"
  hf_up "$BENCH_OUT" followup3/bench_latency_sent.json
  hf_up "$EVAL_DIR/report.json" followup3/report.json
  hf_up "$EVAL_DIR/report.md" followup3/report.md
  hf_up "$CHECK_JSON" followup3/check_reader.json
  hf_up "$TIMINGS" followup3/round3_timings.tsv
  hf_up "$EVAL_DIR" followup3/eval_test "answers_*.jsonl" "selections_*.jsonl"
  hf_up "$RUN_ROOT/results/eval_replication" followup2/replication "report.*" "answers_*.jsonl"
  hf_up "${RUN_ROOT}_units-sentence/results/eval_test" followup/units "report.*" "answers_*.jsonl"
  STAGES=upload bash run_pipeline.sh   # pruners of the full run
}

for step in "${STEPS[@]}"; do   # a typo must fail before hours of earlier steps, not after them
  declare -F "step_$step" >/dev/null || die "unknown step '$step' (check bench select answer checkreader newreader report upload)"
done
for step in "${STEPS[@]}"; do run_step "$step"; done
echo "round3: done ($(date -u +%FT%TZ)); timings in $TIMINGS"
