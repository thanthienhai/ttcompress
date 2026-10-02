#!/usr/bin/env bash
# End-to-end pipeline (METHOD_SPEC.md §7): labels from several readers ->
# fit + ensemble -> pruners (single-reader beta, ensemble beta, span control,
# ablations) -> fixed-budget evaluation with every reader -> report.
#
# One process per GPU group (CUDA_VISIBLE_DEVICES), so every process just
# uses 'cuda'. Readers matching LARGE_READER_PATTERN run with tensor
# parallelism over TP_LARGE GPUs (NUM_GPUS / TP_LARGE processes).
# Every stage is resumable: finished documents / selections / answers on
# disk are skipped, so re-running after a crash continues where it stopped.
#
#   NUM_GPUS=4 ./run_pipeline.sh                          # everything
#   STAGES="labels fit" ./run_pipeline.sh                 # a subset
#   N_TRAIN=50 N_DEV=20 N_TEST=30 RUN_ROOT=runs/pilot ./run_pipeline.sh   # pilot
#   SMOKE=1 ./run_pipeline.sh                             # every stage on a few documents -> <runs>/smoke
#   DISTRACTORS=hard ./run_pipeline.sh                    # hard-distractor ablation -> <RUN_ROOT>_distractors-hard
#   MULTIHOP_UNITS=sentence ./run_pipeline.sh             # multi-hop sentence units -> <RUN_ROOT>_units-sentence
#   STAGES=bench ./run_pipeline.sh                        # selection latency, one arm at a time on one GPU
#   (scripts/followup.sh runs the 2026-09-30 follow-up experiments on an existing RUN_ROOT)
set -euo pipefail
cd "$(dirname "$0")"

# Configuration: ./.env (copy of .env.example: every setting + HF_TOKEN; gitignored), and optionally
# ENV_FILE=<another file> loaded before it. Plain KEY=VALUE lines; a variable already set in the
# environment wins, so   N_TRAIN=100 ./run_pipeline.sh   overrides .env for one run.
load_env_file() {
  local file=$1 line key value
  [[ -f "$file" ]] || { echo "ENV_FILE $file not found"; exit 1; }
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%$'\r'}"
    [[ "$line" =~ ^[[:space:]]*(#|$) ]] && continue
    [[ "$line" =~ ^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] \
      || { echo "$file: cannot parse line: $line"; exit 1; }
    key=${BASH_REMATCH[2]}; value=${BASH_REMATCH[3]}
    if [[ "$value" =~ ^\"(.*)\"$ || "$value" =~ ^\'(.*)\'$ ]]; then value=${BASH_REMATCH[1]}; fi
    [[ -z "$value" ]] && continue   # KEY= means "use the default" (an exported empty HF_HOME/HF_TOKEN breaks the hub)
    [[ -n "${!key+x}" ]] || export "$key=$value"
  done < "$file"
}
[[ -n "${ENV_FILE:-}" ]] && load_env_file "$ENV_FILE"
[[ -f .env ]] && load_env_file .env

# SMOKE=1: every stage on a handful of documents, to surface cluster-side failures (package installs, vLLM,
# baseline APIs (EXTRA_ARMS), paths, permissions) in about an hour -- dominated by model start-ups --
# instead of hours into a long run. Overrides the sizes from .env, writes to <parent of RUN_ROOT>/smoke,
# never uploads. Its labels land in the shared LABELS: same settings, nested documents, reused later.
if [[ "${SMOKE:-0}" == 1 ]]; then
  N_TRAIN=8; N_DEV=4; N_TEST=4; ORACLE_N=4
  RUN_ROOT="$(dirname "${RUN_ROOT:-./main}")/smoke"; HF_RUN_NAME=smoke
  STAGES=${STAGES:-"preflight prefetch labels fit ensemble train select answer report"}
  STAGES=${STAGES//upload/}
  TRAIN_ARGS="${TRAIN_ARGS:-} --epochs 1"
  EXTRA_SEEDS=1   # exercise the seed path once, cheaply
fi

NUM_GPUS=${NUM_GPUS:-4}
BACKEND=${BACKEND:-vllm}
# upload is not a default stage: it pushes ~100k files and left the 4xH100 pod idle ~2.5 h in the pilot.
# Run it on its own, from a CPU pod: STAGES=upload bash run_pipeline.sh
STAGES=${STAGES:-"preflight prefetch labels fit ensemble train select answer report"}
# Label readers (the first one is the primary reader).
LABEL_READERS=${LABEL_READERS:-"Qwen/Qwen3-8B Qwen/Qwen3-1.7B aisingapore/Llama-SEA-LION-v3-8B"}
# Evaluation readers: the label readers plus one NEVER used for labels (held-out, RQ3).
EVAL_READERS=${EVAL_READERS:-"$LABEL_READERS Qwen/Qwen3-32B"}
LARGE_READER_PATTERN=${LARGE_READER_PATTERN:-"(3[0-9]|7[0-9])B"}
TP_LARGE=${TP_LARGE:-2}
TRAIN_SOURCES=${TRAIN_SOURCES:-"uit_viquad vimqa hotpotqa"}
# xquad_vi and 2wiki are never trained on: cross-dataset transfer.
EVAL_SOURCES=${EVAL_SOURCES:-"uit_viquad,xquad_vi,vimqa,hotpotqa,2wiki"}
N_TRAIN=${N_TRAIN:-3000}
N_DEV=${N_DEV:-300}
N_TEST=${N_TEST:-500}
RATIOS=${RATIOS:-"4,8"}
# The ratios the confirmatory hypothesis families test (METHOD_SPEC.md §1, fixed before the full run). Ratios
# added later (RATIOS, EXTRA_RATIOS_SINGLE) are reported and compared, never tested in a family.
CONFIRM_RATIOS=${CONFIRM_RATIOS:-4,8}
# Extra ratios for the single-hop sources only (e.g. "16,32"): their ~30-chunk haystacks keep 5-9 chunks at
# 4x-8x, so every chunk arm keeps the needle and the sources cannot separate the methods. Multi-hop documents
# (10 paragraphs) are not compressed further than 8x.
EXTRA_RATIOS_SINGLE=${EXTRA_RATIOS_SINGLE:-}
# FOLLOWUP_ARMS=1: the exploratory arms of the 2026-09-30 follow-up (outside every confirmatory family) --
# ours_beta, span_sup and the reranker scoring sentences (sent+) and filling the budget paragraphs leave with
# sentences (fill+), and LLMLingua / LongLLMLingua asked for the budget in tokens (_tt) when EXTRA_ARMS runs them.
FOLLOWUP_ARMS=${FOLLOWUP_ARMS:-0}
# ROUND2_ARMS=1: the arms of the second follow-up round (docs/PREREG_SENTENCE_REPLICATION.md §9; exploratory on
# the main test set, pre-registered on the replication documents) -- `train` adds pruner_span_ans (binary "chunk
# contains the answer string" label: an annotation-free control), `select` adds span_ans, span_ans_sent and
# fuse_sent (rank fusion of ours_beta and span_sup scoring sentences).
ROUND2_ARMS=${ROUND2_ARMS:-0}
# LOO_ARMS=1: the leave-one-out labeling baseline (docs/prereg_loo.json), primary reader only -- `labels` measures
# LOO masks (one per chunk, each dropping that chunk, + the full context) of the train/dev documents into
# <labels>/raw_loo; `fit` writes the LOO labels (<fit>/loo/...) and the equal-cost control, Ridge on K=11 of the
# existing random masks (<fit>/k11/..., alpha on dev with the same subsampling); `train` adds pruner_loo and
# pruner_k11 (ours_beta's loss) with the EXTRA_SEEDS reruns; `select` adds ours_loo, loo_sent, ours_k11, k11_sent
# and their _s<k> seed arms. LOO_BIN=1 also trains pruner_loo_bin (LooComp-style binary BCE on the LOO labels,
# LOO_BIN_THRESHOLD -> --loo-threshold) with arms loocomp_bin, loocomp_bin_sent (+ seeds). LOO_ONLY=1 (needs
# LOO_ARMS=1): labels / fit / train run only these steps -- an existing run's labels, fits and pruners stay
# untouched (STAGES="labels fit train"); select is resumable and runs only the arms not yet on disk.
LOO_ARMS=${LOO_ARMS:-0}
LOO_BIN=${LOO_BIN:-0}
LOO_BIN_THRESHOLD=${LOO_BIN_THRESHOLD:-}
LOO_ONLY=${LOO_ONLY:-0}
LOO_K_MASKS=11   # in the arm names (ours_k11): not a setting
if [[ "$LOO_ONLY" == 1 && "$LOO_ARMS" != 1 ]]; then echo "!! LOO_ONLY=1 needs LOO_ARMS=1"; exit 1; fi
# A replication on documents no earlier result has seen (docs/PREREG_*.md): EVAL_N documents per source after
# skipping the first EVAL_OFFSET (hash order; the main test set is the first N_TEST), into its own EVAL_DIR.
# N_TEST itself stays the run's (run_config.txt). ONLY_ARMS replaces the arm list of `select` (full specs);
# PREREG_FILE adds the families fixed in that file to the report.
EVAL_N=${EVAL_N:-}
EVAL_OFFSET=${EVAL_OFFSET:-0}
ONLY_ARMS=${ONLY_ARMS:-}
ONLY_ARMS=${ONLY_ARMS// /}
PREREG_FILE=${PREREG_FILE:-}
# oracle_beta (H1 upper bound = the unamortized attribution): Stage A labels of the first ORACLE_N test
# documents per eval source, primary reader only (test docs are nested across n). 0 disables the arm.
ORACLE_N=${ORACLE_N:-100}
(( ORACLE_N <= N_TEST )) || ORACLE_N=$N_TEST
# report: equivalence margin (H2b), non-inferiority margin to oracle_beta (H1), minimum full-context
# weak->strong F1 gap for an upgrade-retention ratio to be read (H3)
EQUIV_MARGIN=${EQUIV_MARGIN:-0.02}
ORACLE_MARGIN=${ORACLE_MARGIN:-0.05}
MIN_UPGRADE_GAP=${MIN_UPGRADE_GAP:-0.05}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}        # longest document measured: ~9k Qwen3 tokens
GPU_MEM=${GPU_MEM:-0.90}
DOCS_PER_CALL=${DOCS_PER_CALL:-16}
BACKBONE=${BACKBONE:-BAAI/bge-reranker-v2-m3}
EMBED_MODEL=${EMBED_MODEL:-BAAI/bge-m3}
# llmlingua / longllmlingua arms need transformers<=4.47.1 (microsoft/LLMLingua#210), the vLLM image ships 5.x:
# that transformers + llmlingua are installed into this directory and put on PYTHONPATH for those arms only.
LLMLINGUA_SITE=${LLMLINGUA_SITE:-$PWD/.llmlingua_site}
LEGACY_ARM_RE='^([^=]*=)?(llmlingua|longllmlingua)(_tt)?(:.*)?$'
# Baseline packages go into the image's environment when its python has pip. An image whose venv has no pip
# (the 2026-09-28 smoke image) gets them here instead, installed with the system pip3 for the venv's Python
# version; this dir is on PYTHONPATH for the select stage only, never for the vLLM stages.
BASELINE_SITE=${BASELINE_SITE:-$PWD/.baseline_site}
# "recomp, exit" must not hide arms from has_arm while evaluate.parse_arms (which strips spaces) runs them
EXTRA_ARMS=${EXTRA_ARMS:-}
EXTRA_ARMS=${EXTRA_ARMS// /}
RUN_ROOT=${RUN_ROOT:-.}
# Distractor ablation (METHOD_SPEC.md §6): DISTRACTORS=hard (single-hop haystacks padded from the needle's own
# article) and/or MULTIHOP_PAD_CHARS=<n> (multi-hop documents lengthened with easy distractors), applied to
# labels AND evaluation. Labels, fits, models and results get their own suffixed directories, so the
# ablation never mixes with the main run (the label dirs would refuse mixed settings anyway) -- except the
# raw labels of the sources an ablation does not change (DISTRACTORS: single-hop only; MULTIHOP_PAD_CHARS:
# multi-hop only), which are read from the main run's LABELS instead of being measured again (raw_root).
DISTRACTORS=${DISTRACTORS:-}
MULTIHOP_PAD_CHARS=${MULTIHOP_PAD_CHARS:-}
# MULTIHOP_UNITS=sentence: multi-hop documents with one chunk per sentence (supporting-fact sentences as gold),
# for labels, pruners and evaluation alike -- paragraph selection keeps ~1 of 10 paragraphs at 8x. An ablation
# like the two above (own suffixed dirs, single-hop labels read from the main run).
MULTIHOP_UNITS=${MULTIHOP_UNITS:-}
[[ "$MULTIHOP_UNITS" == paragraph ]] && MULTIHOP_UNITS=""
LABELS_MAIN=${LABELS:-$RUN_ROOT/labels}
ABLATION=${DISTRACTORS:+_distractors-$DISTRACTORS}${MULTIHOP_PAD_CHARS:+_pad-$MULTIHOP_PAD_CHARS}${MULTIHOP_UNITS:+_units-$MULTIHOP_UNITS}
if [[ -n "$ABLATION" ]]; then
  RUN_ROOT="$RUN_ROOT$ABLATION"
  [[ -n "${LABELS:-}" ]] && LABELS="$LABELS$ABLATION"
  HF_RUN_NAME=$(basename "$RUN_ROOT")
  doc_flags="${DISTRACTORS:+ --distractors $DISTRACTORS}${MULTIHOP_PAD_CHARS:+ --multihop-pad-chars $MULTIHOP_PAD_CHARS}${MULTIHOP_UNITS:+ --multihop-units $MULTIHOP_UNITS}"
  MEASURE_ARGS="${MEASURE_ARGS:-}$doc_flags"   # argparse: the last --distractors wins over .env's
  SELECT_ARGS="${SELECT_ARGS:-}$doc_flags"
fi
# LABELS/raw: reader measurements, expensive, SHARED across runs (nested documents: a larger N reuses a
# smaller run's). FIT: ridge fits + ensembles, cheap, PER RUN and restricted to this run's N_TRAIN / N_DEV /
# ORACLE_N documents -- a shared fit dir would train a small-N run on whatever a larger run measured.
LABELS=${LABELS:-$RUN_ROOT/labels}
SINGLE_HOP_SOURCES=$(python -c "from ttcompress.sources import HOP; print(' '.join(s for s, h in HOP.items() if h == 'single'))")
raw_root() {  # raw_root <source>: where this run's Stage A records of <source> live (the main run's unless
              # the ablation changes that source's documents)
  local single=0; [[ " $SINGLE_HOP_SOURCES " == *" $1 "* ]] && single=1
  if { (( single )) && [[ -n "$DISTRACTORS" ]]; } || { (( ! single )) && [[ -n "$MULTIHOP_PAD_CHARS$MULTIHOP_UNITS" ]]; }; then
    echo "$LABELS/raw"
  else
    echo "$LABELS_MAIN/raw"
  fi
}
FIT=${FIT:-$RUN_ROOT/labels_fit}
MODELS=${MODELS:-$RUN_ROOT/models}
EVAL_DIR=${EVAL_DIR:-$RUN_ROOT/results/eval_test}
LOGS=${LOGS:-$RUN_ROOT/logs}
# Extra flags passed through verbatim, e.g. MEASURE_ARGS="--distractors hard" for the hard-negative ablation
# (use a separate RUN_ROOT per ablation: label and selection dirs refuse mixed settings).
MEASURE_ARGS=${MEASURE_ARGS:-}
TRAIN_ARGS=${TRAIN_ARGS:-}
# Training-seed robustness: ours_beta and ours_ens are also trained with these seeds (arms ours_beta_s<k>,
# ours_ens_s<k>); the report shows F1 per seed and whether every seed agrees with the H1 verdicts.
# EXTRA_SEEDS=none turns it off (an empty KEY= in .env means "default").
EXTRA_SEEDS=${EXTRA_SEEDS:-"1 2"}
[[ "$EXTRA_SEEDS" == none ]] && EXTRA_SEEDS=""
SELECT_ARGS=${SELECT_ARGS:-}
ANSWER_ARGS=${ANSWER_ARGS:-}
# HuggingFace upload (stage `upload`): pruners -> model repos, eval outputs (+ labels) -> one dataset repo.
# HF_NAMESPACE empty = the HF_TOKEN owner; HF_RUN_NAME defaults to the RUN_ROOT folder name.
HF_NAMESPACE=${HF_NAMESPACE:-}
HF_REPO_PREFIX=${HF_REPO_PREFIX:-ttcompress}
HF_RUN_NAME=${HF_RUN_NAME:-$(basename "$RUN_ROOT")}
HF_PRIVATE=${HF_PRIVATE:-true}
HF_UPLOAD_LABELS=${HF_UPLOAD_LABELS:-false}

echo "== config${ENV_FILE:+ ($ENV_FILE)}: RUN_ROOT=$RUN_ROOT LABELS=$LABELS FIT=$FIT NUM_GPUS=$NUM_GPUS BACKEND=$BACKEND"
echo "   N_TRAIN=$N_TRAIN N_DEV=$N_DEV N_TEST=$N_TEST ORACLE_N=$ORACLE_N RATIOS=$RATIOS STAGES=\"$STAGES\""
echo "   LOO_ARMS=$LOO_ARMS LOO_BIN=$LOO_BIN LOO_ONLY=$LOO_ONLY FOLLOWUP_ARMS=$FOLLOWUP_ARMS ROUND2_ARMS=$ROUND2_ARMS"
echo "   EXTRA_SEEDS=\"$EXTRA_SEEDS\"${ABLATION:+ ABLATION=$ABLATION}${SMOKE:+ SMOKE=$SMOKE}"
echo "   LABEL_READERS=\"$LABEL_READERS\""
echo "   EVAL_READERS=\"$EVAL_READERS\""
echo "   TRAIN_SOURCES=\"$TRAIN_SOURCES\" EVAL_SOURCES=$EVAL_SOURCES HF_TOKEN=$([[ -n "${HF_TOKEN:-}" ]] && echo set || echo unset)"
echo "   HF_HOME=${HF_HOME:-~/.cache/huggingface} HF_NAMESPACE=${HF_NAMESPACE:-<token user>} HF_RUN_NAME=$HF_RUN_NAME HF_PRIVATE=$HF_PRIVATE"

export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export PYTHONUNBUFFERED=1

tag() { echo "${1//\//--}"; }
first() { echo "$1"; }
PRIMARY_MODEL=$(first $LABEL_READERS)
PRIMARY=$(tag "$PRIMARY_MODEL")
has_stage() { [[ " $STAGES " == *" $1 "* ]]; }
# A from-scratch smoke run (preflight in STAGES) starts from an empty smoke dir: a previous smoke with another
# GPU count would otherwise make select refuse its num_shards and train skip its finished runs. Only the
# smoke's own RUN_ROOT is removed; its labels live in the shared LABELS and are kept (SMOKE_KEEP=1: keep all).
if [[ "${SMOKE:-0}" == 1 && "${SMOKE_KEEP:-0}" != 1 ]] && has_stage preflight \
   && [[ "$(basename "$RUN_ROOT")" == smoke* && -d "$RUN_ROOT" ]]; then
  echo "== SMOKE: removing the previous smoke run $RUN_ROOT (labels in $LABELS are kept)"
  rm -rf -- "$RUN_ROOT"
fi
tp_for() { if [[ "$1" =~ $LARGE_READER_PATTERN ]]; then echo "$TP_LARGE"; else echo 1; fi; }
mkdir -p "$LOGS"
# One RUN_ROOT = one configuration. Its fits, pruners and selections are restricted to these sizes, and train
# skips any pruner whose train_log.json exists: a pilot launched without its own RUN_ROOT (landing in .env's
# runs/main) would make the later full run evaluate the pilot's pruners. Checked before the labels stage, so
# a mismatch costs nothing. The labels stage alone is exempt: LABELS is shared and nested across N.
if has_stage fit || has_stage train || has_stage select; then
  run_config="N_TRAIN=$N_TRAIN N_DEV=$N_DEV N_TEST=$N_TEST TRAIN_SOURCES=$(echo $TRAIN_SOURCES) LABEL_READERS=$(echo $LABEL_READERS)"
  mkdir -p "$RUN_ROOT"
  if [[ ! -f "$RUN_ROOT/run_config.txt" ]]; then
    # outputs without the record come from a run before this check (older code, sizes unknown)
    for d in "$FIT" "$MODELS" "$EVAL_DIR"; do
      if [[ -n "$(ls -A "$d" 2>/dev/null)" ]]; then
        echo "!! $d holds outputs of a run that predates run_config.txt; give this run its own RUN_ROOT"; exit 1
      fi
    done
    echo "$run_config" > "$RUN_ROOT/run_config.txt"
  elif [[ "$(cat "$RUN_ROOT/run_config.txt")" != "$run_config" ]]; then
    echo "!! $RUN_ROOT belongs to another configuration:"
    echo "     it holds: $(cat "$RUN_ROOT/run_config.txt")"
    echo "     this run: $run_config"
    echo "   Give this run its own RUN_ROOT (a pilot: RUN_ROOT=<runs>/pilot), or start this one over by removing"
    echo "   $FIT $MODELS $EVAL_DIR and $RUN_ROOT/run_config.txt (the labels in $LABELS can stay)."
    exit 1
  fi
fi
# Physical GPU ids: honour a scheduler-provided CUDA_VISIBLE_DEVICES (Slurm etc.), else 0..NUM_GPUS-1.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
else mapfile -t GPU_IDS < <(seq 0 $((NUM_GPUS - 1))); fi
(( ${#GPU_IDS[@]} >= NUM_GPUS )) || { echo "NUM_GPUS=$NUM_GPUS but only ${#GPU_IDS[@]} GPU id(s): ${GPU_IDS[*]}"; exit 1; }
gpu_group() {  # gpu_group <first-slot> <count> -> "id,id"
  local out=""; for ((j = $1; j < $1 + $2; j++)); do out="$out,${GPU_IDS[$j]}"; done; echo "${out#,}"
}

# run_sharded <name> <gpus-per-process> <cmd...>
# Launches NUM_GPUS/gpp processes; process s sees GPUs [s*gpp, (s+1)*gpp) and gets
# --shard s --num-shards P. On failure, prints the tail of each failing log and exits.
run_sharded() {
  local name=$1 gpp=$2; shift 2
  local procs=$(( NUM_GPUS / gpp ))
  (( procs >= 1 )) || { echo "need at least $gpp GPUs for $name"; exit 1; }
  local pids=() logs=()
  for ((s = 0; s < procs; s++)); do
    local gpus; gpus=$(gpu_group $((s * gpp)) "$gpp")
    local log="$LOGS/${name}_shard${s}.log"
    echo "   [$name] shard $s/$procs on GPU $gpus -> $log"
    CUDA_VISIBLE_DEVICES=$gpus "$@" --shard "$s" --num-shards "$procs" >> "$log" 2>&1 &
    pids+=($!); logs+=("$log")
  done
  local failed=0
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      failed=1
      echo "!! [$name] shard $i FAILED -- last lines of ${logs[$i]}:"; tail -n 25 "${logs[$i]}"
    fi
  done
  (( failed == 0 )) || { echo "!! stage $name failed; fix and re-run (finished work is kept)"; exit 1; }
}
# Job pool: at most POOL_N jobs at once; a job starts as soon as a slot frees (the training stage uses the
# slot as its GPU). A failed job stops the pipeline with the tail of its log.
pool_start() { POOL_N=$1; POOL_PID=(); POOL_NAME=(); POOL_LOG=(); }
pool_reap() {  # pool_reap <slot>
  local s=$1
  [[ -n "${POOL_PID[$s]:-}" ]] || return 0
  if ! wait "${POOL_PID[$s]}"; then
    echo "!! ${POOL_NAME[$s]} FAILED -- last lines of ${POOL_LOG[$s]}:"; tail -n 25 "${POOL_LOG[$s]}"; exit 1
  fi
  POOL_PID[$s]=""
}
pool_slot() {  # sets POOL_SLOT to a free slot, reaping a finished job
  local s
  while true; do
    for ((s = 0; s < POOL_N; s++)); do
      if [[ -z "${POOL_PID[$s]:-}" ]] || ! kill -0 "${POOL_PID[$s]}" 2>/dev/null; then
        pool_reap "$s"; POOL_SLOT=$s; return
      fi
    done
    sleep 1
  done
}
pool_run() {  # pool_run <name> <log> <cmd...>: waits for a free slot, then starts cmd there
  local name=$1 log=$2; shift 2
  pool_slot
  "$@" > "$log" 2>&1 &
  POOL_PID[$POOL_SLOT]=$!; POOL_NAME[$POOL_SLOT]=$name; POOL_LOG[$POOL_SLOT]=$log
}
pool_wait() { local s; for ((s = 0; s < POOL_N; s++)); do pool_reap "$s"; done; }
CPU_JOBS=${CPU_JOBS:-$(( $(nproc 2>/dev/null || echo 4) < 8 ? $(nproc 2>/dev/null || echo 4) : 8 ))}

# Background shards / training runs ignore SIGINT in a non-interactive shell: without this, Ctrl-C (or a
# failed stage) leaves them holding the GPUs and the relaunch runs out of memory.
stop_children() { local p; p=$(jobs -p); [[ -z "$p" ]] || kill $p 2>/dev/null || true; }
trap 'stop_children' EXIT
trap 'stop_children; exit 130' INT TERM

# Published-compressor arms need packages the vLLM image does not ship: llmlingua2 -> `llmlingua`;
# provence:/XProvence -> `spacy` + its multilingual sentence model `xx_sent_ud_sm` (loaded when the
# remote modeling code is imported). Installed on demand with the image's torch / transformers / vllm /
# numpy / tokenizers pinned as constraints, so pip fails instead of swapping the CUDA build vLLM needs.
has_pip() { python -m pip --version >/dev/null 2>&1; }
bpy() { PYTHONPATH="$BASELINE_SITE${PYTHONPATH:+:$PYTHONPATH}" python "$@"; }   # python + baseline packages
image_pins() {  # the image's versions of what vLLM needs, as pip constraints (importlib: works without pip)
  python - <<'PY'
import importlib.metadata as m
for p in ('torch', 'transformers', 'vllm', 'numpy', 'tokenizers'):
    try:
        print(f'{p}=={m.version(p)}')
    except m.PackageNotFoundError:
        pass
PY
}
pip_target() {  # pip_target <dir> <pip install args...>: into a directory, with whichever pip exists
  local dir=$1; shift
  if has_pip; then
    python -m pip install --quiet --upgrade --target "$dir" "$@"
  elif command -v pip3 >/dev/null 2>&1; then
    # the system pip3 may belong to another Python: ask for wheels of the venv's version
    pip3 install --quiet --upgrade --target "$dir" --only-binary=:all: \
      --python-version "$(python -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')" "$@"
  else
    echo "!! $(command -v python) has no pip and there is no pip3 on PATH"; return 1
  fi
}
# pip_pinned <pip install args...>: into the image env (image pins as constraints, so pip fails instead of
# swapping the CUDA build vLLM needs), or into BASELINE_SITE when the image's python has no pip. Packages that
# depend on torch (peft, accelerate, llmlingua) are installed --no-deps: into a --target dir pip would
# otherwise download its own torch and shadow the image's.
pip_pinned() {
  local pins rc=0; pins=$(mktemp); image_pins > "$pins"
  if has_pip; then
    python -m pip install --quiet "$@" -c "$pins" || rc=$?
  else
    echo "   (no pip in $(command -v python): installing into $BASELINE_SITE with the system pip3)"
    pip_target "$BASELINE_SITE" "$@" -c "$pins" || rc=$?
  fi
  if (( rc )); then
    echo "!! pip install $* failed (pinned: $(paste -sd' ' "$pins")); install by hand or drop the arm from EXTRA_ARMS"
    rm -f "$pins"; exit 1
  fi
  rm -f "$pins"
}
has_arm() {  # has_arm <arm name>: EXTRA_ARMS has it, bare or labeled, with or without ':<arg>' (or its _tt variant)
  local re=",([^=,]*=)?$1(_tt)?(:[^,]*)?,"
  [[ ",${EXTRA_ARMS:-}," =~ $re ]]
}
llmlingua_site_ok() {
  PYTHONPATH="$LLMLINGUA_SITE" python -c "import transformers, llmlingua; assert transformers.__version__ == '4.46.3'" 2>/dev/null
}
ensure_baseline_deps() {
  if has_arm llmlingua2 && ! bpy -c "import llmlingua" 2>/dev/null; then
    echo "== installing llmlingua (EXTRA_ARMS has llmlingua2)"
    pip_pinned --no-deps llmlingua accelerate nltk joblib click tiktoken
    bpy -c "import llmlingua" || { echo "!! llmlingua installed but does not import"; exit 1; }
  fi
  if has_arm provence && ! bpy -c "import spacy; spacy.load('xx_sent_ud_sm')" 2>/dev/null; then
    echo "== installing spacy + xx_sent_ud_sm (EXTRA_ARMS has a provence: arm)"
    pip_pinned spacy
    # the model wheel directly: `spacy download` shells out to `python -m pip`, which a pip-less venv lacks
    local url; url=$(bpy -c "import spacy; v = spacy.about.__version__.split('.'); t = f'xx_sent_ud_sm-{v[0]}.{v[1]}.0'; print(f'https://github.com/explosion/spacy-models/releases/download/{t}/{t}-py3-none-any.whl')")
    pip_pinned --no-deps "$url" || true
    bpy -c "import spacy; spacy.load('xx_sent_ud_sm')" 2>/dev/null || bpy -m spacy download xx_sent_ud_sm \
      || { echo "!! spacy model xx_sent_ud_sm could not be installed ($url)"; exit 1; }
    bpy -c "import spacy; spacy.load('xx_sent_ud_sm')" || { echo "!! xx_sent_ud_sm does not load"; exit 1; }
  fi
  if has_arm provence && ! bpy -c "import nltk; nltk.data.find('tokenizers/punkt_tab')" 2>/dev/null; then
    echo "== installing nltk + punkt (English Provence splits sentences with nltk)"
    bpy -c "import nltk" 2>/dev/null || pip_pinned --no-deps nltk joblib click
    # nltk.download reports failure by returning False: raise, then check the data really is there
    bpy -c "import nltk; nltk.download('punkt_tab', quiet=True, raise_on_error=True); nltk.download('punkt', quiet=True, raise_on_error=True); nltk.data.find('tokenizers/punkt_tab')" \
      || { echo "!! nltk punkt download failed"; exit 1; }
  fi
  if has_arm exit && ! bpy -c "import peft" 2>/dev/null; then
    echo "== installing peft (EXTRA_ARMS has exit: Gemma-2B + LoRA adapter)"
    pip_pinned --no-deps peft accelerate
    bpy -c "import peft" || { echo "!! peft installed but does not import"; exit 1; }
  fi
  if has_arm llmlingua || has_arm longllmlingua; then
    if ! llmlingua_site_ok; then
      # --no-deps: torch, numpy, safetensors, ... come from the image; only what must differ is installed here
      echo "== installing transformers 4.46.3 + llmlingua 0.2.2 into $LLMLINGUA_SITE (llmlingua/longllmlingua arms)"
      pip_target "$LLMLINGUA_SITE" --no-deps "transformers==4.46.3" "tokenizers==0.20.3" "huggingface_hub==0.26.5" \
        "accelerate==1.1.1" "llmlingua==0.2.2" tiktoken nltk joblib click \
        || { echo "!! could not install into $LLMLINGUA_SITE"; exit 1; }
      llmlingua_site_ok || { echo "!! $LLMLINGUA_SITE does not import transformers 4.46.3 + llmlingua"; exit 1; }
    fi
    # llmlingua reports lengths with tiktoken's gpt-3.5-turbo encoding: fetch it once, not in every shard
    PYTHONPATH="$LLMLINGUA_SITE" python -c "import tiktoken; tiktoken.encoding_for_model('gpt-3.5-turbo')" \
      || { echo "!! tiktoken encoding download failed"; exit 1; }
  fi
}

# The whole unit-test suite on CPU (a few minutes; tests/test_cluster_safety.py runs offline, so the datasets and
# the tiny test models must be cached: after prefetch, or from an earlier run's cache).
TEST_MODELS="hf-internal-testing/tiny-random-gpt2 hf-internal-testing/tiny-random-RobertaModel"
run_tests() {
  echo "== unit tests (log: $LOGS/tests.log)"
  if ! python -m pytest -q -x -p no:cacheprovider tests/ > "$LOGS/tests.log" 2>&1; then
    tail -n 40 "$LOGS/tests.log"; echo "!! unit tests failed"; exit 1
  fi
  tail -n 1 "$LOGS/tests.log"
}

if has_stage preflight; then
  echo "== preflight"
  visible=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true)
  (( ${visible:-0} >= NUM_GPUS )) || { echo "NUM_GPUS=$NUM_GPUS but nvidia-smi sees ${visible:-0} GPU(s)"; exit 1; }
  python - "$BACKEND" <<'PY'
import sys
import torch, transformers, numpy, scipy, pandas, datasets, huggingface_hub  # noqa: F401
print(f"torch {torch.__version__} (cuda {torch.version.cuda}, {torch.cuda.device_count()} GPUs), "
      f"transformers {transformers.__version__}")
assert torch.cuda.is_available(), "torch cannot see CUDA"
if sys.argv[1] == 'vllm':
    import vllm
    print(f"vllm {vllm.__version__}")
PY
  has_stage prefetch || run_tests   # with prefetch in STAGES they run once every dataset and model is cached
  # every package version of this run, for reproducing it (the image, the baseline and llmlingua sites)
  { echo "# $(date -u +%FT%TZ) $(python --version 2>&1) $(command -v python)"
    for site in "" "$BASELINE_SITE" "$LLMLINGUA_SITE"; do
      echo "## ${site:-image}"
      PYTHONPATH="$site" python -c "import importlib.metadata as m, sys; print('\n'.join(sorted({f'{d.metadata[\"Name\"]}=={d.version}' for d in m.distributions(**({\"path\": [sys.argv[1]]} if sys.argv[1] else {}))})))" "$site" 2>/dev/null || true
    done; } > "$LOGS/environment.txt"
  echo "   package versions -> $LOGS/environment.txt"
  ensure_baseline_deps  # an optional baseline must not fail hours later at select
  if has_stage upload; then  # a missing or read-only token should fail now, not after training
    python scripts/upload_hf.py --check ${HF_NAMESPACE:+--namespace "$HF_NAMESPACE"}
  fi
fi

if has_stage prefetch; then
  echo "== prefetch (single process; avoids N processes racing on the HF cache)"
  # --arms adds the hub ids the EXTRA_ARMS baselines load (ttcompress.selection.arm_models)
  all_models=$(echo "$LABEL_READERS $EVAL_READERS $BACKBONE $EMBED_MODEL $TEST_MODELS" | tr ' ' '\n' | awk 'NF && !seen[$0]++' | paste -sd, -)
  all_sources=$(echo "$TRAIN_SOURCES ${EVAL_SOURCES//,/ }" | tr ' ' '\n' | awk 'NF && !seen[$0]++' | paste -sd, -)
  python scripts/prefetch.py --sources "$all_sources" --models "$all_models" --arms "${EXTRA_ARMS:-}" 2>&1 \
    | tee -a "$LOGS/prefetch.log"
  if has_stage preflight; then run_tests; fi   # a launch from scratch: the unit tests, before any GPU hour
fi

# LOO_ONLY=1: the main run's label / fit / train steps are skipped (its outputs stay as they are)
MAIN_READERS=$LABEL_READERS; MAIN_ORACLE_N=$ORACLE_N
[[ "$LOO_ONLY" == 1 ]] && { MAIN_READERS=""; MAIN_ORACLE_N=0; }

if has_stage labels; then
  for reader in $MAIN_READERS; do
    tp=$(tp_for "$reader")
    for src in $TRAIN_SOURCES; do
      for split in train dev; do
        n=$N_TRAIN; [[ $split == dev ]] && n=$N_DEV
        # the log-prob pass (a second full prefill) only where it is trained: pruner_logprob_primary.
        # After $MEASURE_ARGS, so it wins over an --outcomes left in an older .env.
        outcomes=f1; [[ "$reader" == "$PRIMARY_MODEL" ]] && outcomes=f1,logprob
        echo "== labels: $reader $src/$split (n=$n, tp=$tp, outcomes=$outcomes)"
        run_sharded "labels_$(tag "$reader")_${src}_$split" "$tp" \
          python generate_labels.py measure --source "$src" --split "$split" --n "$n" \
          --reader-model "$reader" --backend "$BACKEND" --max-model-len "$MAX_MODEL_LEN" --tp "$tp" \
          --gpu-memory-utilization "$GPU_MEM" --docs-per-call "$DOCS_PER_CALL" --out-root "$(raw_root "$src")" $MEASURE_ARGS \
          --outcomes "$outcomes"
      done
    done
  done
  if (( MAIN_ORACLE_N > 0 )); then  # never read by training: label_dirs() only lists train/dev
    tp=$(tp_for "$PRIMARY_MODEL")
    for src in ${EVAL_SOURCES//,/ }; do
      echo "== labels (oracle_beta): $PRIMARY_MODEL $src/test (n=$ORACLE_N, tp=$tp)"
      run_sharded "labels_${PRIMARY}_${src}_test" "$tp" \
        python generate_labels.py measure --source "$src" --split test --n "$ORACLE_N" \
        --reader-model "$PRIMARY_MODEL" --backend "$BACKEND" --max-model-len "$MAX_MODEL_LEN" --tp "$tp" \
        --gpu-memory-utilization "$GPU_MEM" --docs-per-call "$DOCS_PER_CALL" --out-root "$(raw_root "$src")" $MEASURE_ARGS \
        --outcomes f1
    done
  fi
  if [[ "$LOO_ARMS" == 1 ]]; then
    # same documents (--n, hash order) as the main measure; a separate out-root: measure refuses mixed mask schemes
    tp=$(tp_for "$PRIMARY_MODEL")
    for src in $TRAIN_SOURCES; do
      for split in train dev; do
        n=$N_TRAIN; [[ $split == dev ]] && n=$N_DEV
        echo "== labels (LOO): $PRIMARY_MODEL $src/$split (n=$n, tp=$tp)"
        run_sharded "labels_loo_${PRIMARY}_${src}_$split" "$tp" \
          python generate_labels.py measure --source "$src" --split "$split" --n "$n" \
          --reader-model "$PRIMARY_MODEL" --backend "$BACKEND" --max-model-len "$MAX_MODEL_LEN" --tp "$tp" \
          --gpu-memory-utilization "$GPU_MEM" --docs-per-call "$DOCS_PER_CALL" --out-root "$(raw_root "$src")_loo" $MEASURE_ARGS \
          --mask-scheme loo --outcomes f1
      done
    done
  fi
fi

if has_stage fit; then
  # independent CPU jobs (alpha comes from the raw dev measurements, not from another fit): run them together
  mkdir -p "$LOGS/fit"
  pool_start "$CPU_JOBS"
  for reader in $MAIN_READERS; do
    r=$(tag "$reader")
    targets=f1; [[ "$reader" == "$PRIMARY_MODEL" ]] && targets="f1 logprob"
    for target in $targets; do
      for src in $TRAIN_SOURCES; do
        for split in train dev; do
          echo "== fit: $r $target $src/$split"
          pool_run "fit $r $target $src/$split" "$LOGS/fit/${r}_${target}_${src}_$split.log" \
            env OMP_NUM_THREADS=1 python generate_labels.py fit --raw-dir "$(raw_root "$src")/$r/${src}_$split" \
            --target "$target" --alpha-from "$(raw_root "$src")/$r/${src}_dev" --out-dir "$FIT/$r/$target/${src}_$split" \
            --n "$([[ $split == dev ]] && echo "$N_DEV" || echo "$N_TRAIN")" --alpha-n "$N_DEV"
        done
      done
    done
  done
  if (( MAIN_ORACLE_N > 0 )); then
    pooled_dev=""
    for s in $TRAIN_SOURCES; do pooled_dev="$pooled_dev,$(raw_root "$s")/$PRIMARY/${s}_dev"; done
    for src in ${EVAL_SOURCES//,/ }; do
      # alpha from the source's own dev labels; eval-only sources (xquad_vi, 2wiki) pool the train-source dev labels
      alpha_from=${pooled_dev#,}
      [[ " $TRAIN_SOURCES " == *" $src "* ]] && alpha_from="$(raw_root "$src")/$PRIMARY/${src}_dev"
      echo "== fit (oracle_beta): $PRIMARY f1 $src/test"
      pool_run "fit oracle $src" "$LOGS/fit/${PRIMARY}_f1_${src}_test.log" \
        env OMP_NUM_THREADS=1 python generate_labels.py fit --raw-dir "$(raw_root "$src")/$PRIMARY/${src}_test" \
        --target f1 --alpha-from "$alpha_from" --out-dir "$FIT/$PRIMARY/f1/${src}_test" --n "$ORACLE_N" --alpha-n "$N_DEV"
    done
  fi
  if [[ "$LOO_ARMS" == 1 ]]; then
    # <fit>/loo|k11/<reader>/f1/<set>: one level below the main fits, so label_quality's table keeps its rows
    for src in $TRAIN_SOURCES; do
      for split in train dev; do
        n=$N_TRAIN; [[ $split == dev ]] && n=$N_DEV
        echo "== fit (LOO): $PRIMARY $src/$split"
        pool_run "fit loo $src/$split" "$LOGS/fit/loo_${PRIMARY}_${src}_$split.log" \
          env OMP_NUM_THREADS=1 python generate_labels.py fit --raw-dir "$(raw_root "$src")_loo/$PRIMARY/${src}_$split" \
          --estimator loo --out-dir "$FIT/loo/$PRIMARY/f1/${src}_$split" --n "$n"
        # equal-cost control: the method's Ridge on K=11 of the existing random masks (~ the LOO call count)
        echo "== fit (K=$LOO_K_MASKS masks): $PRIMARY $src/$split"
        pool_run "fit k$LOO_K_MASKS $src/$split" "$LOGS/fit/k${LOO_K_MASKS}_${PRIMARY}_${src}_$split.log" \
          env OMP_NUM_THREADS=1 python generate_labels.py fit --raw-dir "$(raw_root "$src")/$PRIMARY/${src}_$split" \
          --target f1 --max-masks "$LOO_K_MASKS" --mask-seed 0 --alpha-from "$(raw_root "$src")/$PRIMARY/${src}_dev" \
          --out-dir "$FIT/k$LOO_K_MASKS/$PRIMARY/f1/${src}_$split" --n "$n" --alpha-n "$N_DEV"
      done
    done
  fi
  pool_wait
fi

if has_stage ensemble; then
  mkdir -p "$LOGS/ensemble"
  pool_start "$CPU_JOBS"
  for src in $TRAIN_SOURCES; do
    for split in train dev; do
      dirs=""
      for reader in $LABEL_READERS; do dirs="$dirs,$FIT/$(tag "$reader")/f1/${src}_$split"; done
      echo "== ensemble: $src/$split"
      pool_run "ensemble $src/$split" "$LOGS/ensemble/${src}_$split.log" \
        env OMP_NUM_THREADS=1 python generate_labels.py ensemble --fit-dirs "${dirs#,}" --out-dir "$FIT/ensemble/f1/${src}_$split"
    done
  done
  pool_wait
fi

label_dirs() {  # label_dirs <reader-tag-or-ensemble> <target> <split>
  local out=""
  for src in $TRAIN_SOURCES; do out="$out,$FIT/$1/$2/${src}_$3"; done
  echo "${out#,}"
}

if has_stage train; then
  # name | label source | label reader | target | extra flags
  RUNS=(
    "pruner_beta_primary|beta|$PRIMARY|f1|"
    "pruner_beta_ensemble|ensemble|ensemble|f1|"
    "pruner_span|span|$PRIMARY|f1|"
    "pruner_logprob_primary|beta|$PRIMARY|logprob|"
    "pruner_beta_primary_posadj|beta|$PRIMARY|f1|--position-adjust"
  )
  for seed in $EXTRA_SEEDS; do  # name | source | reader | target | extra | seed
    RUNS+=("pruner_beta_primary_s$seed|beta|$PRIMARY|f1||$seed" "pruner_beta_ensemble_s$seed|ensemble|ensemble|f1||$seed")
  done
  # same fit dirs (documents) as pruner_beta_primary; the label itself never reads the reader
  [[ "$ROUND2_ARMS" == 1 ]] && RUNS+=("pruner_span_ans|answer|$PRIMARY|f1|")
  [[ "$LOO_ONLY" == 1 ]] && RUNS=()
  if [[ "$LOO_ARMS" == 1 ]]; then
    # reader field = the fit dir under $FIT (label_dirs): <fit>/loo/<primary>, <fit>/k11/<primary>
    loo_runs=("pruner_loo|beta|loo/$PRIMARY|f1|" "pruner_k$LOO_K_MASKS|beta|k$LOO_K_MASKS/$PRIMARY|f1|")
    [[ "$LOO_BIN" == 1 ]] && loo_runs+=("pruner_loo_bin|loo_bin|loo/$PRIMARY|f1|${LOO_BIN_THRESHOLD:+--loo-threshold $LOO_BIN_THRESHOLD}")
    for run in "${loo_runs[@]}"; do   # seed reruns as for pruner_beta_primary: name_s<k>, 6th field = seed
      RUNS+=("$run")
      for seed in $EXTRA_SEEDS; do RUNS+=("${run%%|*}_s$seed|${run#*|}|$seed"); done
    done
  fi
  # one run per GPU; the next run starts as soon as any GPU frees (9 runs on 4 GPUs: no 1-GPU last wave)
  pool_start "$NUM_GPUS"
  for run in "${RUNS[@]}"; do
    IFS='|' read -r name source reader target extra seed <<< "$run"
    if [[ -f "$MODELS/$name/train_log.json" ]]; then echo "== train $name: done, skipping"; continue; fi
    pool_slot
    echo "== train $name (GPU ${GPU_IDS[$POOL_SLOT]}) -> $LOGS/train_$name.log"
    # shellcheck disable=SC2086
    pool_run "train $name" "$LOGS/train_$name.log" \
      env CUDA_VISIBLE_DEVICES="${GPU_IDS[$POOL_SLOT]}" python train_pruner.py --label-source "$source" \
      --train-labels "$(label_dirs "$reader" "$target" train)" --dev-labels "$(label_dirs "$reader" "$target" dev)" \
      --backbone "$BACKBONE" --grad-checkpointing $extra $TRAIN_ARGS ${seed:+--seed "$seed"} --out-dir "$MODELS/$name"
  done
  pool_wait
fi

if has_stage select; then
  ensure_baseline_deps  # also when select runs without preflight (STAGES="select answer report")
  # reranker = the pruner's backbone zero-shot: ours vs reranker isolates what the attribution labels add
  ARMS="full,lead,random,bm25,embed=embed:$EMBED_MODEL,reranker=reranker:$BACKBONE,oracle_span,oracle_support"
  ARMS="$ARMS,ours_beta=pruner:$MODELS/pruner_beta_primary,ours_ens=pruner:$MODELS/pruner_beta_ensemble"
  ARMS="$ARMS,span_sup=pruner:$MODELS/pruner_span,abl_logprob=pruner:$MODELS/pruner_logprob_primary"
  ARMS="$ARMS,abl_posadj=pruner:$MODELS/pruner_beta_primary_posadj"
  for seed in $EXTRA_SEEDS; do
    ARMS="$ARMS,ours_beta_s$seed=pruner:$MODELS/pruner_beta_primary_s$seed"
    ARMS="$ARMS,ours_ens_s$seed=pruner:$MODELS/pruner_beta_ensemble_s$seed"
  done
  ARMS="$ARMS${EXTRA_ARMS:+,$EXTRA_ARMS}"   # published compressors, see .env.example
  if [[ "$FOLLOWUP_ARMS" == 1 ]]; then
    for pair in "ours|pruner:$MODELS/pruner_beta_primary" "span|pruner:$MODELS/pruner_span" "reranker|reranker:$BACKBONE"; do
      ARMS="$ARMS,${pair%%|*}_sent=sent+${pair#*|},${pair%%|*}_fill=fill+${pair#*|}"
    done
    # seed reruns of the sentence arm: "every seed agrees" for ours_sent, like H1's for ours_beta
    for seed in $EXTRA_SEEDS; do ARMS="$ARMS,ours_sent_s$seed=sent+pruner:$MODELS/pruner_beta_primary_s$seed"; done
    has_arm llmlingua && ARMS="$ARMS,llmlingua_tt"
    has_arm longllmlingua && ARMS="$ARMS,longllmlingua_tt"
  fi
  if [[ "$ROUND2_ARMS" == 1 ]]; then
    ARMS="$ARMS,span_ans=pruner:$MODELS/pruner_span_ans,span_ans_sent=sent+pruner:$MODELS/pruner_span_ans"
    ARMS="$ARMS,fuse_sent=sent+rrf:pruner:$MODELS/pruner_beta_primary|pruner:$MODELS/pruner_span"
  fi
  if [[ "$LOO_ARMS" == 1 ]]; then   # paragraph arm | sentence arm | pruner, + seed reruns of both (docs/prereg_loo.json)
    loo_arms="ours_loo|loo_sent|pruner_loo ours_k$LOO_K_MASKS|k${LOO_K_MASKS}_sent|pruner_k$LOO_K_MASKS"
    [[ "$LOO_BIN" == 1 ]] && loo_arms="$loo_arms loocomp_bin|loocomp_bin_sent|pruner_loo_bin"
    for spec in $loo_arms; do
      IFS='|' read -r para sent pruner <<< "$spec"
      ARMS="$ARMS,$para=pruner:$MODELS/$pruner,$sent=sent+pruner:$MODELS/$pruner"
      for seed in $EXTRA_SEEDS; do
        ARMS="$ARMS,${para}_s$seed=pruner:$MODELS/${pruner}_s$seed,${sent}_s$seed=sent+pruner:$MODELS/${pruner}_s$seed"
      done
    done
  fi
  oracle_flags=()
  if (( ORACLE_N > 0 )); then
    oracle_dirs=""
    for src in ${EVAL_SOURCES//,/ }; do
      d="$FIT/$PRIMARY/f1/${src}_test"
      [[ -f "$d/summary.json" ]] || { echo "!! $d missing: run the labels + fit stages first (ORACLE_N=$ORACLE_N)"; exit 1; }
      oracle_dirs="$oracle_dirs,$d"
    done
    ARMS="$ARMS,oracle_beta"
    oracle_flags=(--oracle-beta-dir "${oracle_dirs#,}")
  fi
  if [[ -n "$ONLY_ARMS" ]]; then
    ARMS="${ONLY_ARMS//@MODELS@/$MODELS}"   # @MODELS@ / @BACKBONE@: this run's paths, as in the arms above
    ARMS="${ARMS//@BACKBONE@/$BACKBONE}"
    [[ ",$ARMS," == *",oracle_beta,"* ]] || oracle_flags=()
  fi
  doc_set_flags=(--n "${EVAL_N:-$N_TEST}")
  (( EVAL_OFFSET > 0 )) && doc_set_flags+=(--offset "$EVAL_OFFSET")
  # our checkpoints must exist before 4 shards start (e.g. STAGES="select ..." after a failed train)
  for ckpt in $(echo "$ARMS" | grep -oE 'pruner:[^|,]+' | sed 's/^pruner://'); do   # also inside rrf:a|b
    if [[ "$ckpt" == "$MODELS"/* && ! -f "$ckpt/pruner_config.json" ]]; then
      echo "!! $ckpt is not a finished pruner checkpoint; run the train stage first (logs: $LOGS/train_*.log)"; exit 1
    fi
  done
  # llmlingua / longllmlingua run in a second pass with the old transformers on PYTHONPATH; it reuses the
  # documents_shard files the first pass wrote (same settings), so it never imports `datasets`
  main_arms=$(echo "$ARMS" | tr ',' '\n' | { grep -vE "$LEGACY_ARM_RE" || true; } | paste -sd, -)
  legacy_arms=$(echo "$ARMS" | tr ',' '\n' | { grep -E "$LEGACY_ARM_RE" || true; } | paste -sd, -)
  echo "== select ($main_arms)"
  run_sharded select 1 env PYTHONPATH="$BASELINE_SITE${PYTHONPATH:+:$PYTHONPATH}" python evaluate.py select \
    --sources "$EVAL_SOURCES" --split test "${doc_set_flags[@]}" \
    --arms "$main_arms" --ratios "$RATIOS" --budget-tokenizer "$PRIMARY_MODEL" --out-dir "$EVAL_DIR" ${oracle_flags[@]+"${oracle_flags[@]}"} $SELECT_ARGS
  if [[ -n "$legacy_arms" ]]; then
    echo "== select with transformers 4.46.3 from $LLMLINGUA_SITE ($legacy_arms)"
    run_sharded select_llmlingua 1 env PYTHONPATH="$LLMLINGUA_SITE${PYTHONPATH:+:$PYTHONPATH}" python evaluate.py select \
      --sources "$EVAL_SOURCES" --split test "${doc_set_flags[@]}" --arms "$legacy_arms" --ratios "$RATIOS" \
      --budget-tokenizer "$PRIMARY_MODEL" --out-dir "$EVAL_DIR" $SELECT_ARGS
  fi
  if [[ -n "$EXTRA_RATIOS_SINGLE" ]]; then
    # same document set (select_config.json), only its single-hop documents, only the extra ratios.
    # oracle_beta has labels for ORACLE_N documents of every source: it joins this pass too.
    single_eval=$(for s in ${EVAL_SOURCES//,/ }; do if [[ " $SINGLE_HOP_SOURCES " == *" $s "* ]]; then echo "$s"; fi; done | paste -sd, -)
    if [[ -n "$single_eval" ]]; then
      echo "== select, single-hop only ($single_eval) at $EXTRA_RATIOS_SINGLE"
      run_sharded select_single 1 env PYTHONPATH="$BASELINE_SITE${PYTHONPATH:+:$PYTHONPATH}" python evaluate.py select \
        --sources "$EVAL_SOURCES" --split test "${doc_set_flags[@]}" --restrict-sources "$single_eval" \
        --arms "$main_arms" --ratios "$EXTRA_RATIOS_SINGLE" --budget-tokenizer "$PRIMARY_MODEL" --out-dir "$EVAL_DIR" \
        ${oracle_flags[@]+"${oracle_flags[@]}"} $SELECT_ARGS
      if [[ -n "$legacy_arms" ]]; then
        run_sharded select_single_llmlingua 1 env PYTHONPATH="$LLMLINGUA_SITE${PYTHONPATH:+:$PYTHONPATH}" python evaluate.py select \
          --sources "$EVAL_SOURCES" --split test "${doc_set_flags[@]}" --restrict-sources "$single_eval" --arms "$legacy_arms" \
          --ratios "$EXTRA_RATIOS_SINGLE" --budget-tokenizer "$PRIMARY_MODEL" --out-dir "$EVAL_DIR" $SELECT_ARGS
      fi
    fi
  fi
fi

if has_stage answer; then
  for reader in $EVAL_READERS; do
    tp=$(tp_for "$reader")
    echo "== answer: $reader (tp=$tp)"
    run_sharded "answer_$(tag "$reader")" "$tp" python evaluate.py answer --out-dir "$EVAL_DIR" \
      --reader-model "$reader" --backend "$BACKEND" --max-model-len "$MAX_MODEL_LEN" --tp "$tp" \
      --gpu-memory-utilization "$GPU_MEM" $ANSWER_ARGS
  done
fi

if has_stage report; then
  echo "== report"
  # the first report of a run is kept: a later report over added arms / ratios / oracle documents rewrites
  # report.json, and the numbers quoted from the first one must stay checkable
  for f in report.json report.md; do
    [[ -f "$EVAL_DIR/$f" && ! -f "$EVAL_DIR/first_$f" ]] && cp "$EVAL_DIR/$f" "$EVAL_DIR/first_$f"
  done
  heldout=""
  for reader in $EVAL_READERS; do [[ " $LABEL_READERS " == *" $reader "* ]] || heldout="$heldout,$(tag "$reader")"; done
  # REPORT_OURS: the arms compared against every other arm in the (exploratory) paired table; the confirmatory
  # families always test ours_beta / ours_ens by name
  python evaluate.py report --out-dir "$EVAL_DIR" --ours "${REPORT_OURS:-ours_beta,ours_ens}" --primary-reader "$PRIMARY" \
    --heldout-readers "${heldout#,}" --equiv-margin "$EQUIV_MARGIN" --oracle-margin "$ORACLE_MARGIN" \
    --confirm-ratios "$CONFIRM_RATIOS" ${PREREG_FILE:+--prereg "$PREREG_FILE"} \
    --min-upgrade-gap "$MIN_UPGRADE_GAP" --labels-dir "$(echo "$LABELS,$LABELS_MAIN" | tr ',' '\n' | awk '!seen[$0]++' | paste -sd, -)" --fit-dir "$FIT" > "$LOGS/report.log" 2>&1 \
    || { tail -n 25 "$LOGS/report.log"; exit 1; }
  python scripts/paper_tables.py --report "$EVAL_DIR/report.json" --primary-reader "$PRIMARY" >> "$LOGS/report.log" 2>&1 \
    || { tail -n 25 "$LOGS/report.log"; exit 1; }
  if python -c "import matplotlib" 2>/dev/null; then   # figures are optional: they can be drawn locally from report.json
    python scripts/paper_figures.py --report "$EVAL_DIR/report.json" --primary-reader "$PRIMARY" >> "$LOGS/report.log" 2>&1 \
      || { tail -n 25 "$LOGS/report.log"; exit 1; }
  else
    echo "   (no matplotlib: run scripts/paper_figures.py --report $EVAL_DIR/report.json locally for the figures)"
  fi
  echo "report: $EVAL_DIR/report.md (paper tables + CSVs + figures: $EVAL_DIR/paper/)"
fi

if has_stage bench; then
  # selection latency for the RQ1 cost table: one arm at a time on ONE GPU, nothing else running (the select
  # stage's per-document times come from four shards side by side). BENCH_ARMS overrides the default list.
  # default: ours (paragraphs, and with the sentence fill), its backbone, embed, bm25 and every published arm of
  # EXTRA_ARMS that runs with the image's transformers (llmlingua / longllmlingua need the old one: not here)
  bench_arms=${BENCH_ARMS:-"ours_beta=pruner:$MODELS/pruner_beta_primary,ours_fill=fill+pruner:$MODELS/pruner_beta_primary,ours_sent=sent+pruner:$MODELS/pruner_beta_primary,reranker=reranker:$BACKBONE,embed=embed:$EMBED_MODEL,bm25"}
  if [[ -z "${BENCH_ARMS:-}" && -n "${EXTRA_ARMS:-}" ]]; then
    published=$(echo "$EXTRA_ARMS" | tr ',' '\n' | { grep -vE "$LEGACY_ARM_RE" || true; } | paste -sd, -)
    bench_arms="$bench_arms${published:+,$published}"
  fi
  [[ -f "$EVAL_DIR/documents_shard0.jsonl" ]] || { echo "!! bench reads $EVAL_DIR/documents_shard*.jsonl: run select first"; exit 1; }
  ensure_baseline_deps   # published arms in BENCH_ARMS need their packages (installed for the arms in EXTRA_ARMS)
  echo "== bench (GPU ${GPU_IDS[0]}): $bench_arms"
  CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}" PYTHONPATH="$BASELINE_SITE${PYTHONPATH:+:$PYTHONPATH}" python scripts/bench_latency.py \
    --eval-dir "$EVAL_DIR" --arms "$bench_arms" --budget-tokenizer "$PRIMARY_MODEL" \
    --n-per-source "${BENCH_N:-100}" --out "${BENCH_OUT:-$EVAL_DIR/bench_latency.json}" 2>&1 | tee "$LOGS/bench.log"
fi

if has_stage upload; then
  echo "== upload to HuggingFace (run '$HF_RUN_NAME', private=$HF_PRIVATE, labels=$HF_UPLOAD_LABELS)"
  upload_flags=(--prefix "$HF_REPO_PREFIX" --run-name "$HF_RUN_NAME" --models-dir "$MODELS" --eval-dir "$EVAL_DIR"
                --labels-dir "$LABELS" --fit-dir "$FIT")
  [[ -n "$HF_NAMESPACE" ]] && upload_flags+=(--namespace "$HF_NAMESPACE")
  [[ "${HF_PRIVATE,,}" =~ ^(false|0|no)$ ]] && upload_flags+=(--public)   # anything else stays private
  [[ "$HF_UPLOAD_LABELS" == true ]] && upload_flags+=(--include-labels)
  python scripts/upload_hf.py "${upload_flags[@]}" 2>&1 | tee -a "$LOGS/upload.log"
fi
