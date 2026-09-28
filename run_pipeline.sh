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
# XProvence / LLMLingua-2 APIs, paths, permissions) in about an hour -- dominated by model start-ups --
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
STAGES=${STAGES:-"preflight prefetch labels fit ensemble train select answer report upload"}
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
# oracle_beta (H1 upper bound = the unamortized attribution): Stage A labels of the first ORACLE_N test
# documents per eval source, primary reader only (test docs are nested across n). 0 disables the arm.
ORACLE_N=${ORACLE_N:-100}
(( ORACLE_N <= N_TEST )) || ORACLE_N=$N_TEST
# report: equivalence margin (H2b), non-inferiority margin to oracle_beta (H1), minimum full-context
# weak->strong F1 gap for an upgrade-retention ratio to be read (H3b)
EQUIV_MARGIN=${EQUIV_MARGIN:-0.02}
ORACLE_MARGIN=${ORACLE_MARGIN:-0.05}
MIN_UPGRADE_GAP=${MIN_UPGRADE_GAP:-0.05}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}        # longest document measured: ~9k Qwen3 tokens
GPU_MEM=${GPU_MEM:-0.90}
DOCS_PER_CALL=${DOCS_PER_CALL:-16}
BACKBONE=${BACKBONE:-BAAI/bge-reranker-v2-m3}
EMBED_MODEL=${EMBED_MODEL:-BAAI/bge-m3}
RUN_ROOT=${RUN_ROOT:-.}
# Distractor ablation (METHOD_SPEC.md §6): DISTRACTORS=hard (single-hop haystacks padded from the needle's own
# article) and/or MULTIHOP_PAD_CHARS=<n> (multi-hop documents lengthened with easy distractors), applied to
# labels AND evaluation. Labels, fits, models and results get their own suffixed directories, so the
# ablation never mixes with the main run (the label dirs would refuse mixed settings anyway).
DISTRACTORS=${DISTRACTORS:-}
MULTIHOP_PAD_CHARS=${MULTIHOP_PAD_CHARS:-}
ABLATION=${DISTRACTORS:+_distractors-$DISTRACTORS}${MULTIHOP_PAD_CHARS:+_pad-$MULTIHOP_PAD_CHARS}
if [[ -n "$ABLATION" ]]; then
  RUN_ROOT="$RUN_ROOT$ABLATION"
  [[ -n "${LABELS:-}" ]] && LABELS="$LABELS$ABLATION"
  HF_RUN_NAME=$(basename "$RUN_ROOT")
  doc_flags="${DISTRACTORS:+ --distractors $DISTRACTORS}${MULTIHOP_PAD_CHARS:+ --multihop-pad-chars $MULTIHOP_PAD_CHARS}"
  MEASURE_ARGS="${MEASURE_ARGS:-}$doc_flags"   # argparse: the last --distractors wins over .env's
  SELECT_ARGS="${SELECT_ARGS:-}$doc_flags"
fi
# LABELS/raw: reader measurements, expensive, SHARED across runs (nested documents: a larger N reuses a
# smaller run's). FIT: ridge fits + ensembles, cheap, PER RUN and restricted to this run's N_TRAIN / N_DEV /
# ORACLE_N documents -- a shared fit dir would train a small-N run on whatever a larger run measured.
LABELS=${LABELS:-$RUN_ROOT/labels}
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
tp_for() { if [[ "$1" =~ $LARGE_READER_PATTERN ]]; then echo "$TP_LARGE"; else echo 1; fi; }
mkdir -p "$LOGS"
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

# Published-compressor arms need packages the vLLM image does not ship: llmlingua2 -> `llmlingua`;
# provence:/XProvence -> `spacy` + its multilingual sentence model `xx_sent_ud_sm` (loaded when the
# remote modeling code is imported). Installed on demand with the image's torch / transformers / vllm /
# numpy / tokenizers pinned as constraints, so pip fails instead of swapping the CUDA build vLLM needs.
pip_pinned() {  # pip_pinned <package...>
  local pins; pins=$(mktemp)
  python -m pip freeze 2>/dev/null | grep -iE '^(torch|transformers|vllm|numpy|tokenizers)==' > "$pins" || true
  if ! python -m pip install --quiet "$@" -c "$pins"; then
    echo "!! pip install $* failed (conflicts with pinned: $(paste -sd' ' "$pins"));"
    echo "   install by hand or drop the arm from EXTRA_ARMS"; rm -f "$pins"; exit 1
  fi
  rm -f "$pins"
}
ensure_baseline_deps() {
  if [[ ",${EXTRA_ARMS:-}," == *llmlingua2* ]] && ! python -c "import llmlingua" 2>/dev/null; then
    echo "== installing llmlingua (EXTRA_ARMS has llmlingua2)"
    pip_pinned llmlingua
    python -c "import llmlingua" || { echo "!! llmlingua installed but does not import"; exit 1; }
  fi
  if [[ ",${EXTRA_ARMS:-}" == *provence:* ]] && ! python -c "import spacy; spacy.load('xx_sent_ud_sm')" 2>/dev/null; then
    echo "== installing spacy + xx_sent_ud_sm (EXTRA_ARMS has a provence: arm)"
    pip_pinned spacy
    python -m spacy download xx_sent_ud_sm || { echo "!! spacy model xx_sent_ud_sm download failed"; exit 1; }
    python -c "import spacy; spacy.load('xx_sent_ud_sm')" || { echo "!! xx_sent_ud_sm does not load"; exit 1; }
  fi
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
  python -m pytest -q -x tests/test_attribution.py tests/test_selection_metrics.py tests/test_evaluate_report.py
  ensure_baseline_deps  # an optional baseline must not fail hours later at select
  if has_stage upload; then  # a missing or read-only token should fail now, not after training
    python scripts/upload_hf.py --check ${HF_NAMESPACE:+--namespace "$HF_NAMESPACE"}
  fi
fi

if has_stage prefetch; then
  echo "== prefetch (single process; avoids N processes racing on the HF cache)"
  # + hub ids of provence:/embed:/reranker: arms in EXTRA_ARMS (llmlingua2 downloads its own model)
  extra_models=$(echo "${EXTRA_ARMS:-}" | tr ',' '\n' | sed -nE 's/^([^=]*=)?(provence|embed|reranker):(.+)$/\3/p')
  all_models=$(echo "$LABEL_READERS $EVAL_READERS $BACKBONE $EMBED_MODEL $extra_models" | tr ' ' '\n' | awk 'NF && !seen[$0]++' | paste -sd, -)
  all_sources=$(echo "$TRAIN_SOURCES ${EVAL_SOURCES//,/ }" | tr ' ' '\n' | awk 'NF && !seen[$0]++' | paste -sd, -)
  python scripts/prefetch.py --sources "$all_sources" --models "$all_models" 2>&1 | tee -a "$LOGS/prefetch.log"
fi

if has_stage labels; then
  for reader in $LABEL_READERS; do
    tp=$(tp_for "$reader")
    for src in $TRAIN_SOURCES; do
      for split in train dev; do
        n=$N_TRAIN; [[ $split == dev ]] && n=$N_DEV
        echo "== labels: $reader $src/$split (n=$n, tp=$tp)"
        run_sharded "labels_$(tag "$reader")_${src}_$split" "$tp" \
          python generate_labels.py measure --source "$src" --split "$split" --n "$n" \
          --reader-model "$reader" --backend "$BACKEND" --max-model-len "$MAX_MODEL_LEN" --tp "$tp" \
          --gpu-memory-utilization "$GPU_MEM" --docs-per-call "$DOCS_PER_CALL" --out-root "$LABELS/raw" $MEASURE_ARGS
      done
    done
  done
  if (( ORACLE_N > 0 )); then  # never read by training: label_dirs() only lists train/dev
    tp=$(tp_for "$PRIMARY_MODEL")
    for src in ${EVAL_SOURCES//,/ }; do
      echo "== labels (oracle_beta): $PRIMARY_MODEL $src/test (n=$ORACLE_N, tp=$tp)"
      run_sharded "labels_${PRIMARY}_${src}_test" "$tp" \
        python generate_labels.py measure --source "$src" --split test --n "$ORACLE_N" \
        --reader-model "$PRIMARY_MODEL" --backend "$BACKEND" --max-model-len "$MAX_MODEL_LEN" --tp "$tp" \
        --gpu-memory-utilization "$GPU_MEM" --docs-per-call "$DOCS_PER_CALL" --out-root "$LABELS/raw" $MEASURE_ARGS
    done
  fi
fi

if has_stage fit; then
  for reader in $LABEL_READERS; do
    r=$(tag "$reader")
    for target in f1 logprob; do
      for src in $TRAIN_SOURCES; do
        for split in train dev; do
          echo "== fit: $r $target $src/$split"
          python generate_labels.py fit --raw-dir "$LABELS/raw/$r/${src}_$split" --target "$target" \
            --alpha-from "$LABELS/raw/$r/${src}_dev" --out-dir "$FIT/$r/$target/${src}_$split" \
            --n "$([[ $split == dev ]] && echo "$N_DEV" || echo "$N_TRAIN")" --alpha-n "$N_DEV" \
            >> "$LOGS/fit.log" 2>&1 || { echo "!! fit failed:"; tail -n 25 "$LOGS/fit.log"; exit 1; }
        done
      done
    done
  done
  if (( ORACLE_N > 0 )); then
    pooled_dev=""
    for s in $TRAIN_SOURCES; do pooled_dev="$pooled_dev,$LABELS/raw/$PRIMARY/${s}_dev"; done
    for src in ${EVAL_SOURCES//,/ }; do
      # alpha from the source's own dev labels; eval-only sources (xquad_vi, 2wiki) pool the train-source dev labels
      alpha_from=${pooled_dev#,}
      [[ " $TRAIN_SOURCES " == *" $src "* ]] && alpha_from="$LABELS/raw/$PRIMARY/${src}_dev"
      echo "== fit (oracle_beta): $PRIMARY f1 $src/test"
      python generate_labels.py fit --raw-dir "$LABELS/raw/$PRIMARY/${src}_test" --target f1 \
        --alpha-from "$alpha_from" --out-dir "$FIT/$PRIMARY/f1/${src}_test" --n "$ORACLE_N" --alpha-n "$N_DEV" \
        >> "$LOGS/fit.log" 2>&1 || { echo "!! fit failed:"; tail -n 25 "$LOGS/fit.log"; exit 1; }
    done
  fi
fi

if has_stage ensemble; then
  for src in $TRAIN_SOURCES; do
    for split in train dev; do
      dirs=""
      for reader in $LABEL_READERS; do dirs="$dirs,$FIT/$(tag "$reader")/f1/${src}_$split"; done
      echo "== ensemble: $src/$split"
      python generate_labels.py ensemble --fit-dirs "${dirs#,}" --out-dir "$FIT/ensemble/f1/${src}_$split" \
        >> "$LOGS/ensemble.log" 2>&1 || { echo "!! ensemble failed:"; tail -n 25 "$LOGS/ensemble.log"; exit 1; }
    done
  done
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
  pids=(); names=(); g=0
  flush_train() {
    local failed=0
    for i in "${!pids[@]}"; do
      if ! wait "${pids[$i]}"; then
        failed=1; echo "!! train ${names[$i]} FAILED:"; tail -n 25 "$LOGS/train_${names[$i]}.log"
      fi
    done
    pids=(); names=()
    (( failed == 0 )) || exit 1
  }
  for run in "${RUNS[@]}"; do
    IFS='|' read -r name source reader target extra seed <<< "$run"
    if [[ -f "$MODELS/$name/train_log.json" ]]; then echo "== train $name: done, skipping"; continue; fi
    echo "== train $name (GPU $g) -> $LOGS/train_$name.log"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=${GPU_IDS[$g]} python train_pruner.py --label-source "$source" \
      --train-labels "$(label_dirs "$reader" "$target" train)" --dev-labels "$(label_dirs "$reader" "$target" dev)" \
      --backbone "$BACKBONE" --grad-checkpointing $extra $TRAIN_ARGS ${seed:+--seed "$seed"} --out-dir "$MODELS/$name" \
      > "$LOGS/train_$name.log" 2>&1 &
    pids+=($!); names+=("$name")
    g=$(( (g + 1) % NUM_GPUS ))
    if (( g == 0 )); then flush_train; fi
  done
  flush_train
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
  ARMS="$ARMS${EXTRA_ARMS:+,$EXTRA_ARMS}"   # e.g. EXTRA_ARMS="xprovence=provence:naver/xprovence-reranker-bgem3-v1,llmlingua2"
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
  # our checkpoints must exist before 4 shards start (e.g. STAGES="select ..." after a failed train)
  for ckpt in $(echo "$ARMS" | tr ',' '\n' | sed -nE 's/^([^=]*=)?pruner:(.+)$/\2/p'); do
    if [[ "$ckpt" == "$MODELS"/* && ! -f "$ckpt/pruner_config.json" ]]; then
      echo "!! $ckpt is not a finished pruner checkpoint; run the train stage first (logs: $LOGS/train_*.log)"; exit 1
    fi
  done
  echo "== select ($ARMS)"
  run_sharded select 1 python evaluate.py select --sources "$EVAL_SOURCES" --split test --n "$N_TEST" \
    --arms "$ARMS" --ratios "$RATIOS" --budget-tokenizer "$PRIMARY_MODEL" --out-dir "$EVAL_DIR" ${oracle_flags[@]+"${oracle_flags[@]}"} $SELECT_ARGS
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
  heldout=""
  for reader in $EVAL_READERS; do [[ " $LABEL_READERS " == *" $reader "* ]] || heldout="$heldout,$(tag "$reader")"; done
  python evaluate.py report --out-dir "$EVAL_DIR" --ours ours_beta,ours_ens --primary-reader "$PRIMARY" \
    --heldout-readers "${heldout#,}" --equiv-margin "$EQUIV_MARGIN" --oracle-margin "$ORACLE_MARGIN" \
    --min-upgrade-gap "$MIN_UPGRADE_GAP" --labels-dir "$LABELS" --fit-dir "$FIT" > "$LOGS/report.log" 2>&1 \
    || { tail -n 25 "$LOGS/report.log"; exit 1; }
  python scripts/paper_tables.py --report "$EVAL_DIR/report.json" --primary-reader "$PRIMARY" >> "$LOGS/report.log" 2>&1 \
    || { tail -n 25 "$LOGS/report.log"; exit 1; }
  echo "report: $EVAL_DIR/report.md (paper tables + CSVs: $EVAL_DIR/paper/)"
fi

if has_stage upload; then
  echo "== upload to HuggingFace (run '$HF_RUN_NAME', private=$HF_PRIVATE, labels=$HF_UPLOAD_LABELS)"
  upload_flags=(--prefix "$HF_REPO_PREFIX" --run-name "$HF_RUN_NAME" --models-dir "$MODELS" --eval-dir "$EVAL_DIR"
                --labels-dir "$LABELS" --fit-dir "$FIT")
  [[ -n "$HF_NAMESPACE" ]] && upload_flags+=(--namespace "$HF_NAMESPACE")
  [[ "$HF_PRIVATE" == true ]] || upload_flags+=(--public)
  [[ "$HF_UPLOAD_LABELS" == true ]] && upload_flags+=(--include-labels)
  python scripts/upload_hf.py "${upload_flags[@]}" 2>&1 | tee -a "$LOGS/upload.log"
fi
