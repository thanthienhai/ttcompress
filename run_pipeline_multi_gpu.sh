#!/usr/bin/env bash
# run_pipeline_multi_gpu.sh -- same flow as run_pipeline.sh (install, fetch,
# sanity-test, sweep lambda on dev, official bench on test, optional push),
# but steps 4/5 (the only GPU-bound, expensive ones) are SHARDED across
# NUM_GPUS GPUs instead of running on one.
#
# Neither train.py nor evaluate.py parallelizes internally -- each is a
# single process pinned to one --device -- so plain run_pipeline.sh only
# ever uses one GPU no matter how many are in the machine. This script
# splits train.py's 11-point lambda grid, and evaluate.py's arm/ratio cells
# (encoder_pcs@4x, encoder_pcs@8x, h2o@8x, snapkv@8x by default -- 4 cells,
# a natural fit for 4 GPUs), across NUM_GPUS background processes pinned to
# cuda:0..cuda:N-1 via scripts/shard_pipeline.py, then merges every shard's
# output JSON back into the exact file shape (results/lambda_sweep_dev.json,
# results/official_bench.json) a plain single-GPU run_pipeline.sh run would
# have produced -- step 5's push and any other downstream consumer don't
# need to know sharding happened.
#
#   ./run_pipeline_multi_gpu.sh --reader-model Qwen/Qwen3-8B --encoder-path models/encoder_compressor
#   ./run_pipeline_multi_gpu.sh --num-gpus 4 --relevance synthetic --skip-install   # smoke-test the sharding itself
#   NUM_GPUS=4 ./run_pipeline_multi_gpu.sh   # via .env instead of --num-gpus
#
# Every step is individually skippable, same convention as run_pipeline.sh.
# A GPU shard that gets zero work (num-gpus > 11 lambdas, or > 4 arm/ratio
# cells) is simply not launched, not an error.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

READER_MODEL="${READER_MODEL:-Qwen/Qwen3-8B}"
ENCODER_PATH="${ENCODER_PATH:-}"
RELEVANCE="${RELEVANCE:-e6}"
RATIOS="${RATIOS:-4,8}"
ARMS="${ARMS:-encoder_pcs,h2o,snapkv}"
NUM_GPUS="${NUM_GPUS:-1}"
HF_REPO_ID="${HF_REPO_ID:-}"
HF_PRIVATE="${HF_PRIVATE:-1}"
VENV=""
LAM=""
FORCE_FETCH=0
PUSH_TO_HUB="${PUSH_TO_HUB:-0}"
SKIP_INSTALL=0
SKIP_FETCH=0
SKIP_TEST=0
SKIP_TRAIN=0
SKIP_EVAL=0
NUM_DEV_ESSAYS=10
SAMPLES_PER_ESSAY=2
LONGBENCH_DEV=20
SEED=42
RUN_DIR="results/shards"

usage() {
    sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'
    cat <<'EOF'
Flags:
  --reader-model NAME     HF causal LM id/path (default: Qwen/Qwen3-8B)
  --encoder-path PATH     trained E6 token-classifier checkpoint (required unless --relevance synthetic)
  --relevance e6|synthetic
  --num-gpus N            shard count, cuda:0..cuda:N-1 (default: NUM_GPUS from .env, else 1)
  --ratios LIST           e.g. 4,8 (passed to train.py's sweep)
  --arms LIST             e.g. encoder_pcs,h2o,snapkv (passed to evaluate.py)
  --lam FLOAT             skip the sweep, evaluate directly at this lambda
  --venv PATH             create/use a venv here before installing requirements.txt
  --force-fetch           re-download data/ even if already present
  --push-to-hub           push results/*.json to a HF Dataset repo after step 5 (needs --hf-repo-id, HF_TOKEN)
  --hf-repo-id ID         e.g. you/ttcompress-pcs-results
  --hf-public             push as a public repo instead of the private default
  --run-dir DIR           where per-shard intermediate files go (default: results/shards)
  --skip-install / --skip-fetch / --skip-test / --skip-train / --skip-eval
  -h, --help
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --reader-model) READER_MODEL="$2"; shift 2 ;;
        --encoder-path) ENCODER_PATH="$2"; shift 2 ;;
        --relevance) RELEVANCE="$2"; shift 2 ;;
        --num-gpus) NUM_GPUS="$2"; shift 2 ;;
        --ratios) RATIOS="$2"; shift 2 ;;
        --arms) ARMS="$2"; shift 2 ;;
        --lam) LAM="$2"; shift 2 ;;
        --venv) VENV="$2"; shift 2 ;;
        --force-fetch) FORCE_FETCH=1; shift ;;
        --push-to-hub) PUSH_TO_HUB=1; shift ;;
        --hf-repo-id) HF_REPO_ID="$2"; shift 2 ;;
        --hf-public) HF_PRIVATE=0; shift ;;
        --run-dir) RUN_DIR="$2"; shift 2 ;;
        --skip-install) SKIP_INSTALL=1; shift ;;
        --skip-fetch) SKIP_FETCH=1; shift ;;
        --skip-test) SKIP_TEST=1; shift ;;
        --skip-train) SKIP_TRAIN=1; shift ;;
        --skip-eval) SKIP_EVAL=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown flag: $1" >&2; usage; exit 1 ;;
    esac
done

if [ "$RELEVANCE" = "e6" ] && [ -z "$ENCODER_PATH" ] && { [ "$SKIP_TRAIN" -eq 0 ] || [ "$SKIP_EVAL" -eq 0 ]; }; then
    echo "[FATAL] --relevance e6 needs --encoder-path (a trained E6 checkpoint)." >&2
    echo "        Pass --relevance synthetic instead to smoke-test the pipeline without one." >&2
    exit 1
fi
if [ "$PUSH_TO_HUB" -eq 1 ] && [ -z "$HF_REPO_ID" ]; then
    echo "[FATAL] --push-to-hub needs --hf-repo-id (or HF_REPO_ID in .env), e.g. you/ttcompress-pcs-results" >&2
    exit 1
fi
if ! [[ "$NUM_GPUS" =~ ^[0-9]+$ ]] || [ "$NUM_GPUS" -lt 1 ]; then
    echo "[FATAL] --num-gpus must be a positive integer (got '$NUM_GPUS')." >&2
    exit 1
fi

PY=python
[ -n "$VENV" ] && PY="$VENV/bin/python"

step() { echo; echo "=== $* ==="; }

mkdir -p "$RUN_DIR"
SPLIT_OUT="results/dev_test_split.json"

# ---------------------------------------------------------------------------
# 1. Install
# ---------------------------------------------------------------------------
if [ "$SKIP_INSTALL" -eq 0 ]; then
    step "1/6  Install"
    if [ -n "$VENV" ]; then
        [ -d "$VENV" ] || python -m venv "$VENV"
        "$VENV/bin/pip" install -q -U pip
        "$VENV/bin/pip" install -q -r requirements.txt
    else
        "$PY" -m pip install -q -U pip
        "$PY" -m pip install -q -r requirements.txt
    fi
else
    step "1/6  Install (skipped)"
fi

# ---------------------------------------------------------------------------
# 2. Fetch data
# ---------------------------------------------------------------------------
if [ "$SKIP_FETCH" -eq 0 ]; then
    step "2/6  Fetch data"
    if [ "$FORCE_FETCH" -eq 1 ]; then
        ./scripts/fetch_data.sh --force
    else
        ./scripts/fetch_data.sh
    fi
else
    step "2/6  Fetch data (skipped)"
fi

# ---------------------------------------------------------------------------
# 3. Sanity test
# ---------------------------------------------------------------------------
if [ "$SKIP_TEST" -eq 0 ]; then
    step "3/6  Sanity tests"
    "$PY" -m pytest tests/ -q
else
    step "3/6  Sanity tests (skipped)"
fi

# ---------------------------------------------------------------------------
# 4. Train = the lambda sweep on the dev split, sharded across NUM_GPUS GPUs.
#    Every shard resolves the IDENTICAL dev/test split (same seed/args), so
#    it's written ONCE here (CPU only, no model load) instead of N processes
#    racing to write the same file; every train.py shard then runs with
#    --skip-split-write and just reads it back for its own dev pool.
# ---------------------------------------------------------------------------
if [ -n "$LAM" ]; then
    step "4/6  Train (skipped -- using --lam $LAM)"
elif [ "$SKIP_TRAIN" -eq 0 ]; then
    step "4/6  Train (lambda sweep on dev, sharded across $NUM_GPUS GPU(s))"
    "$PY" -c "
from ttcompress.public_datasets import build_dev_test_split, save_dev_test_split
dev, test = build_dev_test_split(num_dev_essays=$NUM_DEV_ESSAYS, samples_per_essay=$SAMPLES_PER_ESSAY, longbench_num_dev=$LONGBENCH_DEV, seed=$SEED)
save_dev_test_split(dev, test, '$SPLIT_OUT')
print(f'Dev/test split saved to $SPLIT_OUT ({len(dev)} dev, {len(test)} test)')
"
    train_shard_files=()
    pids=()
    for i in $(seq 0 $((NUM_GPUS - 1))); do
        shard_lambdas="$("$PY" scripts/shard_pipeline.py lambdas-for-shard --shard-index "$i" --num-shards "$NUM_GPUS")"
        [ -z "$shard_lambdas" ] && continue
        shard_out="$RUN_DIR/train_shard${i}.json"
        train_shard_files+=("$shard_out")
        train_args=(--reader-model "$READER_MODEL" --relevance "$RELEVANCE" --ratios "$RATIOS" --device "cuda:$i"
            --lambdas "$shard_lambdas" --skip-split-write --split-out "$SPLIT_OUT" --out "$shard_out"
            --num-dev-essays "$NUM_DEV_ESSAYS" --samples-per-essay "$SAMPLES_PER_ESSAY"
            --longbench-dev "$LONGBENCH_DEV" --seed "$SEED")
        [ -n "$ENCODER_PATH" ] && train_args+=(--encoder-path "$ENCODER_PATH")
        echo "  [cuda:$i] lambdas: $shard_lambdas -> $shard_out"
        "$PY" train.py "${train_args[@]}" > "$RUN_DIR/train_shard${i}.log" 2>&1 &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "$pid"; done
    joined="$(IFS=,; echo "${train_shard_files[*]}")"
    "$PY" scripts/shard_pipeline.py merge-train --shard-files "$joined" --out results/lambda_sweep_dev.json
    LAM="$("$PY" -c "import json; print(json.load(open('results/lambda_sweep_dev.json'))['chosen_lambda'])")"
    echo "Chosen lambda* = $LAM"
else
    step "4/6  Train (skipped)"
    if [ -f results/lambda_sweep_dev.json ]; then
        LAM="$("$PY" -c "import json; print(json.load(open('results/lambda_sweep_dev.json'))['chosen_lambda'])")"
        echo "Reusing lambda* = $LAM from an earlier results/lambda_sweep_dev.json"
    fi
fi

# ---------------------------------------------------------------------------
# 5. Evaluate = the official bench on test, sharded by (arm, ratio) cell
#    across NUM_GPUS GPUs -- 4 cells by default (encoder_pcs@4x/@8x, h2o@8x,
#    snapkv@8x), a natural fit for 4 GPUs. A GPU that draws more than one
#    cell (num-gpus < cell count) runs its cells sequentially, still on its
#    own device; a GPU that draws zero (num-gpus > cell count) is skipped.
# ---------------------------------------------------------------------------
if [ "$SKIP_EVAL" -eq 0 ]; then
    step "5/6  Evaluate (sharded across $NUM_GPUS GPU(s))"
    if [ -z "$LAM" ]; then
        echo "[FATAL] No lambda available -- pass --lam FLOAT, or drop --skip-train so step 4 can pick one." >&2
        exit 1
    fi
    eval_shard_files=()
    pids=()
    for i in $(seq 0 $((NUM_GPUS - 1))); do
        cells="$("$PY" scripts/shard_pipeline.py cells-for-shard --shard-index "$i" --num-shards "$NUM_GPUS" --arms "$ARMS")"
        [ -z "$cells" ] && continue
        j=0
        while IFS=' ' read -r arm ratio; do
            [ -z "$arm" ] && continue
            eval_shard_files+=("$RUN_DIR/eval_shard${i}_cell${j}.json")
            j=$((j + 1))
        done <<< "$cells"
        (
            j=0
            while IFS=' ' read -r arm ratio; do
                [ -z "$arm" ] && continue
                out="$RUN_DIR/eval_shard${i}_cell${j}.json"
                eval_args=(--reader-model "$READER_MODEL" --relevance "$RELEVANCE" --device "cuda:$i" --lam "$LAM"
                    --arms "$arm" --ratios "$ratio" --split-file "$SPLIT_OUT" --out "$out")
                [ -n "$ENCODER_PATH" ] && eval_args+=(--encoder-path "$ENCODER_PATH")
                echo "  [cuda:$i] $arm @ ${ratio}x -> $out"
                "$PY" evaluate.py "${eval_args[@]}"
                j=$((j + 1))
            done <<< "$cells"
        ) > "$RUN_DIR/eval_shard${i}.log" 2>&1 &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "$pid"; done
    joined="$(IFS=,; echo "${eval_shard_files[*]}")"
    "$PY" scripts/shard_pipeline.py merge-eval --shard-files "$joined" --out results/official_bench.json
else
    step "5/6  Evaluate (skipped)"
fi

# ---------------------------------------------------------------------------
# 6. Push results/*.json to a HF Dataset repo -- unchanged from run_pipeline.sh
# ---------------------------------------------------------------------------
if [ "$PUSH_TO_HUB" -eq 1 ]; then
    step "6/6  Push results to HF"
    push_args=(--repo-id "$HF_REPO_ID")
    [ "$HF_PRIVATE" -eq 0 ] && push_args+=(--public)
    "$PY" scripts/push_results.py "${push_args[@]}"
else
    step "6/6  Push results to HF (skipped -- pass --push-to-hub to enable)"
fi

step "Done"
echo "Results under ./results/ (per-shard intermediates + logs under $RUN_DIR/)"
