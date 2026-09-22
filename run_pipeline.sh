#!/usr/bin/env bash
# run_pipeline.sh -- the whole PCS flow in one command: install deps, fetch
# data, sanity-test, sweep lambda on dev ("training", PCS_METHOD_SPEC.md
# §6/§11), run the official bench on test, optionally push results to HF.
#
#   ./run_pipeline.sh --reader-model Qwen/Qwen3-8B --encoder-path models/encoder_compressor
#   ./run_pipeline.sh --relevance synthetic --skip-install    # smoke-test, no checkpoint/deps reinstall
#   ./run_pipeline.sh --lam 0.4 --skip-train                  # skip the sweep, evaluate at a known lambda
#   ./run_pipeline.sh --push-to-hub --hf-repo-id you/ttcompress-pcs-results
#
# Every step is individually skippable (--skip-install/--skip-fetch/--skip-test/
# --skip-train/--skip-eval) so a partial re-run doesn't redo expensive steps.
# --push-to-hub is opt-in (off by default) and pushes only run artifacts
# (results/*.json: the lambda curve, the exact dev/test split, the official
# CI) to a private HF Dataset repo -- PCS never trains a model
# (PCS_METHOD_SPEC.md constraint #1: bolt-on only), so there is no model
# repo to push to. See scripts/push_results.py.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# Load .env if present (see .env.example) -- `set -a` auto-exports every
# variable it defines, so plain `${VAR:-default}` below and every child
# process (python, huggingface_hub) see it, not just this script.
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

# ---------------------------------------------------------------------------
# Defaults (.env / env-var overridable; CLI flags below take precedence over both)
# ---------------------------------------------------------------------------
READER_MODEL="${READER_MODEL:-Qwen/Qwen3-8B}"
ENCODER_PATH="${ENCODER_PATH:-}"
RELEVANCE="${RELEVANCE:-e6}"          # e6 | synthetic (see ttcompress/relevance.py)
DEVICE="${DEVICE:-cuda}"
RATIOS="${RATIOS:-4,8}"
ARMS="${ARMS:-encoder_pcs,h2o,snapkv}"
HF_REPO_ID="${HF_REPO_ID:-}"           # dataset repo id, e.g. you/ttcompress-pcs-results
HF_PRIVATE="${HF_PRIVATE:-1}"          # 1 = private (default), 0 = public
VENV=""
LAM=""                                  # if set, skip the sweep and use this lambda directly
FORCE_FETCH=0
PUSH_TO_HUB="${PUSH_TO_HUB:-0}"
SKIP_INSTALL=0
SKIP_FETCH=0
SKIP_TEST=0
SKIP_TRAIN=0
SKIP_EVAL=0

usage() {
    sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'
    cat <<'EOF'
Flags:
  --reader-model NAME     HF causal LM id/path (default: Qwen/Qwen3-8B)
  --encoder-path PATH     trained E6 token-classifier checkpoint (required unless --relevance synthetic)
  --relevance e6|synthetic
  --device cuda|cpu
  --ratios LIST           e.g. 4,8 (passed to train.py's sweep)
  --arms LIST             e.g. encoder_pcs,h2o,snapkv (passed to evaluate.py)
  --lam FLOAT             skip the sweep, evaluate directly at this lambda
  --venv PATH             create/use a venv here before installing requirements.txt
  --force-fetch           re-download data/ even if already present
  --push-to-hub           push results/*.json to a HF Dataset repo after step 5 (needs --hf-repo-id, HF_TOKEN)
  --hf-repo-id ID         e.g. you/ttcompress-pcs-results
  --hf-public             push as a public repo instead of the private default
  --skip-install / --skip-fetch / --skip-test / --skip-train / --skip-eval
  -h, --help
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --reader-model) READER_MODEL="$2"; shift 2 ;;
        --encoder-path) ENCODER_PATH="$2"; shift 2 ;;
        --relevance) RELEVANCE="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --ratios) RATIOS="$2"; shift 2 ;;
        --arms) ARMS="$2"; shift 2 ;;
        --lam) LAM="$2"; shift 2 ;;
        --venv) VENV="$2"; shift 2 ;;
        --force-fetch) FORCE_FETCH=1; shift ;;
        --push-to-hub) PUSH_TO_HUB=1; shift ;;
        --hf-repo-id) HF_REPO_ID="$2"; shift 2 ;;
        --hf-public) HF_PRIVATE=0; shift ;;
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

PY=python
[ -n "$VENV" ] && PY="$VENV/bin/python"

step() { echo; echo "=== $* ==="; }

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
# 2. Fetch data (scripts/fetch_data.sh -- see data/SOURCES.md; not committed)
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
# 3. Sanity test -- the mandatory lambda=1 <-> TruncationCompressor
#    equivalence test (PCS_METHOD_SPEC.md §2/§11) must pass before any other
#    number from this run is trusted. No GPU needed, seconds to run.
# ---------------------------------------------------------------------------
if [ "$SKIP_TEST" -eq 0 ]; then
    step "3/6  Sanity tests"
    "$PY" -m pytest tests/ -q
else
    step "3/6  Sanity tests (skipped)"
fi

# ---------------------------------------------------------------------------
# 4. Train = the lambda sweep on the dev split (no gradients -- see
#    train.py's docstring on why this is GPU-bound, not the "CPU only" §9
#    describes). Writes results/dev_test_split.json + results/lambda_sweep_dev.json.
# ---------------------------------------------------------------------------
if [ -n "$LAM" ]; then
    step "4/6  Train (skipped -- using --lam $LAM)"
elif [ "$SKIP_TRAIN" -eq 0 ]; then
    step "4/6  Train (lambda sweep on dev)"
    train_args=(--reader-model "$READER_MODEL" --relevance "$RELEVANCE" --ratios "$RATIOS" --device "$DEVICE")
    [ -n "$ENCODER_PATH" ] && train_args+=(--encoder-path "$ENCODER_PATH")
    "$PY" train.py "${train_args[@]}"
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
# 5. Evaluate = the official bench on test (pooled + per-source CI). Reads
#    its test set from results/dev_test_split.json (written by step 4) by
#    default -- see evaluate.py's docstring on why that avoids dev/test leakage.
# ---------------------------------------------------------------------------
if [ "$SKIP_EVAL" -eq 0 ]; then
    step "5/6  Evaluate"
    if [ -z "$LAM" ]; then
        echo "[FATAL] No lambda available -- pass --lam FLOAT, or drop --skip-train so step 4 can pick one." >&2
        exit 1
    fi
    eval_args=(--reader-model "$READER_MODEL" --relevance "$RELEVANCE" --arms "$ARMS" --device "$DEVICE" --lam "$LAM")
    [ -n "$ENCODER_PATH" ] && eval_args+=(--encoder-path "$ENCODER_PATH")
    "$PY" evaluate.py "${eval_args[@]}"
else
    step "5/6  Evaluate (skipped)"
fi

# ---------------------------------------------------------------------------
# 6. Push results/*.json to a HF Dataset repo -- opt-in (--push-to-hub /
#    PUSH_TO_HUB=1), off by default. See scripts/push_results.py's docstring
#    on why this is a Dataset repo, not a Model repo.
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
echo "Results under ./results/"
