#!/usr/bin/env bash
# run_outcome_pipeline.sh -- OUTCOME_SUPERVISED_RELEVANCE_SPEC.md end to end:
# overlap check -> Stage A (measure+fit, train & dev) -> Stage B (decompose)
# -> Stage C (train relevance_v2 + relevance_v3) -> Stage 6 (sweep + the
# one-time public-Vietnamese-test-pool + English-test-pool evaluation), BOTH
# label variants. Every data source is a public, citable dataset -- the
# original vcc_bench_v2.json (an unpublished internal vncompress benchmark)
# was dropped entirely; see ttcompress/vietnamese_public_test.py for what
# replaced it (UIT-ViQuAD 2.0 + XQuAD-vi reserved portions).
#
# Multilingual upgrade: --sources picks any mix of uit_viquad, xquad_vi,
# vimqa (Vietnamese) and longbench/ruler/kamradt/infinitebench (English) for
# Stage A/Stage 6's sweep; Stage 6's evaluate always reports the
# public Vietnamese test pool and, unless the checkpoint is Vietnamese-only,
# the English test pool too. Default --base-checkpoint is multilingual
# (xlm-roberta-base) precisely because --sources defaults to 'all' now, not
# the Vietnamese-only vinai/phobert-base this pipeline started with.
#
# Defaults to spec §7's explicit pilot scale (N=10, K=15) -- read the real
# per-mask timing this prints before scaling up with --n/--k.
#
#   ./run_outcome_pipeline.sh --reader-model Qwen/Qwen3-8B
#   ./run_outcome_pipeline.sh --reader-model Qwen/Qwen3-8B --n 200 --k 20   # past the pilot
#   ./run_outcome_pipeline.sh --reader-model Qwen/Qwen3-8B --sources uit_viquad \
#       --base-checkpoint vinai/phobert-base   # original Vietnamese-only scope
#
# Every step is individually skippable, same convention as run_pipeline.sh.
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
DEVICE="${DEVICE:-cuda}"
SOURCES="${SOURCES:-all}"     # uit_viquad,longbench,ruler,kamradt or any comma subset
N="${N:-10}"     # spec §7 pilot default -- documents (across --sources) per split
K="${K:-15}"     # spec §7 pilot default -- masks per document
BASE_CHECKPOINT="${BASE_CHECKPOINT:-xlm-roberta-base}"   # multilingual; use vinai/phobert-base for --sources uit_viquad
EPOCHS="${EPOCHS:-3}"
RUN_DIR="${RUN_DIR:-results/outcome_run}"
SKIP_OVERLAP=0
SKIP_MEASURE=0
SKIP_FIT=0
SKIP_DECOMPOSE=0
SKIP_TRAIN=0
SKIP_SWEEP=0
SKIP_EVAL=0

usage() {
    sed -n '2,27p' "$0" | sed 's/^# \{0,1\}//'
    cat <<'EOF'
Flags:
  --reader-model NAME     HF causal LM id/path (default: Qwen/Qwen3-8B)
  --device cuda|cpu
  --sources LIST          comma list from uit_viquad,xquad_vi,vimqa,longbench,ruler,kamradt,infinitebench, or 'all' (default: all)
  --n N                   documents per split, split evenly across --sources (default: 10, spec §7 pilot)
  --k K                   masks per document (default: 15, spec §7 pilot)
  --base-checkpoint NAME  E6 base encoder for Stage C (default: xlm-roberta-base)
  --epochs N              Stage C training epochs (default: 3)
  --run-dir DIR           where all intermediate artifacts go (default: results/outcome_run)
  --skip-overlap / --skip-measure / --skip-fit / --skip-decompose / --skip-train / --skip-sweep / --skip-eval
  -h, --help
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --reader-model) READER_MODEL="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --sources) SOURCES="$2"; shift 2 ;;
        --n) N="$2"; shift 2 ;;
        --k) K="$2"; shift 2 ;;
        --base-checkpoint) BASE_CHECKPOINT="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --run-dir) RUN_DIR="$2"; shift 2 ;;
        --skip-overlap) SKIP_OVERLAP=1; shift ;;
        --skip-measure) SKIP_MEASURE=1; shift ;;
        --skip-fit) SKIP_FIT=1; shift ;;
        --skip-decompose) SKIP_DECOMPOSE=1; shift ;;
        --skip-train) SKIP_TRAIN=1; shift ;;
        --skip-sweep) SKIP_SWEEP=1; shift ;;
        --skip-eval) SKIP_EVAL=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown flag: $1" >&2; usage; exit 1 ;;
    esac
done

step() { echo; echo "=== $* ==="; }

OVERLAP_REPORT="$RUN_DIR/train_test_overlap.json"
if [ "$SKIP_OVERLAP" -eq 0 ]; then
    step "1/7  Train/test overlap check (§1, mandatory before Stage A) -- train vs. both reserved test pools"
    python scripts/check_uit_viquad_overlap.py --out "$OVERLAP_REPORT"
else
    step "1/7  Overlap check (skipped)"
fi

if [ "$SKIP_MEASURE" -eq 0 ]; then
    step "2/7  Stage A: measure (GPU -- calls the reader once per mask per document)"
    python generate_labels.py measure --split train --sources "$SOURCES" --reader-model "$READER_MODEL" --device "$DEVICE" \
        --n "$N" --k "$K" --overlap-report "$OVERLAP_REPORT" --out-dir "$RUN_DIR/outcome_raw/train"
    python generate_labels.py measure --split dev --sources "$SOURCES" --reader-model "$READER_MODEL" --device "$DEVICE" \
        --n "$N" --k "$K" --out-dir "$RUN_DIR/outcome_raw/dev"
else
    step "2/7  Stage A measure (skipped)"
fi

if [ "$SKIP_FIT" -eq 0 ]; then
    step "3/7  Stage A: fit (CPU -- ridge regression + bootstrap CI)"
    python generate_labels.py fit --in-dir "$RUN_DIR/outcome_raw/train" \
        --select-alpha-from "$RUN_DIR/outcome_raw/dev" --out-dir "$RUN_DIR/outcome_labels_raw/train"
else
    step "3/7  Stage A fit (skipped)"
fi

if [ "$SKIP_DECOMPOSE" -eq 0 ]; then
    step "4/7  Stage B: position-signal decomposition (CPU)"
    python decompose_labels.py --in-dir "$RUN_DIR/outcome_labels_raw/train" --out-dir "$RUN_DIR/training_labels/train"
else
    step "4/7  Stage B (skipped)"
fi

if [ "$SKIP_TRAIN" -eq 0 ]; then
    step "5/7  Stage C: train relevance_v2 (raw) + relevance_v3 (residual) -- real gradient training, GPU"
    python train_relevance.py --label-variant raw --training-labels-dir "$RUN_DIR/training_labels/train" \
        --base-checkpoint "$BASE_CHECKPOINT" --epochs "$EPOCHS" --device "$DEVICE" --out-dir "$RUN_DIR/models/relevance_v2"
    python train_relevance.py --label-variant residual --training-labels-dir "$RUN_DIR/training_labels/train" \
        --base-checkpoint "$BASE_CHECKPOINT" --epochs "$EPOCHS" --device "$DEVICE" --out-dir "$RUN_DIR/models/relevance_v3"
else
    step "5/7  Stage C training (skipped)"
fi

if [ "$SKIP_SWEEP" -eq 0 ]; then
    step "6/7  Stage 6: lambda sweep on dev splits (GPU, both checkpoints)"
    python evaluate_relevance.py sweep --relevance-checkpoint "$RUN_DIR/models/relevance_v2" --sources "$SOURCES" \
        --reader-model "$READER_MODEL" --device "$DEVICE" --n "$N" --out "$RUN_DIR/relevance_v2_sweep.json"
    python evaluate_relevance.py sweep --relevance-checkpoint "$RUN_DIR/models/relevance_v3" --sources "$SOURCES" \
        --reader-model "$READER_MODEL" --device "$DEVICE" --n "$N" --out "$RUN_DIR/relevance_v3_sweep.json"
else
    step "6/7  Stage 6 sweep (skipped)"
fi

if [ "$SKIP_EVAL" -eq 0 ]; then
    step "7/7  Stage 6: the ONE-TIME official numbers -- Vietnamese + English test pools (both checkpoints)"
    LAM_V2="$(python -c "import json; print(json.load(open('$RUN_DIR/relevance_v2_sweep.json'))['chosen_lambda'])")"
    LAM_V3="$(python -c "import json; print(json.load(open('$RUN_DIR/relevance_v3_sweep.json'))['chosen_lambda'])")"
    python evaluate_relevance.py evaluate --relevance-checkpoint "$RUN_DIR/models/relevance_v2" \
        --reader-model "$READER_MODEL" --device "$DEVICE" --lam "$LAM_V2" --out "$RUN_DIR/relevance_v2_official.json"
    python evaluate_relevance.py evaluate --relevance-checkpoint "$RUN_DIR/models/relevance_v3" \
        --reader-model "$READER_MODEL" --device "$DEVICE" --lam "$LAM_V3" --out "$RUN_DIR/relevance_v3_official.json"
else
    step "7/7  Stage 6 evaluate (skipped)"
fi

step "Done"
echo "All artifacts under $RUN_DIR/"
