#!/usr/bin/env bash
# Follow-up experiments to the 2026-09-29 full run (reports/RUN_REPORT_2026-09-30_full_h100x4.md and the
# per-document analysis of 2026-09-30, METHOD_SPEC.md §9). Every step goes through run_pipeline.sh with the
# run's own .env, reuses the main run's labels, pruners and selections, and is resumable: finished
# selections / answers / labels are skipped, so re-running a step after a crash continues it.
#
#   bash scripts/followup.sh                    # every step, in order
#   bash scripts/followup.sh bench arms         # some of them
#
# Steps, cheapest first (4xH100 estimates from the full run's throughput):
#   bench   selection latency, one arm at a time on one GPU (~20 min). The select stage's per-document times
#           come from four shards side by side: ours_beta measured 36-76 ms, its identical seed reruns 12-50.
#   arms    exploratory arms on the main documents: ours / span_sup / reranker scoring sentences (sent+) and
#           filling the budget that paragraphs leave with sentences (fill+); LLMLingua and LongLLMLingua asked
#           for the budget in tokens (_tt; the rate form was cut to the budget on up to half the documents);
#           and every arm at 16x / 32x on the single-hop sources, whose 4x / 8x budgets keep 5-9 chunks and
#           therefore the needle for every chunk arm. Then every reader answers and the report is rebuilt
#           (confirmatory families stay on 4x / 8x; the first report is kept as first_report.*). ~3-4 h.
#   oracle  ORACLE_N=500: oracle_beta labels for 400 more test documents per source (primary reader), so the
#           non-inferiority test of H1-oracle is not decided by 100 documents. ~1-1.5 h.
#   units   MULTIHOP_UNITS=sentence ablation: multi-hop documents with one chunk per sentence (supporting-fact
#           sentences as gold) for labels, pruners and evaluation; single-hop labels come from the main run.
#           Evaluated on the multi-hop sources only, compared with the main run by doc_id. ~5-6 h.
#   hard    DISTRACTORS=hard ablation: single-hop haystacks padded from the needle's own article. ~5-6 h.
#   compare scripts/compare_runs.py: main vs the ablations that finished.
#
# Knobs: FOLLOWUP_READERS (label readers of the two ablations, e.g. "Qwen/Qwen3-8B" to save ~2/3 of their
# label time; the ensemble pruner then has one reader), ABL_SEEDS (training seeds of the ablations' pruners,
# default none), ORACLE_N (default 500), EXTRA_RATIOS_SINGLE (default 16,32).
set -euo pipefail
cd "$(dirname "$0")/.."

STEPS=("$@")
(( ${#STEPS[@]} )) || STEPS=(bench arms oracle units hard compare)
ABL_SEEDS=${ABL_SEEDS:-none}

# Every step lists the follow-up arms and the single-hop extra ratios, so a later step never selects fewer
# arms than an earlier one on the same documents.
export FOLLOWUP_ARMS=1 EXTRA_RATIOS_SINGLE=${EXTRA_RATIOS_SINGLE:-16,32}
export REPORT_OURS=${REPORT_OURS:-ours_beta,ours_ens,ours_fill,ours_sent}

ablation_env=()
[[ -n "${FOLLOWUP_READERS:-}" ]] && ablation_env+=("LABEL_READERS=$FOLLOWUP_READERS")

setting() {  # setting <KEY> <default>: the environment's value, else .env's, else the default (as run_pipeline.sh)
  local key=$1 value
  if [[ -n "${!key:-}" ]]; then echo "${!key}"; return; fi
  value=$( { grep -E "^[[:space:]]*(export[[:space:]]+)?$key=" .env 2>/dev/null || true; } | tail -n 1 | sed -E "s/^[^=]*=//; s/\r$//; s/^[\"']//; s/[\"']$//")
  echo "${value:-$2}"
}

for step in "${STEPS[@]}"; do
  echo "================ follow-up step: $step ================"
  case $step in
    bench)
      STAGES=bench bash run_pipeline.sh ;;
    arms)
      STAGES="select answer report" bash run_pipeline.sh ;;
    oracle)
      # labels: every (reader, source, split) already measured returns before loading a model; only the
      # primary reader's test documents 101-500 are new. fit: CPU refits (deterministic, same train/dev labels).
      ORACLE_N=${ORACLE_N:-500} STAGES="labels fit select answer report" bash run_pipeline.sh ;;
    units)
      # EXTRA_ARMS empty: the published compressors are compared on the main run's paragraph documents
      env MULTIHOP_UNITS=sentence EVAL_SOURCES=vimqa,hotpotqa,2wiki EXTRA_SEEDS="$ABL_SEEDS" FOLLOWUP_ARMS=0 \
        EXTRA_RATIOS_SINGLE= EXTRA_ARMS= ${ablation_env[@]+"${ablation_env[@]}"} \
        STAGES="prefetch labels fit ensemble train select answer report" bash run_pipeline.sh ;;
    hard)
      env DISTRACTORS=hard EVAL_SOURCES=uit_viquad,xquad_vi EXTRA_SEEDS="$ABL_SEEDS" FOLLOWUP_ARMS=0 \
        ${ablation_env[@]+"${ablation_env[@]}"} \
        STAGES="prefetch labels fit ensemble train select answer report" bash run_pipeline.sh ;;
    compare)
      root=$(setting RUN_ROOT .)
      readers=$(setting LABEL_READERS "Qwen/Qwen3-8B")
      primary=$(echo "$readers" | awk '{print $1}'); primary=${primary//\//--}
      runs=(--run "main=$root/results/eval_test/report.json")
      for abl in units-sentence distractors-hard; do
        f="${root}_$abl/results/eval_test/report.json"
        if [[ -f "$f" ]]; then runs+=(--run "${abl}=$f"); fi
      done
      if (( ${#runs[@]} < 4 )); then echo "compare: no finished ablation yet, skipped"; continue; fi
      python scripts/compare_runs.py --primary-reader "$primary" --out-dir "$(dirname "$root")/compare_followup" \
        --arms "${COMPARE_ARMS:-lead,bm25,embed,reranker,xprovence,recomp,exit,span_sup,ours_beta,ours_ens,ours_fill,ours_sent,oracle_span,oracle_beta}" \
        "${runs[@]}" ;;
    *)
      echo "unknown step $step (bench arms oracle units hard compare)"; exit 1 ;;
  esac
done
