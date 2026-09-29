#!/usr/bin/env bash
# The cluster work of TODO.md §1 as ONE unattended job: smoke test -> a probe of every published compressor ->
# pilot -> full run -> distractor ablations -> comparison -> HuggingFace upload. Started by the self-contained
# job script (python scripts/build_cluster_job.py -> dist/ttcompress_job.sh), which carries this repository
# and unpacks it to $BASE/code/<build>; it also runs from a checkout:
#
#   BASE=/mnt/hps/anhm-paper/ttcompress bash scripts/cluster_campaign.sh
#
# Built for a user who can submit a job but not log into the cluster:
#   - resumable: a finished phase leaves $STATE_DIR/<phase>.done and run_pipeline.sh resumes inside a phase,
#     so re-submitting the job (the same build or a newer one) continues where the last one stopped;
#   - a published compressor that cannot run here (package install, gated model, API change) is dropped after
#     its probe instead of failing the campaign; the next submission probes it again, and once it works the
#     finished runs get it too (select + answer + report only);
#   - progress: stdout, and the private HF dataset repo $STATUS_REPO (STATUS.md, every finished run's report,
#     log tails, a projection of the full run from the pilot) after every phase and every HEARTBEAT_MIN min;
#   - a failed phase is retried (PHASE_ATTEMPTS, resuming); if the smoke test, the pilot or the full run still
#     fails the job stops; a failed ablation does not stop the other one.
set -uo pipefail

# ---- settings: the job's environment wins, then build_cluster_job.py --set, then these defaults ----
: "${BASE:=/mnt/hps/anhm-paper/ttcompress}"        # NFS root: code/, runs/, campaigns/, packages
: "${CAMPAIGN:=coling2027}"                        # names the state dir, the run dirs and the status repo
: "${PHASES:=smoke probe pilot main abl_hard abl_pad compare upload}"   # in order; finished ones are skipped
: "${PILOT_SIZES:=300 30 50}"                      # N_TRAIN N_DEV N_TEST
: "${MAIN_SIZES:=3000 300 500}"
: "${CANDIDATE_ARMS:=}"                            # published compressors to probe; empty = EXTRA_ARMS of .env.example, none = no baselines
: "${ABL_PAD_CHARS:=20000}"                        # MULTIHOP_PAD_CHARS of the abl_pad run
: "${ABL_EXTRA_SEEDS:=none}"                       # the ablations skip the training-seed reruns
: "${PHASE_ATTEMPTS:=2}"                           # tries per phase (and per probe); a retry resumes
: "${HEARTBEAT_MIN:=30}"                           # status upload period during a phase (0 = off)
: "${STATUS_REPO:=}"                               # default <HF_NAMESPACE or token user>/ttcompress-campaign-<CAMPAIGN>
: "${UPLOAD_LABELS:=false}"                        # final upload with the per-document labels (~100k files, hours)
: "${HF_HOME:=/mnt/hps/anhm-paper/hf_cache}"
: "${LABELS:=$BASE/runs/labels_v2}"                # Stage A measurements, shared by every run and campaign
: "${PYLIBS:=$BASE/pylibs}"                        # python packages the image lacks
: "${BASELINE_SITE:=$BASE/sites/baseline}"         # run_pipeline.sh: published-compressor packages
: "${LLMLINGUA_SITE:=$BASE/sites/llmlingua}"       # run_pipeline.sh: transformers 4.46.3 + llmlingua
: "${PIP_INDEX_URL:=https://hub.fci.vn/repository/pypi/simple/}"   # pypi.org timed out in the cluster (2026-09-28)
: "${NUM_GPUS:=}"                                  # empty = every GPU nvidia-smi lists
# where an HF token may already be on the NFS (first hit wins; HF_TOKEN in the environment wins over all)
: "${HF_TOKEN_SOURCES:=$HF_HOME/token $BASE/.env /mnt/hps/anhm-paper/ttcompress_new/.env}"

CODE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CAMPAIGN_DIR=$BASE/campaigns/$CAMPAIGN
STATE_DIR=$CAMPAIGN_DIR/state
LOG_DIR=$CAMPAIGN_DIR/logs
RUNS=$BASE/runs/$CAMPAIGN
mkdir -p "$STATE_DIR" "$LOG_DIR" "$RUNS"
export BASE CAMPAIGN STATUS_REPO HF_HOME PIP_INDEX_URL
export PIP_DEFAULT_TIMEOUT=${PIP_DEFAULT_TIMEOUT:-60} PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1

read -r P_TRAIN P_DEV P_TEST <<< "$PILOT_SIZES"
read -r M_TRAIN M_DEV M_TEST <<< "$MAIN_SIZES"
# the sizes are in the names: run_pipeline.sh refuses a RUN_ROOT that holds another configuration
PILOT_ROOT=$RUNS/pilot_$P_TRAIN-$P_DEV-$P_TEST
MAIN_ROOT=$RUNS/main_$M_TRAIN-$M_DEV-$M_TEST
SMOKE_ROOT=$RUNS/smoke                                   # run_pipeline.sh: SMOKE=1 -> <dirname RUN_ROOT>/smoke
SMOKE_ENV=(SMOKE=1 RUN_ROOT="$RUNS/main")
COMPARE_ARMS=lead,random,bm25,embed,reranker,provence,xprovence,recomp,exit,llmlingua,longllmlingua,llmlingua2,span_sup,ours_beta,ours_ens,abl_logprob,abl_posadj,oracle_span,oracle_beta

ST_TIMEOUT=900
say() { printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG_DIR/campaign.log"; }
st() {  # campaign bookkeeping (scripts/campaign_status.py); never fatal
  timeout "$ST_TIMEOUT" python "$CODE_DIR/scripts/campaign_status.py" --dir "$CAMPAIGN_DIR" "$@" \
    || say "(status '$1' failed)"
}
dotenv() {  # dotenv <KEY>: the value run_pipeline.sh will use (the environment, else .env.example)
  local v="${!1:-}"
  if [[ -n "$v" ]]; then echo "$v"; return; fi
  sed -nE "s/^$1=\"?([^\"]*)\"?[[:space:]]*\$/\1/p" "$CODE_DIR/.env.example" | tail -n 1
}
last_error() { grep -E '^!!|Error|error:' "$1" 2>/dev/null | tail -n 2 | cut -c1-300 | tr '\n' ' '; }
arm_label() { if [[ "$1" == *=* ]]; then echo "${1%%=*}"; else echo "${1%%:*}"; fi; }

# ---- setup ----
take_lock() {
  if command -v flock >/dev/null 2>&1; then
    exec 9>"$CAMPAIGN_DIR/lock"
    # waits a little: the NFS lock of a pod that was just killed lingers until its lease expires
    flock -w 180 9 || { echo "!! another job is running campaign $CAMPAIGN ($CAMPAIGN_DIR/lock)"; exit 1; }
  fi
}
count_gpus() {
  [[ -n "$NUM_GPUS" ]] || NUM_GPUS=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true)
  (( NUM_GPUS >= 1 )) || { say "!! no GPU visible (nvidia-smi -L)"; return 1; }
  (( NUM_GPUS >= 2 )) || say "(1 GPU: the held-out Qwen3-32B reader needs TP_LARGE=2 GPUs)"
}
find_hf_token() {  # never printed; only where it came from
  if [[ -n "${HF_TOKEN:-}" ]]; then say "HF token: from the job environment"; export HF_TOKEN; return; fi
  local f tok
  for f in $HF_TOKEN_SOURCES; do
    [[ -f "$f" ]] || continue
    if [[ "$f" == *.env ]]; then
      tok=$(sed -nE "s/^[[:space:]]*(export[[:space:]]+)?HF_TOKEN=[\"']?([^\"'[:space:]]*).*/\2/p" "$f" | tail -n 1)
    else
      tok=$(head -n 1 "$f" | tr -d '[:space:]')
    fi
    if [[ -n "$tok" ]]; then HF_TOKEN=$tok; export HF_TOKEN; say "HF token: from $f"; return; fi
  done
  say "HF token: none found -- gated models must already be cached, nothing is uploaded (set HF_TOKEN in the job)"
}
ensure_python_deps() {  # what run_pipeline.sh imports; the 2026-09-28 image lacked datasets/pandas/scipy and pip
  export PYTHONPATH="$PYLIBS${PYTHONPATH:+:$PYTHONPATH}"
  local m missing=() pins
  for m in datasets pandas scipy pyarrow huggingface_hub pytest packaging matplotlib; do
    python -c "import $m" 2>/dev/null || missing+=("$m")
  done
  (( ${#missing[@]} )) || return 0
  say "installing into $PYLIBS: ${missing[*]} (index $PIP_INDEX_URL)"
  pins=$(mktemp)   # the image's builds that vLLM needs stay as they are
  python - > "$pins" <<'PY'
import importlib.metadata as m
for p in ('torch', 'transformers', 'vllm', 'numpy', 'tokenizers'):
    try:
        print(f'{p}=={m.version(p)}')
    except m.PackageNotFoundError:
        pass
PY
  if python -m pip --version >/dev/null 2>&1; then
    python -m pip install -q --upgrade --target "$PYLIBS" -c "$pins" "${missing[@]}"
  elif command -v pip3 >/dev/null 2>&1; then   # the system pip3 may belong to another python: ask for its version
    pip3 install -q --upgrade --target "$PYLIBS" --only-binary=:all: \
      --python-version "$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])')" -c "$pins" "${missing[@]}"
  else
    say "!! $(command -v python) has no pip and there is no pip3"
  fi
  rm -f "$pins"
  for m in "${missing[@]}"; do
    python -c "import $m" 2>/dev/null && continue
    if [[ $m == matplotlib ]]; then say "(no matplotlib: figures can be drawn later from report.json)"; continue; fi
    say "!! $m still does not import"; return 1
  done
}

# ---- one run_pipeline.sh call ----
pipeline() {  # pipeline <log> <VAR=value...>: the environment wins over the code dir's .env (.env.example)
  local log=$1; shift
  rm -f "$CAMPAIGN_DIR/.rc"
  # in the background + wait: a TERM reaches the trap at once, not after the whole stage
  ( cd "$CODE_DIR" && env "${COMMON[@]}" "$@" bash run_pipeline.sh; echo $? > "$CAMPAIGN_DIR/.rc" ) 9>&- 2>&1 \
    | tee -a "$log" &
  wait $!
  return "$(cat "$CAMPAIGN_DIR/.rc" 2>/dev/null || echo 1)"
}
run_root_of() {
  case $1 in
    smoke) echo "$SMOKE_ROOT" ;;
    pilot) echo "$PILOT_ROOT" ;;
    main) echo "$MAIN_ROOT" ;;
    abl_hard) echo "${MAIN_ROOT}_distractors-hard" ;;     # run_pipeline.sh appends the ablation suffix
    abl_pad) echo "${MAIN_ROOT}_pad-$ABL_PAD_CHARS" ;;
  esac
}
set_run_env() {  # RUN_ENV = run_pipeline.sh settings of pilot | main | abl_hard | abl_pad
  case $1 in
    pilot) RUN_ENV=(RUN_ROOT="$PILOT_ROOT" N_TRAIN="$P_TRAIN" N_DEV="$P_DEV" N_TEST="$P_TEST" HF_RUN_NAME=pilot) ;;
    main) RUN_ENV=(RUN_ROOT="$MAIN_ROOT" N_TRAIN="$M_TRAIN" N_DEV="$M_DEV" N_TEST="$M_TEST" HF_RUN_NAME=main) ;;
    abl_hard) RUN_ENV=(RUN_ROOT="$MAIN_ROOT" N_TRAIN="$M_TRAIN" N_DEV="$M_DEV" N_TEST="$M_TEST"
                       DISTRACTORS=hard EXTRA_SEEDS="$ABL_EXTRA_SEEDS") ;;
    abl_pad) RUN_ENV=(RUN_ROOT="$MAIN_ROOT" N_TRAIN="$M_TRAIN" N_DEV="$M_DEV" N_TEST="$M_TEST"
                      MULTIHOP_PAD_CHARS="$ABL_PAD_CHARS" EXTRA_SEEDS="$ABL_EXTRA_SEEDS") ;;
  esac
}

# ---- phases ----
run_phase() {  # run_phase <phase> <function> [args...]: once per campaign; the function gets the log last
  local phase=$1; shift
  if [[ -f "$STATE_DIR/$phase.done" ]]; then say "== $phase: finished earlier, skipped"; return 0; fi
  local log="$LOG_DIR/$phase.log"
  for ((ATTEMPT = 1; ATTEMPT <= PHASE_ATTEMPTS; ATTEMPT++)); do
    say "== $phase: attempt $ATTEMPT/$PHASE_ATTEMPTS (log $log)"
    st phase "$phase" running --note "attempt $ATTEMPT/$PHASE_ATTEMPTS" --log "$log"
    st publish --heartbeat
    if "$@" "$log"; then
      touch "$STATE_DIR/$phase.done"
      st phase "$phase" done --log "$log"
      st publish
      return 0
    fi
    say "!! $phase: attempt $ATTEMPT failed: $(last_error "$log")"
    (( ATTEMPT < PHASE_ATTEMPTS )) && sleep 60
  done
  st phase "$phase" failed --note "$(last_error "$log")" --log "$log"
  st publish
  return 1
}
phase_smoke() {  # every stage on a few documents, without the published compressors (they get probes)
  local keep=0; (( ATTEMPT > 1 )) && keep=1   # a retry resumes instead of starting the smoke over
  st current smoke --log "$1" --run-root "$SMOKE_ROOT"
  pipeline "$1" "${SMOKE_ENV[@]}" SMOKE_KEEP=$keep EXTRA_ARMS= \
    STAGES="preflight prefetch labels fit ensemble train select answer report"
}
working_from_markers() {  # WORKING_ARMS = the candidates whose probe succeeded (this or an earlier job)
  local arm; WORKING_ARMS=""
  for arm in ${CANDIDATES[@]+"${CANDIDATES[@]}"}; do
    [[ -n "$arm" && -f "$STATE_DIR/probe_$(arm_label "$arm").ok" ]] && WORKING_ARMS+=",$arm"
  done
  WORKING_ARMS=${WORKING_ARMS#,}
}
probe_arms() {  # one select on the smoke run per published compressor not known to work yet
  local arm label log result ok attempt
  for arm in ${CANDIDATES[@]+"${CANDIDATES[@]}"}; do
    [[ -n "$arm" ]] || continue
    label=$(arm_label "$arm")
    [[ -f "$STATE_DIR/probe_$label.ok" ]] && continue
    log="$LOG_DIR/probe_$label.log"; ok=0; result=""
    for ((attempt = 1; attempt <= PHASE_ATTEMPTS; attempt++)); do
      say "== probe $label ($arm), attempt $attempt/$PHASE_ATTEMPTS"
      st arm "$label" probing --spec "$arm" --note "attempt $attempt"
      st current "probe_$label" --log "$log" --run-root "$SMOKE_ROOT"
      # prefetch: a gated model fails here; select: packages, model load, every smoke document
      if pipeline "$log" "${SMOKE_ENV[@]}" SMOKE_KEEP=1 EXTRA_ARMS="$arm" STAGES="prefetch select"; then
        if result=$(python "$CODE_DIR/scripts/campaign_status.py" --dir "$CAMPAIGN_DIR" probe-result \
                     --eval-dir "$SMOKE_ROOT/results/eval_test" --arm "$label"); then ok=1; break; fi
      else
        result=$(last_error "$log")
      fi
    done
    if (( ok )); then
      touch "$STATE_DIR/probe_$label.ok"; st arm "$label" ok --spec "$arm" --note "$result"
    else
      st arm "$label" failed --spec "$arm" --note "$result"; say "!! $label dropped: $result"
    fi
    st publish --heartbeat
  done
  working_from_markers
}
phase_smoke_arms() {  # answer + report with the working compressors, before the pilot depends on them
  st current smoke_arms --log "$1" --run-root "$SMOKE_ROOT"
  pipeline "$1" "${SMOKE_ENV[@]}" SMOKE_KEEP=1 EXTRA_ARMS="$WORKING_ARMS" STAGES="answer report"
}
phase_run() {  # phase_run <run> <log>: one run in three timed stage groups (the pilot's times -> projection)
  local run=$1 log=$2 group t0 root
  set_run_env "$run"; root=$(run_root_of "$run")
  st current "$run" --log "$log" --run-root "$root"
  for group in "preflight prefetch labels" "fit ensemble train" "select answer report"; do
    t0=$(date +%s)
    pipeline "$log" "${RUN_ENV[@]}" EXTRA_ARMS="$WORKING_ARMS" STAGES="$group" || return 1
    st timing "$run" "${group%% *}" $(( $(date +%s) - t0 ))
  done
  echo "$WORKING_ARMS" > "$STATE_DIR/$run.arms"
  st run "$run" "$root" --arms "$WORKING_ARMS"
}
catch_up() {  # finished runs get the published compressors that began to work after they ran
  local run root arm missing
  for run in pilot main abl_hard abl_pad; do
    [[ -f "$STATE_DIR/$run.done" ]] || continue
    missing=""
    for arm in ${WORKING_ARMS//,/ }; do
      [[ ",$(cat "$STATE_DIR/$run.arms" 2>/dev/null)," == *",$arm,"* ]] || missing+=" $(arm_label "$arm")"
    done
    [[ -n "$missing" ]] || continue
    set_run_env "$run"; root=$(run_root_of "$run")
    say "== catch-up $run:$missing"
    st current "catch-up $run" --log "$LOG_DIR/$run.catchup.log" --run-root "$root"
    if pipeline "$LOG_DIR/$run.catchup.log" "${RUN_ENV[@]}" EXTRA_ARMS="$WORKING_ARMS" STAGES="select answer report"; then
      echo "$WORKING_ARMS" > "$STATE_DIR/$run.arms"
      st run "$run" "$root" --arms "$WORKING_ARMS"
      rm -f "$STATE_DIR/compare.done" "$STATE_DIR/upload.done"   # redo them with the new arms
    else
      say "!! catch-up of $run failed: $(last_error "$LOG_DIR/$run.catchup.log")"
    fi
  done
}
compare_runs() {  # COMPARE_RUNS = --run args of the finished full run and ablations
  local run f; COMPARE_RUNS=()
  for run in main abl_hard abl_pad; do
    f="$(run_root_of "$run")/results/eval_test/report.json"
    [[ -f "$STATE_DIR/$run.done" && -f "$f" ]] && COMPARE_RUNS+=(--run "${run#abl_}=$f")
  done
}
phase_compare() {
  st current compare --log "$1" --run-root "$RUNS/compare"
  (cd "$CODE_DIR" && python scripts/compare_runs.py --primary-reader "$PRIMARY_TAG" --out-dir "$RUNS/compare" \
     --arms "$COMPARE_ARMS" "${COMPARE_RUNS[@]}") >> "$1" 2>&1 || return 1
  st run compare "$RUNS/compare"
}
phase_upload() {  # the full run's pruners (one model repo each) + evaluation outputs (one dataset repo)
  [[ -n "${HF_TOKEN:-}" ]] || { echo "!! no HF token: cannot upload" >> "$1"; return 1; }
  set_run_env main
  st current upload --log "$1" --run-root "$MAIN_ROOT"
  pipeline "$1" "${RUN_ENV[@]}" HF_UPLOAD_LABELS="$UPLOAD_LABELS" STAGES=upload
}

finish() {  # finish <exit code> <message>
  say "== $2"
  st info result "$2 ($(date -u +%FT%TZ))"
  st publish
  exit "$1"
}
on_signal() {
  ST_TIMEOUT=25   # the kubelet kills the pod after its grace period
  say "!! terminated by a signal: re-submit the job to resume"
  st info result "interrupted $(date -u +%FT%TZ): re-submit the job to resume"
  st publish --heartbeat
  exit 143
}

main() {
  take_lock
  trap on_signal TERM INT
  trap '[[ -n "${HEARTBEAT_PID:-}" ]] && kill "$HEARTBEAT_PID" 2>/dev/null' EXIT
  say "== campaign $CAMPAIGN, build $(cat "$CODE_DIR/BUILD_ID" 2>/dev/null || echo checkout), code $CODE_DIR"
  st info build "$(cat "$CODE_DIR/BUILD_ID" 2>/dev/null || echo checkout)"
  st info result ""
  count_gpus || finish 1 "no GPU"
  st info gpus "$NUM_GPUS"
  find_hf_token
  ensure_python_deps || finish 1 "python packages could not be installed (see logs/campaign.txt)"
  [[ -f "$CODE_DIR/.env" ]] || cp "$CODE_DIR/.env.example" "$CODE_DIR/.env"   # HF_TOKEN comes from the environment
  mkdir -p "$BASELINE_SITE" "$LLMLINGUA_SITE"
  COMMON=(NUM_GPUS="$NUM_GPUS" LABELS="$LABELS" HF_HOME="$HF_HOME" BASELINE_SITE="$BASELINE_SITE"
          LLMLINGUA_SITE="$LLMLINGUA_SITE")
  local readers primary
  readers=$(dotenv LABEL_READERS); primary=${readers%% *}; PRIMARY_TAG=${primary//\//--}
  CANDIDATE_ARMS=${CANDIDATE_ARMS:-$(dotenv EXTRA_ARMS)}; CANDIDATE_ARMS=${CANDIDATE_ARMS// /}
  [[ "$CANDIDATE_ARMS" == none ]] && CANDIDATE_ARMS=""
  IFS=',' read -r -a CANDIDATES <<< "$CANDIDATE_ARMS"
  st info phases "$PHASES"
  st info sizes "pilot $PILOT_SIZES · main $MAIN_SIZES"
  st info runs "$RUNS"
  st info labels "$LABELS"
  working_from_markers
  if (( HEARTBEAT_MIN > 0 )); then
    ( trap - TERM INT EXIT; while sleep $(( HEARTBEAT_MIN * 60 )); do st publish --heartbeat; done ) 9>&- &
    HEARTBEAT_PID=$!
  fi

  local phase failed=()
  for phase in $PHASES; do
    case $phase in
      smoke) run_phase smoke phase_smoke || finish 1 "smoke test failed: the pipeline itself is broken here" ;;
      probe)
        if [[ ! -f "$STATE_DIR/smoke.done" ]]; then say "probe: needs a finished smoke phase; skipped"; continue; fi
        probe_arms
        say "published compressors that work here: ${WORKING_ARMS:-none}"
        if [[ -n "$WORKING_ARMS" ]]; then
          # redone whenever the set of working compressors changed
          [[ "$(cat "$STATE_DIR/smoke_arms.done" 2>/dev/null)" == "$WORKING_ARMS" ]] || rm -f "$STATE_DIR/smoke_arms.done"
          if run_phase smoke_arms phase_smoke_arms; then
            echo "$WORKING_ARMS" > "$STATE_DIR/smoke_arms.done"
          else
            say "!! answer/report with the published compressors failed on the smoke run: continuing WITHOUT them"
            for a in ${CANDIDATES[@]+"${CANDIDATES[@]}"}; do [[ -n "$a" ]] && rm -f "$STATE_DIR/probe_$(arm_label "$a").ok"; done
            for a in ${WORKING_ARMS//,/ }; do
              st arm "$(arm_label "$a")" failed --note "answer/report with the published arms failed on the smoke run (logs/smoke_arms.txt)"
            done
            WORKING_ARMS=""
          fi
        fi
        st info working_arms "${WORKING_ARMS:-none}"
        catch_up ;;
      pilot)
        run_phase pilot phase_run pilot || finish 1 "pilot failed"
        [[ -f "$STATE_DIR/main.done" ]] || st projection --code-dir "$CODE_DIR" --labels "$LABELS" \
          --gpus "$NUM_GPUS" --pilot-sizes "$PILOT_SIZES" --main-sizes "$MAIN_SIZES" ;;
      main) run_phase main phase_run main || finish 1 "full run failed" ;;
      abl_hard|abl_pad)
        if [[ ! -f "$STATE_DIR/main.done" ]]; then say "$phase: needs the full run (its labels); skipped"; continue; fi
        run_phase "$phase" phase_run "$phase" || failed+=("$phase") ;;
      compare)
        compare_runs
        if (( ${#COMPARE_RUNS[@]} < 4 )); then say "compare: needs the full run and an ablation; skipped"; continue; fi
        run_phase compare phase_compare || failed+=(compare) ;;
      upload)
        if [[ ! -f "$STATE_DIR/main.done" ]]; then say "upload: needs the full run; skipped"; continue; fi
        run_phase upload phase_upload || failed+=(upload) ;;
      *) finish 1 "unknown phase '$phase' in PHASES" ;;
    esac
  done
  if (( ${#failed[@]} )); then finish 1 "finished, but these phases failed: ${failed[*]}"; fi
  finish 0 "finished: $PHASES"
}

main "$@"
