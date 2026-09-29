#!/usr/bin/env bash
# ttcompress: the whole cluster campaign of TODO.md §1 in ONE self-contained job script.
#
#   build @@BUILD_ID@@ (@@BUILD_INFO@@)
#
# Submit this file as the job's only command:    bash ttcompress_job.sh
# It carries the repository (base64 tar.gz at the bottom), unpacks it to $BASE/code/<build> and runs
# scripts/cluster_campaign.sh from there: smoke test -> probe of each published compressor -> pilot -> full
# run -> distractor ablations -> comparison -> HuggingFace upload. Re-submitting this file (or a newer build)
# resumes the campaign where the last job stopped. Progress: the job's stdout, and STATUS.md + every report
# in the private HF dataset repo <token user>/ttcompress-campaign-<CAMPAIGN>.
#
# Settings: see the top of scripts/cluster_campaign.sh (BASE, CAMPAIGN, PHASES, PILOT_SIZES, MAIN_SIZES,
# CANDIDATE_ARMS, HF_TOKEN_SOURCES, ...). The job's environment wins over the values baked in below
# (python scripts/build_cluster_job.py --set KEY=VALUE), which win over the defaults.
# TTC_UNPACK_ONLY=1 only unpacks the code and prints where.
set -euo pipefail

# ---- baked settings ----
@@SETTINGS@@
# ----
: "${BASE:=/mnt/hps/anhm-paper/ttcompress}"
export BASE
BUILD_ID='@@BUILD_ID@@'
CODE_DIR="$BASE/code/$BUILD_ID"

unpack() {  # unpack <dir>: the repository as it was at build time
  base64 -d <<'__TTCOMPRESS_PAYLOAD__' | tar -xzf - -C "$1"
@@PAYLOAD@@
__TTCOMPRESS_PAYLOAD__
}

main() {
  if [[ ! -f "$CODE_DIR/.unpacked" ]]; then
    local tmp="$CODE_DIR.tmp.$$"
    rm -rf "$tmp" && mkdir -p "$tmp"
    unpack "$tmp"
    echo "$BUILD_ID" > "$tmp/BUILD_ID"
    touch "$tmp/.unpacked"
    rm -rf "$CODE_DIR" && mv "$tmp" "$CODE_DIR"
  fi
  echo "ttcompress build $BUILD_ID -> $CODE_DIR"
  [[ "${TTC_UNPACK_ONLY:-0}" == 1 ]] && exit 0
  exec bash "$CODE_DIR/scripts/cluster_campaign.sh"
}

main "$@"
