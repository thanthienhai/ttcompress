#!/usr/bin/env bash
# Downloads + assembles everything under data/ -- nothing in data/*.json or
# data/*.jsonl is committed to git (see .gitignore); run this once before
# train.py/evaluate.py. See data/SOURCES.md for exact provenance of each file.
#
#   scripts/fetch_data.sh            # skip files that already exist
#   scripts/fetch_data.sh --force    # re-download everything
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$REPO_ROOT/data"
FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

mkdir -p "$DATA_DIR"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

RULER_URL_LIST="https://raw.githubusercontent.com/NVIDIA/RULER/main/scripts/data/synthetic/json/PaulGrahamEssays_URLs.txt"
LONGBENCH_ZIP="https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip"

# ---------------------------------------------------------------------------
# 1. Paul Graham essays -- haystack corpus for RULER niah_single_1 + classic
#    Kamradt NIAH (data/SOURCES.md #1). Only the plaintext (non-.html) URLs
#    in RULER's own list, which are RULER's own source:
#    gkamradt/LLMTest_NeedleInAHaystack's bundled essay .txt files -- so this
#    reproduces the exact corpus both recipes use, no HTML-scraping deps needed.
# ---------------------------------------------------------------------------
OUT_ESSAYS="$DATA_DIR/paul_graham_essays.json"
if [ -f "$OUT_ESSAYS" ] && [ "$FORCE" -eq 0 ]; then
    echo "[skip] $OUT_ESSAYS already exists (use --force to re-fetch)"
else
    echo "[1/2] Fetching Paul Graham essay corpus (RULER + Kamradt NIAH haystack)..."
    curl -sSL "$RULER_URL_LIST" -o "$TMP/urls.txt"
    grep -v '\.html' "$TMP/urls.txt" > "$TMP/plaintext_urls.txt" || true
    mkdir -p "$TMP/essays"
    n=0
    while IFS= read -r url; do
        [ -z "$url" ] && continue
        name="$(basename "$url")"
        if curl -sSL "$url" -o "$TMP/essays/$name"; then
            n=$((n + 1))
        else
            echo "  [WARN] failed to fetch $url"
        fi
    done < "$TMP/plaintext_urls.txt"
    echo "  downloaded $n essay files"
    python "$SCRIPT_DIR/build_paul_graham_essays.py" --essays-dir "$TMP/essays" --out "$OUT_ESSAYS"
fi

# ---------------------------------------------------------------------------
# 2. LongBench passage_retrieval_en -- real published split (data/SOURCES.md
#    #2). The HF repo ships one data.zip; extract just the one file we need
#    instead of pulling all ~21 LongBench task files to disk.
# ---------------------------------------------------------------------------
OUT_LONGBENCH="$DATA_DIR/longbench_passage_retrieval_en.jsonl"
if [ -f "$OUT_LONGBENCH" ] && [ "$FORCE" -eq 0 ]; then
    echo "[skip] $OUT_LONGBENCH already exists (use --force to re-fetch)"
else
    echo "[2/2] Fetching LongBench passage_retrieval_en split..."
    curl -sSL "$LONGBENCH_ZIP" -o "$TMP/longbench_data.zip"
    unzip -p "$TMP/longbench_data.zip" data/passage_retrieval_en.jsonl > "$OUT_LONGBENCH"
    n_lines="$(wc -l < "$OUT_LONGBENCH" | tr -d ' ')"
    echo "  wrote $n_lines samples to $OUT_LONGBENCH"
fi

echo "Done. data/ contents:"
ls -la "$DATA_DIR"
