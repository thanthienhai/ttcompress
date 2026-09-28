# ttcompress Pilot Run Report — 4×H100, stg/llms-dev

**Run date**: 2026-09-26 (started 05:45 UTC, completed ~06:52 UTC for main pipeline)
**Run name**: `pilot` (N_TRAIN=60, N_DEV=6, N_TEST=10 — pilot scale per ./env)
**Cluster**: stg / llms-dev, node `fke-ncp-modas-stg-qc8ifaxe-modas-workers-z1-f9f69-dn24z`
**Pod**: `anhm-ttcompress-train-h100x4` (4× NVIDIA H100 80GB HBM3)
**Pipeline**: 10 stages — `preflight prefetch labels fit ensemble train select answer report upload`

## 1. Status: ✅ Success

All main pipeline stages completed with **no failures** (`!! |FAILED|Error` matches empty). 9/10 stages done. Final `upload` stage uploaded 5 pruners to HuggingFace successfully; dataset upload (`thanthienhai/ttcompress-pilot-eval`) was still in progress at release time (50MB over slow HF link).

Speechcheck per-stage markers (all reached):
- ✅ `preflight`: 21 unit tests pass; torch 2.11.0+cu130, vllm 0.26.0, 4 H100s visible
- ✅ `prefetch`: 14 dataset splits cached; 6 models downloaded to `/mnt/hps/fp16_models`
- ✅ `labels`: 18 (reader × source × split) measure jobs across 3 readers × 4 GPUs, no shard failed
- ✅ `fit`: 36 ridge-fit jobs (3 readers × 2 targets × 6 splits), no failures
- ✅ `ensemble`: 6 ensemble merges over train/dev × 3 sources
- ✅ `train`: 5 pruners trained across 4 GPUs (parallel batches of 4, then 1), each 3 epochs
- ✅ `select`: 4 shards (one per GPU), 13 arms × 2 ratios × 5 sources
- ✅ `answer`: 4 readers, last one (Qwen3-32B at tp=2) ran 2 shards across 4 GPUs — completed
- ✅ `report`: produces `results/eval_test/report.md` (188KB) + `report.json` (929KB) — 460 metric cells
- ⏳ `upload`: 5 model repos uploaded ✅; eval+labels dataset upload in progress

## 2. Models Trained (best-only per arm, saved to HPS + HuggingFace)

| Pruner | Best epoch | Best dev metric | n_train | n_dev | Size |
|---|---|---|---|---|---|
| `pruner_beta_primary` | ep2 | ndcg@3_beta=0.7312 | 176 | 18 | 2.2G |
| `pruner_beta_ensemble` | ep3 | ndcg@3_beta=0.7513 | 180 | 18 | 2.2G |
| `pruner_span` | ep1 | gold_recall@25%=0.778 | 180 | 18 | 2.2G |
| `pruner_logprob_primary` | ep3 | ndcg@3_beta=0.7666 | 180 | 18 | 2.2G |
| `pruner_beta_primary_posadj` | ep3 | ndcg@3_beta=0.7332 | 176 | 18 | 2.2G |

**Only best checkpoint saved per arm** (per `train_pruner.py:154-161` `save_pretrained` overwrites `--out-dir` on improvement; the `best==-inf` fallback only fires when no epoch ever saved). Total models on disk: **11 GB**.

## 3. Evaluation Results (Paired Cluster-Bootstrap CIs, no cross-source pooling)

Per spec: Token F1 with 95% cluster bootstrap CIs; never pooled across sources. 460 metric cells = 4 readers × 5 sources × ~23 arms × 2 ratios (4×, 8×). 880 paired F1 diffs at the cell level.

### 3.1 Reader-agnostic compression: `ours vs lead` (the headline arm)
| Metric | Value |
|---|---|
| Comparisons | 80 (ours_beta + ours_ens, across 4 readers × 5 sources × 2 ratios) |
| Wins (diff > 0) | **72 / 80** |
| Losses | 4 |
| Mean F1 diff | **+0.258** |

Example: UIT-ViQuAD ratio 8.0 with Qwen3-1.7B: `ours_beta` F1=0.670 vs `lead` F1=0.037 → **+0.633**, p<0.001.

### 3.2 vs other baselines
- `ours vs bm25`: 47 wins / 80, mean diff +0.042 (modest edge)
- `ours vs embed`: 35 wins / 80, mean diff +0.010 (essentially tied)
- `ours vs full`: 9 significant upgrades at high ratio, none at low ratio (compression never beats full context at 4× — expected)
- `ours_beta vs ours_ens`: ens slightly better (beta wins 14/40, mean diff -0.0145)

### 3.3 Statistically significant upgrades (p<0.05)
99 paired upgrades vs baselines (`lead`, `random`, `bm25`, `embed`, `full`). Top 25 captured in the extracted summary; the most consistent pattern is `ours_ens` and `ours_beta` beating `lead` and `random` across all sources at both ratios with Qwen3-1.7B (the weakest reader, where compression matters most).

### 3.4 Position robustness
`by_depth` analysis: 176 entries (per reader × source × ratio × arm), binning F1 by needle position quintile (q1=top, q5=bottom). Available in `report.json` for paper figures.

## 4. Raw Data Locations

### HPS `/mnt/hps/anhm-paper` (NFS-shared, persistent after pod deletion)
| Path | Size | Content |
|---|---|---|
| `ttcompress/runs/pilot/models/` | 11 GB | 5 best-only pruner checkpoints (2.2 G each) |
| `ttcompress/runs/labels/raw/` | 16 MB | Per-doc mask + outcome json (3 readers × 6 splits) |
| `ttcompress/runs/labels/fit/` | 25 MB | Ridge-fit per-reader + ensemble labels + `summary.json` |
| `ttcompress/runs/pilot/results/eval_test/` | 9.0 MB | report.md (188KB), report.json (929KB), 460 cells |
| `ttcompress/runs/pilot/logs/` | ~1 GB | Per-stage logs (labels shards, train, select shards, answer shards, upload) |
| `ttcompress/runs/pilot/logs/pipeline.log` | 7 KB | Top-level stage markers (full text included in report) |

### HPS `/mnt/hps/fp16_models` (HF cache shared across future pods)
Added to the shared cache during this run's `prefetch`:
- Qwen/Qwen3-1.7B (~3.4 GB, 63s download)
- Qwen/Qwen3-32B (~64 GB, ~7 min — dominates prefetch time)
- aisingapore/Llama-SEA-LION-v3-8B (~16 GB, 225s)
- BAAI/bge-m3 (~2.4 GB)
- BAAI/bge-reranker-v2-m3 (~2.2 GB)
- (Qwen/Qwen3-8B was already cached)

### HuggingFace uploads (run `pilot`, namespace `thanthienhai`)
| Repo | Type | Status |
|---|---|---|
| `thanthienhai/ttcompress-pilot-pruner-beta-primary` | model | ✅ uploaded |
| `thanthienhai/ttcompress-pilot-pruner-beta-ensemble` | model | ✅ uploaded |
| `thanthienhai/ttcompress-pilot-pruner-span` | model | ✅ uploaded |
| `thanthienhai/ttcompress-pilot-pruner-logprob-primary` | model | ✅ uploaded |
| `thanthienhai/ttcompress-pilot-pruner-beta-primary-posadj` | model | ✅ uploaded |
| `thanthienhai/ttcompress-pilot-eval` | dataset | ⏳ in-progress at pod release time |

Re-upload directly from `runs/pilot/results/eval_test/` + `runs/labels/` if needed:
```
python scripts/upload_hf.py --prefix ttcompress --run-name pilot \
  --models-dir runs/pilot/models --eval-dir runs/pilot/results/eval_test \
  --labels-dir runs/labels --include-labels
```

## 5. Cluster / Resource Usage

- **Pod**: `anhm-ttcompress-train-h100x4` in stg/llms-dev, pinned to dn24z (HPS NFS hostPath), 4× H100 (NUM_GPUS=4, requests=limits)
- **Total walltime**: ~3h45m (05:45 → 09:30 UTC). Main pipeline (excluding upload): ~1h07m. Prefetch dominated by Qwen3-32B download.
- **Image**: `hub.fci.vn/ncp-modas/containers/dynamo-vllm-runtime:1.4.1-cu130-vllm0260` (already on node) + `pip install datasets pandas pyarrow` (3 missing from base image)
- **HPS disk**: 18T total, 10T used (57%), 7.4T available — comfortable for the 60 GB Qwen3-32B + 11 GB models

## 6. Code Quality Notes (parallel Oracle audit, 0 BLOCKERs)

Two MAJOR findings worth flagging for the paper writeup (do not block the run; they're methodology interpretations):

- **`ttcompress/pruner_training.py:78-90`** — `listnet_loss` uses SUM reduction over C chunks, but `F.mse_loss` uses MEAN (PyTorch default). Effective `w_mse` per-doc scales as `0.5/log(C)` at init, so MSE contributes ~25% of listnet for short VIMQA docs (C=2) and ~2.5% for long HotpotQA docs (C=20). Spec text just says "ListNet + 0.5·MSE" — conventional for ListNet to sum (Cao 2007), and for MSE to mean. The imbalance is the unintended interaction. Consider normalizing both reductions to mean for paper reproducibility.

- **`generate_labels.py:183`** — `cmd_ensemble` silently intersects doc sets across readers without logging the drop count. Unreachable in normal flow (the pipeline's `run_sharded` halts on shard failure), but if a `STAGES="fit ensemble"` resume is run after a partial-shard recovery, ensemble silently trains on a smaller set than per-reader fits suggest. Add a print of `len(per_dir[i]) - len(common)` per reader for safety.

Plus 4 MINOR hardening notes (logged, not blocking):
- `train_pruner.py:137` partial-batch gradient under-scaled by `n_partial/docs_per_step` (3 partial steps of 1128 → negligible at this N_TRAIN, more visible in pilot)
- `evaluate.py:143 + pruner.py:164` `PrunerScorer` raises raw `FileNotFoundError` on missing pruner dir — no clean message if `STAGES="select answer report"` is run after a bypassed train failure
- `generate_labels.py:63-65` `measure_config.json` doesn't record `num_shards`; safe today because mask generation is shard-independent (per-doc seed), but the guard is incomplete
- `train_pruner.py:123` `windows_cache` ~1.4 GB CPU RAM at N_TRAIN=3000 (grows with train size; fine on H100 box)

Verified-clean items from the audit (the user's specific concerns):
1. ✅ β sign consistency (ridge model m_c, listnet pulls `softmax(scores)` toward `softmax(β/τ)`)
2. ✅ Dev metric keys consistent (ndcg@3_beta for beta/ensemble, gold_recall@25% for span)
3. ✅ Best-epoch save contract (single dir, overwrites on better dev; NO epoch_* subdirs)
4. ✅ NaN fallback only fires when no best ever saved → writes final to *empty* dir, does not overwrite saved best
5. ✅ dtype/device / grad-checkpoint + autocast canonical PyTorch (use_reentrant=False)
6. ✅ `resolve_device('cuda')` correct under per-process `CUDA_VISIBLE_DEVICES`
7. ✅ `ChunkPruner.save_pretrained` signature matches usage; save/load round-trips

## 7. Next Steps Suggested

1. **Verify the 5 models on HF** after pod release: `https://huggingface.co/thanthienhai/ttcompress-pilot-pruner-beta-primary/...`. Re-run the `upload` stage locally if the eval dataset didn't make it.

2. **Decide on the two MAJOR audit findings**: the listnet/MSE reduction imbalance is a paper-fairness concern — either normalize in code and rerun, or document the convention used.

3. **Scale up to full run** when ready: `./env` has the pilot config (N_TRAIN=60). The `.env.example` has the full config (N_TRAIN=3000). Pilot completed in ~1h07m for the substantive stages; full run estimated 5-10× longer, dominated by the `labels` stage at scale. Reuse the existing HF cache (no re-download).

4. **Appendix figures**: `report.json` carries 176 `by_depth` entries for position-robustness plots and 880 paired comparisons for upgrade-retention plots.

## 8. Reproduction

```bash
# Cluster side (already done):
kubectl --context stg -n llms-dev apply -f k8s/train-pod.yaml
# (uses the k8s/dev-pod.yaml template pinned to dn24z where HPS NFS lives)

# Local re-run of select+answer+report (assumes labels/train on HPS):
cd /home/anhm5/code/fci/papers/ttcompress
STAGES="select answer report" ./run_pipeline.sh
```
