# Báo cáo Smoke Test — `ttcompress`

**Ngày chạy**: 28/09/2026  
**Mục đích**: Kiểm tra pipeline `run_pipeline.sh` chạy end-to-end được trên cluster stg, trước khi chạy pilot/full.  
**Kết quả**: ✅ **PASSED** — tất cả 9 stage chạy xong trong 47 phút trên 2× H100.

---

## 1. Cấu hình smoke test

| Param | Giá trị | Ghi chú |
|---|---|---|
| Pod | 2× H100-80GB (NFS `/mnt/hps` 18TB) | Sẵn cho 4× chạy pilot |
| Image | `dynamo-lmcache-runtime:1.4.2-cu130-vllm0.28.1rc1.dev380-...` | vLLM 0.28, transformers 5.16, torch 2.13, CUDA 13.0 |
| Code commit | `80975e9` ("Per-run N-restricted fits, seed robustness…") | Clone mới vào `/mnt/hps/anhm-paper/ttcompress_new` |
| `SMOKE=1` | ON | Repo có sẵn mode này => auto-thu nhỏ N |
| `N_TRAIN` | **8 docs** | (full config = 3000) |
| `N_DEV` | **4 docs** | (full config = 300) |
| `N_TEST` | **4 docs** | (full config = 500) |
| `ORACLE_N` | **4 docs** | (full config = 100) |
| `EPOCHS` | **1** | (full config = 3) |
| `EXTRA_SEEDS` | "1" (chỉ 1 seed phụ) | full = "1 2" |
| `EXTRA_ARMS` | (đã tắt) | skip XProvence/LLMLingua-2 cho smoke, tránh cài spacy/llmlingua |
| `RATIOS` | "4,8" | tokens/chunk = 4 và 8 |
| `LABEL_READERS` | Qwen3-8B, Qwen3-1.7B, Llama-SEA-LION-v3-8B | primary = Qwen3-8B |
| `EVAL_READERS` | trên + **Qwen3-32B** (held-out) | Qwen3-32B chạy TP=2 trên cả 2 GPU |
| `TRAIN_SOURCES` | uit_viquad, vimqa, hotpotqa | |
| `EVAL_SOURCES` | + xquad_vi, 2wiki | cross-dataset transfer |
| Upload stage | (skip — SMOKE=1 tự bỏ) | không đẩy HF Hub |

---

## 2. Kết quả từng stage (9 stage × 2 GPU)

| Stage | Trạng thái | Thời lượng | Chi tiết |
|---|---|---|---|
| `preflight` | ✅ | ~10s | 33/33 pytest unit tests pass; detect 2 GPU; verify vllm/torch/transformers |
| `prefetch` | ~27 phút | download 7 model + 12 dataset về cache |
| `labels` (Stage A) | ✅ | ~1 phút | 3 reader × 3 source × 2 split + oracle_beta trên 5 source = ~24 job chạy shard 2 GPU |
| `fit` | ✅ | <1 phút | Ridge fits cho 32 (reader, target, source, split) + 5 oracle_beta |
| `ensemble` | ✅ | <1 phút | 6 kết hợp |
| `train` (Stage B) | ✅ | ~1 phút | 7 pruner (1 epoch): beta_primary, beta_ensemble, span, logprob_primary, posadj, +2 seed-1 |
| `select` | ✅ | <1 phút | 16 arm × 5 source × 2 ratio / 2 shard GPU |
| `answer` | ✅ | ~8 phút | 4 reader (Qwen3-32B ở TP=2 wire cả 2 GPU); mỗi reader question-answering trên selection từ `select` |
| `report` | ✅ | <5s | xuất `report.md` + LaTeX/CSV |
| **Tổng** | ✅ | **47 phút** | bắt đầu 11:03Z → xong 11:50Z |

---

## 3. Số liệu huấn luyện

### 3.1. Số doc thực tế dùng

Stage A sample (nested doc): 1 article chia thành nhiều chunk, nên `n_train`/`n_dev` lớn hơn `N_TRAIN`/`N_DEV`.

| Tập | N_HARD (config) | N_docs thực tế (sau khi tách chunk) |
|---|---|---|
| Train | 8 | 23–24 (3 source × ~8 article) |
| Dev | 4 | 12 |
| Test (eval) | 4 | 4/source × 5 source = 20 |
| Oracle_beta (test) | 4 | 4/source × 5 = 20 |

### 3.2. Chất lượng labels (Stage A ridge fit)

Trên 2 readers × 6 source/split (lấy 10 mẫu):

| Reader | Source_split | n_docs | CV R² | α | gold_recall@G |
|---|---|---|---|---|---|
| Qwen3-1.7B | uit_viquad_train | 8 | 0.87 (target=logprob) | 1.0 | 1.00 |
| Qwen3-1.7B | uit_viquad_dev | 4 | 0.74 | 1.0 | 1.00 |
| Qwen3-1.7B | vimqa_train | 8 | 0.47 | 1.0 | 0.83 |
| Qwen3-1.7B | hotpotqa_train | 8 | 0.27 | 3.0 | 0.69 |
| Qwen3-8B | uit_viquad_train | 8 | 0.47 (f1) | 1.0 | 1.00 |
| ... còn 26 fit khác tương tự | | | | | |

**Nhận xét**: 
- CV R² 0.19–0.87 vì n=4–8 doc => surrogate còn nhiều nhiễu (kỳ vọng pilot N=300 sẽ lên > 0.6).
- gold_recall@G = 1.0 trên nhiều source => mask strategy còn bảo toàn gold chunk.

### 3.3. Kết quả train 7 pruner (1 epoch mỗi cái)

| Pruner | n_train | n_dev | Epoch 1 loss | listnet | mse | dev gold_recall@25% | dev ndcg@3_beta | dev spearman_beta |
|---|---|---|---|---|---|---|---|---|
| `beta_primary` (seed 0) | 23 | 12 | 3.255 | 2.711 | 1.086 | 0.500 | 0.498 | 0.090 |
| `beta_ensemble` (seed 0) | 24 | 12 | 3.270 | 2.716 | 1.106 | 0.625 | 0.550 | 0.028 |
| `span` (label = gold) | 24 | 12 | (no listnet) | – | – | – | – | – |
| `logprob_primary` | 24 | 12 | 3.264 | 2.713 | 1.101 | 0.583 | 0.708 | 0.114 |
| `beta_primary_posadj` | 23 | 12 | 3.253 | 2.712 | 1.083 | 0.500 | 0.502 | 0.099 |
| `beta_primary_s1` (seed 1) | 23 | 12 | 3.431 | 2.755 | 1.352 | 0.250 | 0.378 | 0.035 |
| `beta_ensemble_s1` (seed 1) | 24 | 12 | 3.413 | 2.730 | 1.366 | 0.250 | 0.309 | -0.011 |

**Nhận xét**:
- Các seed 0 (primary/ensemble/posadj/logprob) có loss ~3.25, seed 1 (s1) cao hơn ~3.43 => như mong đợi với n=23-24 doc, 1 epoch nhiệt định通车 không ổn định.
- `logprob_primary` có spearman_beta cao nhất (0.114), `beta_ensemble` có gold_recall@25% cao nhất (0.625).
- Với 1 epoch / 8 doc, kết quả không có ý nghĩa thống kê — chỉ là "pipeline chạy được".

---

## 4. Verdict các giả thuyết (H1-H4)

Smoke test **không** nhằm kiểm chứng giả thuyết (n quá nhỏ), chỉ kiểm tra cơ chế code. Bảng tóm tắt:

| Family | Claim | Số test | Pass @ α=0.05 | Pass mọi seed cùng ph |
|---|---|---|---|---|
| **H1** | `ours_beta` > cheap baseline (bm25, embed, reranker, lead, random), mọi source/ratio | 50 | 0/50 | 0/50 |
| **H1-oracle** | `ours_beta` trong ±0.05 F1 của `oracle_beta` (non-inferiority) | 10 | **4/10** ✓ | 2/10 |
| **H2a** | multi-hop: `ours_beta` > `span_sup` và > `oracle_span` | 12 | 0/12 | 0/12 |
| **H2b** | single-hop: `ours_beta` tương đương `span_sup` ±0.02 (TOST) | 4 | **3/4** ✓ | 0/4 |
| **H3b** | retention(`ours_ens`) > retention(`ours_beta`) | 18 | 0/18 | – |
| **H3b-heldout** | như H3b, cặp có held-out reader | 16 | 0/16 | – |
| **H4** | source VN: ours_beta + ours_ens > XProvence + LLMLingua-2 | 0 | (chưa chạy, do đã tắt EXTRA_ARMS) | – |

- H1/H2a/H3b fail toàn bộ do n=4 test, CI quá rộng (vd CI [-0.733, +0.000]).
- H1-oracle và H2b pass 4/10 và 3/4 — chủ yếu ở các cell mà cả `ours_beta` và oracle cùng bằng 0 (cell toàn 0 F1, không cạnh tranh thực), p-value trivial. **Không suy ra được kết luận khoa học**.
- "seeds ✗" column cho thấy 1-epoch training trên 8 doc rất không ổn định. Cần pilot N=300 để có ý nghĩa.

---

## 5. Cost (RQ1) — time/doc của Stage A và selection

### 5.1. Stage A labels (GPU heavy)

| Reader | Source | Tập | s/doc | Tổng h (full 8 doc) |
|---|---|---|---|---|
| Qwen3-1.7B | uit_viquad | train | 4.6 | 0.08 |
| Qwen3-8B | uit_viquad | train | 12.3 | 0.21 |
| Llama-SEA-LION-v3-8B | uit_viquad | train | 11.3 | 0.19 |
| Qwen3-1.7B | hotpotqa | train | 0.9 | 0.01 |
| Qwen3-8B | hotpotqa | train | 2.1 | 0.04 |
| ... (21 ô khác tương tự) | | | | |
| Qwen3-8B | 2wiki | test (oracle) | 1.8 | 0.00 |

Mỗi doc gọi reader **65 lần** (mask/ablate từng chunk để lấy outcome). bge-reranker (backbone) ~50ms/doc.

### 5.2. Selection (CPU/GPU pooled) — ms/doc

| Source | Arm | ms/doc |
|---|---|---|
| 2wiki | reranker | 51 |
| 2wiki | embed | 64 |
| 2wiki | ours_beta | 45 |
| 2wiki | ours_ens | 12 |
| 2wiki | bm25/lead/random/oracle | 0-1 |

=> `ours_beta` (45 ms) còn nhanh hơn `reranker` zero-shot (51 ms) — đúng kỳ vọng vì cùng backbone nhưng pruner không rerank toàn corpus.

**Compute dự kiến cho pilot** (N_TRAIN=300, N_DEV=30, N_TEST=50):
- Stage A: ~5-7 h trên 4× H100 (gấp ~40× smoke)
- Train (3 epoch × 7 pruner): ~30 phút
- Stage answer (4 reader): ~40 phút
- ⇒ Pilot ~6-8 h ÷ 4 GPU. **Full** (3000/300/500): ~30-40 h ÷ 4 H100.

---

## 6. Hồ sơ tài nguyên đã dùng

| Hạng mục | Giá trị | Vị trí |
|---|---|---|
| Cluster / namespace | `stg` / `llms-dev` | kubectl context stg |
| Node | `fke-ncp-modas-stg-q-c8ifaxe-...-wknqc` (7 H100 allocatable) | |
| Job name | `ttcompress-smoke-1790593393` | đã Completed |
| Image | `hub.fci.vn/ncp-modas/containers/dynamo-lmcache-runtime:1.4.2-cu130-vllm0.28.1rc1.dev380-...` | chạy với `securityContext.runAsUser: 0` (NFS yêu cầu) |
| Code | `/mnt/hps/anhm-paper/ttcompress_new/` | clone commit `80975e9` |
| Labels (shared, được tái dùng) | `/mnt/hps/anhm-paper/ttcompress/runs/labels/` (42 MB) | nested theo N |
| HF cache | `/mnt/hps/anhm-paper/hf_cache/hub/` | đã có sẵn 7 model |
| Python deps phụ | `/mnt/hps/anhm-paper/ttcompress_new/.pylibs/` | pandas, scipy, datasets… (dùng cho các lần chạy sau) |
| Smoke artifacts | `/mnt/hps/anhm-paper/ttcompress_new/runs/smoke/` (15 GB) | gồm models/15GB, results/6MB |

---

## 7. Vấn đề đã gặp + cách giải quyết

| Vấn đề | Nguyên nhân | Fix |
|---|---|---|
| Pod pending | nodeSelector trỏ tới `dn24z` (full GPU) | chuyển sang `wknqc` (5 GPU rảnh) |
| `No module named pip` | image venv `/opt/venv` thiếu pip | install python deps qua `/usr/bin/pip3 --target` vào HPS dir `.pylibs/` |
| PyPI timeout | pypi.org chậm trong cluster | đổi `--index-url https://hub.fci.vn/repository/pypi/simple/` |
| `/opt/venv` read-only | venv là built-in | thực thi `runAsUser: 0` để ghi NFS |
| `./run_pipeline.sh: Permission denied` | file clone không có +x | dùng `bash run_pipeline.sh` |
| `transformers` test fail vì cache rỗng | HF cache trống | prefetch trước `hf-internal-testing/tiny-random-*` |
| `EXTRA_ARMS` fail pip_pinned llmlingua | venv image không có pip | set `EXTRA_ARMS=` để skip (smoke không cần baselines) |

---

## 8. Đánh giá khả thi

### ✅ Có thể đi tiếp (Pass đủ điều kiện):

1. **Code latest chạy được trên cluster**: commit mới (`80975e9`) cho smoke mode, oracle_beta, seed robustness, distractor ablation, paper tables tất cả pass.
2. **vLLM 0.28 + transformers 5.16 + torch 2.13 tương thích** với repo (image của FCI không phải môi trường chuẩn README, nhưng thêm `.pylibs` là chạy được).
3. **TP=2 cho Qwen3-32B trên 2 GPU** hoạt động đúng.
4. **Label sharing mechanism work**: smoke labels ghi vào `/mnt/hps/anhm-paper/ttcompress/runs/labels/raw/` — pilot sau đó sẽ kế thừa (nested theo N) và không đo lại 8 doc đầu.
5. **HF cache warm** — 7 model (Qwen3-8B/1.7B/32B, Llama-SEA-LION, bge-reranker, bge-m3, tiny-random×2) đãcludedownload xong, các lần sau sẽ không tốn 27 phút prefetch.

### ⚠ Còn phải xử lý trước khi pilot:

| Việc | Kỹ thuật | Thời gian ước tính |
|---|---|---|
| **Enable `EXTRA_ARMS`** cho XProvence + LLMLingua-2 | pre-install `spacy + xx_sent_ud_sm` và `llmlingua` vào image (hoặc vào `.pylibs`) — image hiện tại không có pip trong venv | ~15 phút setup, save reusable |
| **Tăng GPU lên 4× H100** cho pilot | sửa `NUM_GPUS=4` trong `.env` và `nvidia.com/gpu: 4` trong yaml; node `wknqc` có 7 H100 free | ok ngay |
| **Backup smoke model** 15GB | skip qua nếu pilot sẽ train lại với N=300 | – |
| **Xóa smoke run** hoặc giữ lại để so sánh | HPS còn 11 TB rảnh, có thể giữ | – |
| **Xem lại test_selection_metrics::test_reranker_arm_scores_every_chunk** | đã pass sau khi prefetch tiny model — khỏi lo | – |

### 📅 Đề xuất bước tiếp theo

1. **Cài đặt baselines** (`spacy`, `llmlingua`) vào `.pylibs` trên HPS => khoảng 5 phút.
2. **Tạo Job manifest cho pilot** (`N_TRAIN=300 N_DEV=30 N_TEST=50 EXTRA_SEEDS="1 2" NUM_GPUS=4 RUN_ROOT=.../runs/pilot HF_RUN_NAME=pilot`) — file mẫu đã có tại `/tmp/opencode/ttcompress-smoke.yaml`, chỉ cần sửa 4 tham số.
3. **Estimate pilot**: 6–8 giờ trên 4× H100 (1/10 full). Nếu kết quả tốt → chạy full 30-40 giờ.

---

## 9. File artifacts (đường dẫn đầy đủ)

```
/mnt/hps/anhm-paper/ttcompress_new/
├── .env                          # .env.example đã sửa (NUM_GPUS=2, EXTRA_ARMS=, HF_TOKEN=...)
├── .pylibs/                      # python deps dùng chung
├── runs/smoke/                   # kết quả smoke
│   ├── labels_fit/  5MB          # ridge summary.json (chất lượng labels)
│   ├── logs/        981KB        # log tất cả stage + từng shard
│   ├── models/      15GB         # 7 pruner checkpoint + train_log.json
│   └── results/eval_test/
│       ├── report.md             # bản chính: hypothesis verdicts + bảng paired
│       ├── report.json            # dữ liệu thô
│       └── paper/
│           ├── cost.tex          # bảng cost LaTeX
│           ├── hypotheses.tex
│           ├── main_results.tex
│           ├── retention.tex
│           ├── seeds.tex
│           ├── label_quality.tex
│           └── *.csv              # cells, paired, retention, seeds, by_depth
└── runs/labels/   (shared)       # stage A raw, tái dùng cho pilot/main
    └── raw/<reader>/<src>_<split>/

/mnt/hps/anhm-paper/hf_cache/hub/  # 7 model đã download
```

Local report copy: `/home/anhm5@fsoft.fpt.vn/code/fci/papers/ttcompress/reports/RUN_REPORT_2026-09-28_smoke_h100x2.md` (file này)

---

## 10. Kết luận

Smoke test **PASS**. Pipeline `ttcompress` commit `80975e9` chạy end-to-end được trên cluster stg với 2× H100, dùng image `dynamo-lmcache-runtime` của FCI + `.pylibs` để bổ sung deps thiếu. Tất cả 9 stage (preflight → report) hoàn thành trong 47 phút, đúng kỳ vọng cho ~8 doc / 1 epoch:

- Preflight 33/33 unit test pass
- Labels + fit + ensemble + train (7 pruner) ✅
- Select (16 arm) ✅
- Answer trên 4 reader (Qwen3-32B TP=2) ✅
- Report + LaTeX/CSV ✅

Hypothesis verdicts không có ý nghĩa thống kê ở N=4-8 (đúng mục đích thiết kế smoke là kiểm tra cơ chế, không kiểm chứng khoa học). Hàm CV R², gold_recall@G chạy đúng.

Khả thi để **chạy pilot** (N_TRAIN=300, N_DEV=30, N_TEST=50) trên 4× H100 ước tính 6-8 giờ. Khả thi để **chạy full** (3000/300/500) ước tính 30-40 giờ — cần thêm H100 thứ 3-4 (node `wknqc` còn 5 H100 free). Cần cài `spacy` + `llmlingua` vào `.pylibs` nếu muốn mở lại `EXTRA_ARMS` cho H4.
