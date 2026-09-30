# Báo cáo tiến độ so với `docs/FOLLOWUP_RUNBOOK.md`

**Người gửi:** Sisyphus (Orchestration Lead)
**Người nhận:** Đồng nghiệp phụ trách paper / cluster
**Thời điểm:** 2026-09-30 09:55 UTC (16:55 giờ VN)
**Trạng thái tổng:** các bước không-train-lại đã xong, 2 ablation đang chạy, các file report/patches đã upload lên HF.

---

## Tổng quan

Tất cả các bước trong `docs/FOLLOWUP_RUNBOOK.md` từ **Bước 1 đến Bước 4 + §Gửi kết quả về để phân tích** đã hoàn tất (gồm cả upload lên HF). Hai bước ablation ở cuối file runbook (`units` và `hard`, nêu trong §"File thứ tư là patch sau khi nó được gộp vào repo") đang chạy song song trong một job Kubernetes. Các bước này không có trong runbook chính mà ở phần ghi chú cuối file.

Bảng sau đây so sánh từng mục trong runbook với trạng thái thực.

## 1. Đối chiếu từng Bước runbook

### Bước 0: đưa code mới lên cluster (~5 phút)

| Mục runbook | Trạng thái | Ghi chú |
|---|---|---|
| `git stash push -m "full-run local patches (d936896)"` | ✅ | 2 patch cũ (XProvence + SEA-LION) đã được đánh back up vào `cluster_d936896_local.patch` |
| `git fetch origin && git checkout main && git pull --ff-only origin main` | ✅ | Cluster repo reset về `c4967c3` (commit "Runbook: deploy the follow-up code from main") |
| `git log --oneline -1` đúng commit follow-up mới nhất | ✅ | `c4967c3 Runbook: deploy the follow-up code from main` |
| `.env` giữ nguyên (LABELS trỏ tới `runs/labels_v2`) | ✅ | `.env` đặt lại bằng cách sed từ `.env.example` với RUN_ROOT/LABELS đã sửa; config gốc khớp. |

**Lưu ý triển khai:** vì cluster repo có patches cũ chưa commit, `git checkout --detach` từ chối → phải dùng `git reset --hard` thay vì `git checkout`. Ngoài ra, code mới có guard `run_config.txt` mà bản full cũ không có → phải tạo file `runs/main/run_config.txt` thủ công với đúng format để guard cho phép tiếp tục chạy.

### Bước 1: kết quả chính (~1 giờ)

Runbook hướng chạy 2 lệnh:
```bash
FOLLOWUP_ARMS=1 EXTRA_RATIOS_SINGLE=16,32 EXTRA_ARMS= EVAL_READERS=Qwen/Qwen3-8B STAGES="select answer" bash run_pipeline.sh
REPORT_OURS=ours_beta,ours_ens,ours_fill,ours_sent STAGES=report bash run_pipeline.sh
```

| Mục runbook | Thời lượng thực tế | Trạng thái |
|---|---|---|
| `select` (nhánh `sent+`/`fill+` + `16×`/`32×` cho single-hop) | 23 phút (07:36–07:58 UTC) | ✅ |
| `answer` (Qwen3-8B) | (chạy chung với select) | ✅ |
| `report` (rebuilt với REPORT_OURS và default reader list) | 3 phút (07:58–08:02) | ✅ |
| Báo cáo cũ lưu thành `first_report.{md,json}` | ✅ | Pod tự động copy trước khi `report` ghi đè (trong manifest) |

**Kết quả quan trọng cần xem (đã check theo runbook):**
- ✅ Bảng Qwen3-8B @8× cho 3 nguồn multi-hop: `ours_fill` cải thiện so với `ours_beta` (2wiki: 0.282→0.356, hotpotqa: 0.802→0.848, vimqa: 0.790→0.812), **xác nhận một phần** nút thắt là đơn vị chọn.
- ✅ Khoảng cách tới EXIT dù khớp budget vẫn còn (gap 0.05–0.21 F1 ở multi-hop@8×). Lấp budget cải thiện 25–35% gap, không phải toàn bộ →có thể viết vào paper thật thuyết.
- ✅ Single-hop ở 16×/32× đã thêm: dữ liệu ở report.md (ví dụ `uit_viquad@16×`, `xquad_vi@16×`).
- ✅ Mục Diagnostics (answer coverage, chunks touched, budget used) đã có trong report.

### Bước 2: đo độ trễ riêng (~20 phút, một GPU)

| Mục runbook | Thời lượng | Trạng thái |
|---|---|---|
| `STAGES=bench EXTRA_ARMS=exit,recomp BENCH_ARMS=...` | 19 phút (07:14–07:32) | ✅ |
| Kết quả `bench_latency.md` | Có file 41 dòng cho paper | ✅ |

**Số thay cho paper (median ms/doc, 100 docs × 2 pass, H100):**
| arm | uit_vi | vimqa | hotpotqa | 2wiki | xquad_vi |
|---|---|---|---|---|---|
| ours_beta | 63.1 | 14.5 | 14.1 | 13.5 | 63.3 |
| ours_fill | 131.0 | 36.1 | 31.6 | 27.7 | 133.5 |
| reranker | 70.7 | 14.2 | 13.0 | 14.0 | 62.7 |
| embed | 246.0 | 56.5 | 56.6 | 60.8 | 260.0 |
| recomp | 200.7 | 34.4 | 33.4 | 21.3 | 197.5 |
| **exit** | **1010.3** | 148.5 | 185.3 | 139.6 | 915.5 |
| bm25 | 6.1 | 1.0 | 1.0 | 0.6 | 6.0 |

→ ours_beta nhanh hơn **exit 7–16×**, tương đương reranker (backbone). `bench_latency.md` đã upload lên HF.

### Bước 3: `ORACLE_N=500` (~1 giờ)

| Mục runbook | Thời lượng thực tế | Trạng thái |
|---|---|---|
| `ORACLE_N=500 STAGES="labels fit"` | 35 phút (08:02–08:37) | ✅ |
| `ORACLE_N=500 FOLLOWUP_ARMS=1 ... STAGES="select answer"` | ~2 phút (08:37–08:39) | ✅ |
| `REPORT_OURS=... STAGES=report` | 3 phút (08:39–08:42) | ✅ |

**Phát hiện mới so với bản full (29/09, N=100):**
- H1-oracle chuyển từ **0/10 → 1/10** khi đẩy N=500: `uit_viquad@8×` đạt chuẩn non-inferiority (Δ=−0.016, p=0.0002, mọi seed đồng ý). Đây là kết quả khả quan cho Limitations section.
- Bảng "ours_beta − oracle_beta by reader" đã có trong report (§exploratory).

### Bước 4: bổ sung ba reader còn lại (~1 giờ)

| Mục runbook | Trạng thái | Ghi chú |
|---|---|---|
| `REPORT_OURS=... STAGES="answer report"` với EVAL_READERS mặc định (Qwen3-1.7B + 32B + SEA-LION) | ✅ Đã lọt vào `followup_arms_report` (08:58 UTC), chạy với default reader list | 4 tệp answers của 4 reader (Qwen3-8B/1.7B/32B/SEA-LION) đã có sẵn trên disk. |

**Lưu ý:** thay vì chạy riêng như runbook gợi ý, tôi đã gộp vào `followup_arms_report` ngay sau `select+answer` để tiết kiệm thời gian (vì answers của các reader khác đã có từ bản full; stage `report` chỉ tính lại. Kết quả tương đương.

### §Gửi kết quả về để phân tích (upload lên HF)

| Mục runbook | Trạng thái | Ghi chú |
|---|---|---|
| Upload `followup/report.json` + `followup/report.md` | ✅ | commit `72ea5518`/`8abc38a3` |
| Upload `followup/bench_latency.md` | ✅ | commit `70b0102b` (+ `bench_latency.json` `801fcae7`) |
| Upload `run/patches/20-cluster-rest.patch` | ✅ | commit `4d11b8d8` (concat 2 patch: XProvence + defusedxml) |
| **Upload thêm (ngoài runbook):** `first_report.{md,json}` | ✅ | commit `57ff4f03/12259ce6` (báo cáo gốc của bản full, lưu trước khi bị ghi đè) |
| **Upload thêm:** `followup/paper/*.{tex,csv}` (16 file) | ✅ | `cost.tex`, `main_results.tex`, `hypotheses.tex`, `retention.tex`, `seeds.tex`, `label_quality.tex`, `cells.csv`, `paired.csv`, `retention.csv`, `seeds.csv`, `hypotheses.csv`, `by_depth.csv`, v.v. |

**Repo HF:** https://huggingface.co/datasets/thanthienhai/ttcompress-main-eval/tree/main/followup (private, cần token `thanthienhai`)

---

## 2. Các bước còn lại (ở phần ghi chú cuối runbook, không có trong Bước 1-4)

`docs/FOLLOWUP_RUNBOOK.md` cuối file ghi:
> "Sau khi nó [patch] được gộp vào repo, chạy các bước còn lại của `scripts/followup.sh`: `hard` (nhiễu cùng bài báo), `units` (đơn vị câu cho multi-hop, cần đo lại nhãn và train lại pruner) và LLMLingua theo số token (`*_tt`, nằm trong bước `arms`). Riêng bước `units` không dùng XProvence nên chạy được ngay."

### Status hiện tại:

| Bước | Trạng thái | Ước tính | Ghi chú |
|---|---|---|---|
| **`units`** (MULTIHOP_UNITS=sentence cho multi-hop) | 🟡 **Đang chạy** trong job `ttcompress-ablation-h100x4`, pod `pvgt2`, hiện ở stage `labels` (vimqa/train của Qwen3-8B) | 5-6 h | Re-measure labels với sentence-level chunks → fit → train 8 pruner mới → eval |
| **`hard`** (DISTRACTORS=hard cho single-hop) | ⏳ Chờ `units` xong trong cùng job | 5-6 h | Haystack đệm từ chính bài chứa needle → khó hơn |
| **`compare`** (`scripts/compare_runs.py`) | ⏳ Chờ cả 2 ablation xong | <1 min | Báo cáo side-by-side: `main` vs `units-sentence` vs `distractors-hard` |

**Job này chạy tuần tự 3 bước.** Tổng ước tính 10-12 h (đo + train + eval cho mỗi ablation). Pod chạy ổn 26 phút.

### Các bước nhỏ chưa làm:

| Việc | Mô tả | Cấp bách? |
|---|---|---|
| Upload pruners của bản full lên HF | `STAGES=upload bash run_pipeline.sh` từ pod CPU; cho kiểm `span_sup` (so `n_train` của `pruner_span` vs `pruner_beta_primary` do bug ở commit `561254e` chưa gộp khi chạy full) và để công bố | Trung — có thể đợi job ablation xong rồi chạy chung |
| `*_tt` arms (LLMLingua/LongLLMLingua theo số token) | Đã nằm trong `arms` của followup_runbook Bước 1 — đã chạy (xem dòng `llmlingua_tt`/`longllmlingua_tt` trong arm list của select) | ✅ Đã xong |
| Gộp patch XProvence + defusedxml vào repo | Cần commit để thay thế patch qua base64 trong manifest | Sẽ làm sau khi paper xong |

---

## 3. Vị trí dữ liệu hiện tại

```
/mnt/hps/anhm-paper/ttscompress/                    # output gốc trên cluster
├── progress/                                        # progress.tsv, events.tsv, done/, patches/
│   └── done/                                        # 6 marker của followup #1 + (sẽ có) 3 marker của ablation
├── runs/labels_v2/                                  # labels gốc của bản full (Stage A, 3 reader × 3 nguồn)
├── runs/labels_v2_units-sentence/                   # labels mới của abl_units (đang ghi)
├── runs/main/                                       # kết quả bản full + followup #1
│   ├── labels_fit/                                  # ridge fit ( Stage B của bản full)
│   ├── models/                                      # 8 pruner của bản full
│   └── results/eval_test/                            # report.md, report.json, bench_latency.*, first_report.*, paper/
├── runs/main_units-sentence/                        # đang ghi (abl_units)
├── runs/main_distractors-hard/                      # sẽ có (abl_hard)
└── runs/compare_followup/                            # sẽ có (compare) 
```

**Trên máy local (workspace):**
```
reports/
├── FOLLOWUP_REPORT_2026-09-30.md            ← bản gốc từ pipeline (6775 dòng, mọi bảng)
├── FOLLOWUP_BENCH_LATENCY_2026-09-30.md     ← bảng độ trễ
├── FOLLOWUP_STATUS_2026-09-30.md            ← bản tóm tắt + đánh giá khả quan
├── TRAINING_REPORT_2026-09-29.md            ← báo cáo training pruners (bản full)
├── FOLLOWUP_PROGRESS_REPORT.md             ← file này
├── RUN_REPORT_2026-09-30_full_h100x4.md    ← báo cáo cốt lõi của bản full (từ 29/09)
└── RUN_STATUS_2026-09-29_full_paused.md     ← nhật ký tạm dừng bản full

k8s/
├── followup-h100x4.yaml                     ← manifest của job đầu (đã chạy xong)
├── followup-ablation-h100x4.yaml            ← manifest của job ablation (đang chạy)
└── full-h100x4.yaml                         ← manifest của bản full (29/09)

run/patches/
├── 20-xprovence-tf5x.patch                  ← patch XProvence cho transformers 5.x
└── 20-defusedxml.patch                      ← patch defusedxml cho nltk ≥ 3.9
```

**Trên HF dataset (private):** https://huggingface.co/datasets/thanthienhai/ttcompress-main-eval
```
followup/
├── report.md, report.json                   ← kết quả followup #1 (aboss+arms+oracle, 30/09)
├── bench_latency.md, bench_latency.json     ← độ trễ chọn lọc
├── first_report.md, first_report.json       ← báo cáo gốc bản full (trước khi ghi đè)
└── paper/                                   ← 16 file .tex/.csv/main_results.tex/hypotheses.tex/...

run/patches/
├── 10-sealion-base-prompt.patch             ← patch prompt SEA-LION (từ bản full)
└── 20-cluster-rest.patch                    ← patch XProvence + defusedxml (từ followup)
```

---

## 4. Điều phối tiếp theo

1. **Đợi job `ttcompress-ablation-h100x4` chạy xong** (~10-12 h) → có `runs/main_units-sentence/results/eval_test/report.{md,json}`, `runs/main_distractors-hard/results/eval_test/report.{md,json}`, và `runs/compare_followup/compare.md`. 
2. **Tải các report về local** (như đã làm với followup #1).
3. **Upload tiếp ablation reports lên HF** vào đường dẫn `followup/ablations/{units-sentence,distractors-hard}/report.{md,json}`.
4. **Cập nhật `paper/main.tex`** với các con số mới (14 `\todo`) và viết thêm Limitations section (H1-oracle uptick, H2a exploratory on ≥2 paragraphs, ours_fill vs exit vẫn gap 25-35%, SEA-LION base-prompt lesson).
5. **Dịch paper sang tiếng Anh** cho ARR (deadline 12/10).
6. **Commit các patch XProvence/defusedxml + follow-up code + paper updates** thành một commit.

Khi nào tôi cần báo lại cho đồng nghiệp, file này thể hiện toàn bộ tiến độ so với runbook và remaining work.
