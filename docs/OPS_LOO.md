# Đợt LOO: baseline nhãn leave-one-out và Ridge K=11

**Mục đích.** Trả lời câu hỏi chắc chắn sẽ có từ reviewer: "mặt nạ ngẫu nhiên + Ridge (~65 lần gọi reader mỗi tài
liệu) có đáng hơn nhãn leave-one-out (C+1 lần gọi) không?". Các so sánh đã cố định trước trong
`docs/prereg_loo.json` (L1, L1p, L2a, L2b). Không sửa file đó trong lúc chạy.

**Hạn nộp ARR:** 12/10. **Ước lượng:** ~6–7.5 giờ trên 4×H100 (pilot 30 phút, nhãn 1.5–2 giờ, train ~4 giờ,
select + answer ~1 giờ).

| Nhánh | Nhãn | Bộ tỉa |
|---|---|---|
| `ours_loo`, `loo_sent` (+`_s1`, `_s2`) | Δ_i = F1(full) − F1(bỏ đoạn i), z trong tài liệu, loss như `ours_beta` | `pruner_loo` |
| `ours_k11`, `k11_sent` (+`_s1`, `_s2`) | Ridge trên 11 mặt nạ rút từ nhãn ngẫu nhiên **đã có** (không gọi reader) | `pruner_k11` |
| `loocomp_bin`, `loocomp_bin_sent` (+ seed) | nhị phân Δ_i > 0, BCE, kiểu LooComp/EnComp (`LOO_BIN=1`) | `pruner_loo_bin` |

Chỉ Qwen3-8B sinh nhãn. Mọi đầu ra mới nằm ở chỗ riêng: nhãn thô `$LABELS/raw_loo`, fit
`$R/labels_fit/loo/` và `$R/labels_fit/k11/`, pruner `$R/models/pruner_{loo,k11,loo_bin}*`. Với `LOO_ONLY=1`,
pipeline **không** chạy lại nhãn, fit hay train của lần chạy chính.

## 0. Chuẩn bị (~15 phút)

Dùng thư mục code riêng: không `git pull` trong thư mục đang có job chạy.

```bash
cd /mnt/hps/anhm-paper/ttcompress_new
git fetch origin && git checkout main && git pull --ff-only origin main
git log --oneline -1                      # phải là commit có LOO_ARMS (hash trong tin nhắn giao việc)
cp /mnt/hps/anhm-paper/ttscompress/.env .   # .env của bản full, nếu thư mục này chưa có
grep -E '^(RUN_ROOT|LABELS|N_TRAIN|N_DEV|EXTRA_SEEDS)=' .env
#   RUN_ROOT=/mnt/hps/anhm-paper/ttscompress/runs/main
#   LABELS=/mnt/hps/anhm-paper/ttscompress/runs/labels_v2
#   không đặt N_TRAIN/N_DEV/N_TEST khác bản full (run_config.txt sẽ chặn)
export R=/mnt/hps/anhm-paper/ttscompress/runs/main
export LABELS=/mnt/hps/anhm-paper/ttscompress/runs/labels_v2
python -m pytest -q tests --basetemp=/tmp/pt_loo      # 122 passed (~10 phút; có thể chạy song song với Bước 1)
```

Nếu pytest báo `PermissionError` ở thư mục tạm: giữ `--basetemp` như trên.

## 1. Pilot (~20–30 phút)

Mục đích: đo tốc độ thật và tỉ lệ tài liệu mang thông tin trước khi tốn GPU cho cả lần chạy. `--n` lồng nhau
theo hàm băm, nên 60 tài liệu pilot được lần chạy đầy đủ dùng lại.

```bash
LOO_ARMS=1 LOO_ONLY=1 N_TRAIN=60 N_DEV=10 STAGES=labels bash run_pipeline.sh   # KHÔNG đặt N_DEV=0 (= mọi tài liệu)
for s in uit_viquad vimqa hotpotqa; do
  python generate_labels.py fit --raw-dir "$LABELS/raw_loo/Qwen--Qwen3-8B/${s}_train" --estimator loo \
    --out-dir /tmp/loo_pilot/$s --n 60
  python -c "import json;d=json.load(open('/tmp/loo_pilot/$s/summary.json'));print('$s informative',d['n_informative'],'/',d['n_docs'])"
done
find $R/logs -name "*labels_loo*" | xargs grep -h "running mean" | tail -5
```

**Điểm dừng — gửi kết quả pilot rồi mới chạy Bước 2:**

| Kiểm tra | Bình thường | Cần báo ngay |
|---|---|---|
| Tài liệu mang thông tin, nhiều bước (vimqa, hotpotqa) | ≥ 50% (nhãn ngẫu nhiên: 90–99%; LOO dự kiến thấp hơn) | < 30% |
| Tài liệu mang thông tin, UIT-ViQuAD | bất kỳ, chỉ ghi lại | |
| Thời gian, UIT-ViQuAD | ≤ ~7 s/tài liệu/GPU | > 10 s/tài liệu |
| Thời gian, nhiều bước | ≤ ~0.4 s/tài liệu/GPU | > 1 s/tài liệu |
| Lỗi trong log | không có | `Traceback`, câu trả lời rỗng hàng loạt |

## 2. Nhãn, fit, train (~4.5–6 giờ)

```bash
LOO_ARMS=1 LOO_BIN=1 LOO_ONLY=1 STAGES="labels fit train" bash run_pipeline.sh 2>&1 | tee $R/logs/loo_main.log
```

Dòng config đầu log phải có `LOO_ARMS=1 LOO_BIN=1 LOO_ONLY=1`. Pipeline tự bỏ qua phần đã xong, nên nếu bị
ngắt thì chạy lại đúng lệnh này.

**Kiểm tra sau khi xong:**

```bash
# nhãn: 3000 train + 300 dev mỗi nguồn
for s in uit_viquad vimqa hotpotqa; do for sp in train dev; do
  echo $s/$sp $(ls $LABELS/raw_loo/Qwen--Qwen3-8B/${s}_$sp/*.json | grep -vc config); done; done
# fit: tỉ lệ mang thông tin (LOO) và alpha (K=11)
for d in $R/labels_fit/loo/Qwen--Qwen3-8B/f1/*_train $R/labels_fit/k11/Qwen--Qwen3-8B/f1/*_train; do
  python -c "import json;d=json.load(open('$d/summary.json'));print('$d'.split('/f1/')[0].split('/')[-2], '$d'.split('/')[-1], d['n_informative'],'/',d['n_docs'],'alpha',d.get('alpha'))"; done
# train: 9 pruner, mỗi cái có train_log.json
ls $R/models/*/train_log.json | grep -cE '/pruner_(loo|k11|loo_bin)(_s[12])?/train_log.json$'   # 9 (6 nếu không LOO_BIN)
```

- Nhãn: ~1.5–2 giờ. UIT-ViQuAD chiếm phần lớn: ~31 lần gọi mỗi tài liệu, mỗi lần gần đủ ngữ cảnh.
- Fit: vài phút trên CPU. Nếu `alpha` của K=11 bằng 10 (biên lưới) ở mọi nguồn, ghi lại trong báo cáo; không cần
  chạy lại.
- Train: 9 pruner (3 loại × 3 seed), ~80 phút mỗi pruner trên 1 GPU, ~4 giờ trên 4 GPU. **Thiếu thời gian:** chạy
  lại không có `LOO_BIN=1` (6 pruner, ~2h40); `loocomp_bin` chỉ là khám phá, không nằm trong họ so sánh nào.

## 3. Tập test giữ lại (xếp hạng 501–2500), Qwen3-8B + Qwen3-32B (~45–60 phút) — ưu tiên cao nhất

Điều kiện: `ours_sent`, `ours_beta` và các seed của chúng đã có trong `$R/results/eval_replication` từ Đợt 2
(kiểm: `ls $R/results/eval_replication | grep -c ours_sent`).

```bash
A="ours_loo=pruner:@MODELS@/pruner_loo,loo_sent=sent+pruner:@MODELS@/pruner_loo"
A="$A,ours_loo_s1=pruner:@MODELS@/pruner_loo_s1,ours_loo_s2=pruner:@MODELS@/pruner_loo_s2"
A="$A,loo_sent_s1=sent+pruner:@MODELS@/pruner_loo_s1,loo_sent_s2=sent+pruner:@MODELS@/pruner_loo_s2"
A="$A,ours_k11=pruner:@MODELS@/pruner_k11,k11_sent=sent+pruner:@MODELS@/pruner_k11"
A="$A,ours_k11_s1=pruner:@MODELS@/pruner_k11_s1,ours_k11_s2=pruner:@MODELS@/pruner_k11_s2"
A="$A,loocomp_bin=pruner:@MODELS@/pruner_loo_bin,loocomp_bin_sent=sent+pruner:@MODELS@/pruner_loo_bin"
# bỏ dòng loocomp_bin nếu đã chạy không có LOO_BIN=1

EVAL_DIR=$R/results/eval_replication EVAL_SOURCES=vimqa,hotpotqa,2wiki EVAL_N=2000 EVAL_OFFSET=500 ORACLE_N=0 \
  EVAL_READERS="Qwen/Qwen3-8B Qwen/Qwen3-32B" EXTRA_ARMS= FOLLOWUP_ARMS=0 EXTRA_RATIOS_SINGLE= \
  ONLY_ARMS="$A" STAGES="select answer" bash run_pipeline.sh

# giữ báo cáo Đợt 2 trước khi report ghi đè
cp $R/results/eval_replication/report.json $R/results/eval_replication/report_round2.json
cp $R/results/eval_replication/report.md   $R/results/eval_replication/report_round2.md
EVAL_DIR=$R/results/eval_replication PREREG_FILE=docs/prereg_loo.json REPORT_OURS=ours_sent,ours_beta,ours_k11 \
  STAGES=report bash run_pipeline.sh
head -40 $R/results/eval_replication/report.md
```

`report.md` phải mở đầu bằng các họ L1, L1p, L2a, L2b. Gửi ngay 40 dòng đầu.

## 4. Tập phân tích (500 tài liệu/nguồn), Qwen3-8B (~20–30 phút; chỉ làm nếu còn thời gian)

```bash
cp $R/results/eval_test/report.json $R/results/eval_test/report_main.json
cp $R/results/eval_test/report.md   $R/results/eval_test/report_main.md
EVAL_READERS="Qwen/Qwen3-8B" EXTRA_ARMS= EXTRA_RATIOS_SINGLE= ONLY_ARMS="$A" \
  STAGES="select answer report" bash run_pipeline.sh
```

## 5. Upload

```bash
H=thanthienhai/ttcompress-main-eval
for f in report.json report.md; do
  huggingface-cli upload $H $R/results/eval_replication/$f followup3_loo/replication/$f --repo-type dataset
done
for m in pruner_loo pruner_k11 pruner_loo_bin; do
  huggingface-cli upload $H $R/models/$m/train_log.json followup3_loo/train_log_$m.json --repo-type dataset
done
for d in $R/labels_fit/loo/Qwen--Qwen3-8B/f1/* $R/labels_fit/k11/Qwen--Qwen3-8B/f1/*; do
  huggingface-cli upload $H $d/summary.json followup3_loo/fit_summaries/$(basename $(dirname $(dirname $(dirname $d))))_$(basename $d).json --repo-type dataset
done
huggingface-cli upload $H "$LABELS/raw_loo" followup3_loo/raw_loo --repo-type dataset   # nhãn thô, để phân tích offline
# nếu làm Bước 4:
huggingface-cli upload $H $R/results/eval_test/report.json followup3_loo/analysis/report.json --repo-type dataset
```

## 6. Báo lại

1. Kết quả pilot (bảng ở Bước 1), **trước** khi chạy Bước 2.
2. Thời gian thực của từng bước: nhãn (tách UIT-ViQuAD và nhiều bước), fit, train, select, answer. Bảng chi phí
   của `report` chỉ đọc `$LABELS/raw`, không tính `raw_loo`, nên paper cần số này.
3. 40 dòng đầu `report.md` của Bước 3, và link HF.

## Sự cố thường gặp

| Hiện tượng | Xử lý |
|---|---|
| `!! ... belongs to another configuration` | Đang đặt `N_TRAIN`/`N_DEV`/`N_TEST`/`LABEL_READERS` khác `.env`. Bỏ các biến đó (chỉ pilot Bước 1 mới đặt, và chỉ với `STAGES=labels`). |
| `measure_config.json was measured with ...` | Có tài liệu cũ trong `raw_loo` với cấu hình khác. Không xóa: báo lại. |
| `--estimator loo needs a dir measured with --mask-scheme loo` | Fit trỏ nhầm vào `raw` thay vì `raw_loo`. |
| `--label-source loo_bin needs leave-one-out fit records` | Train trỏ nhầm vào fit dir của Ridge. |
| `has N measured documents, --n asks for M` | Bước nhãn chưa xong: chạy lại lệnh Bước 2. |
| OOM khi train | Giữ nguyên; pipeline dùng `--grad-checkpointing`. Nếu vẫn OOM, báo lại tên pruner. |
