# Đợt 3, phần 1: hướng dẫn vận hành

Hạn nộp paper: **12/10**. Đợt này không train lại gì. Nó dùng lại pruner và lựa chọn của bản full
(`runs/main`) để lấy thêm kết quả cho paper:

1. đo riêng độ trễ của `ours_sent` và `fuse_sent`;
2. ba reader còn lại trả lời các nhánh follow-up (`*_sent`, `*_fill`, 16×/32×) và các nhánh mới;
3. LLMLingua và LongLLMLingua với ngân sách theo số token (`*_tt`), để không còn bị cắt đuôi;
4. một reader khác họ Qwen, **`google/gemma-3-27b-it`**, chỉ trả lời, không sinh nhãn.

Mọi thứ chạy bằng một script: **`scripts/round3.sh`**.
**Tổng thời gian ước tính: khoảng 3.5–4.5 giờ trên 4×H100**, sai số khoảng ±30%.

---

## 1. Chuẩn bị (một lần, khoảng 10 phút)

### 1.1. Chấp nhận license Gemma

Đăng nhập Hugging Face bằng **đúng tài khoản của `HF_TOKEN`** trong `.env`. Mở
<https://huggingface.co/google/gemma-3-27b-it> và bấm chấp nhận license. License Gemma chấp nhận theo từng repo:
đã chấp nhận cho `gemma-2b-it` (dùng cho EXIT) **không** đủ.

### 1.2. Lấy code mới

Không `git pull` trong thư mục mà một job khác đang chạy từ đó, vì bash đọc script dần trong lúc chạy. Nếu có job
đang chạy, dùng một thư mục code riêng (cách 2, Bước 2.0 của `docs/FOLLOWUP_RUNBOOK.md`).

```bash
cd /mnt/hps/anhm-paper/ttcompress_new
git fetch origin && git checkout main && git pull --ff-only origin main
ls scripts/round3.sh scripts/check_reader.py     # phải có cả hai
grep -E '^(RUN_ROOT|LABELS|HF_TOKEN)=' .env       # RUN_ROOT phải trỏ tới bản full (.../runs/main)
```

### 1.3. Kiểm tra môi trường (khoảng 1 phút, không tốn GPU)

```bash
bash scripts/round3.sh check
```

Bước này dừng với dòng `!! ...` nếu:

| Thông báo | Cách xử lý |
|---|---|
| `missing .../pruner_beta_primary` hoặc `documents_shard0.jsonl` | `RUN_ROOT` trong `.env` sai; sửa thành thư mục của bản full |
| `Gemma 3 needs vllm>=0.8 and transformers>=4.50` | Image quá cũ: báo lại, **không** tự nâng cấp vLLM trong image đang dùng |
| `cannot read google/gemma-3-27b-it` | Chưa chấp nhận license (1.1), hoặc `HF_TOKEN` sai hay hết hạn |
| `warning: a process is using a GPU` | Không phải lỗi, nhưng phải dừng tiến trình đó trước bước `bench` |

---

## 2. Chạy

Chạy trong `tmux` (hoặc `nohup`), để mất kết nối SSH không làm dừng job:

```bash
tmux new -s round3
cd /mnt/hps/anhm-paper/ttcompress_new
bash scripts/round3.sh 2>&1 | tee -a round3_$(date +%m%d_%H%M).log
# tách khỏi tmux: Ctrl-b rồi d; vào lại: tmux attach -t round3
```

Nếu chạy bằng job k8s, lệnh của container là `bash scripts/round3.sh`, cần 4×H100 và mount HPS như các job trước.

Script chạy lần lượt các bước sau. Mỗi bước dùng cả 4 GPU, trừ `bench` (một GPU, phải chạy một mình).

| Bước | Làm gì | Ước tính |
|---|---|---|
| `check` | Như mục 1.3 | 1 phút |
| `bench` | Đo độ trễ chọn câu của `ours_sent`, `fuse_sent`, kèm mốc `ours_beta`, `reranker`; dừng ngay nếu có tiến trình khác trên GPU | 15–20 phút |
| `select` | Tạo các lựa chọn còn thiếu; **đồng thời tải Gemma (khoảng 55 GB) chạy nền** | 45–60 phút |
| `answer` | Qwen3-8B, Qwen3-1.7B, SEA-LION, Qwen3-32B trả lời các lựa chọn mới | 1–1.5 giờ |
| `checkreader` | Gemma trả lời 30 tài liệu mỗi nguồn với toàn bộ ngữ cảnh; **dừng script nếu kết quả bất thường** (mục 3) | 10 phút |
| `newreader` | Gemma trả lời mọi lựa chọn (khoảng 250k), TP2 × 2 tiến trình | 1.25–1.75 giờ |
| `report` | Dựng lại `report.json` / `report.md` với năm reader | 5 phút |

Thời gian thực của từng bước được ghi vào `$RUN_ROOT/logs/round3_timings.tsv`.

### Chạy lại sau lỗi

Mọi bước đều tiếp tục từ chỗ dừng: lựa chọn và câu trả lời đã có trên đĩa được bỏ qua. Sau khi sửa nguyên nhân, chạy
lại từ bước bị lỗi:

```bash
bash scripts/round3.sh answer checkreader newreader report     # ví dụ: lỗi ở answer
```

`bench` tự bỏ qua nếu `bench_latency_sent.json` đã có. Muốn đo lại thì xóa file đó trước.

---

## 3. Điểm dừng bắt buộc: `checkreader`

Đây là chỗ duy nhất cần người xem kết quả. Lần trước, SEA-LION trả lời rỗng 57% câu hỏi tiếng Việt, và lỗi chỉ lộ ra
sau nhiều giờ. Bước này tìm lỗi kiểu đó trong 10 phút.

Output có dạng:

```
== uit_viquad: n=30  empty=0%  full-context F1=0.712
   gold='...'  answer='...'
...
gate passed
```

- **`gate passed`**: script tự chạy tiếp `newreader` và `report`.
- **`GATE FAILED on ...`**: script dừng. Ngưỡng là: ở một nguồn nào đó, hơn 5% câu trả lời rỗng hoặc F1 dưới 0.30.
  **Không chạy tiếp.** Gửi lại `$RUN_ROOT/logs/round3_check_reader.log` và `round3_check_reader.json`.

Kể cả khi gate đạt, hãy xem các câu trả lời mẫu. Để đối chiếu, F1 toàn ngữ cảnh của Qwen3-32B là 0.73 / 0.77 / 0.69 /
0.72 / 0.63 (UIT-ViQuAD / XQuAD-vi / VIMQA / HotpotQA / 2Wiki). Gửi lại output nếu thấy một trong các dấu hiệu sau:
- câu trả lời là cả câu dài thay vì cụm từ ngắn;
- trả lời bằng tiếng Anh cho câu hỏi tiếng Việt;
- F1 thấp hơn Qwen3-32B quá khoảng 0.15 ở một nguồn.

Chỉ khi nhóm nghiên cứu đã xem output và đồng ý, mới chạy tiếp bất chấp gate:

```bash
FORCE_NEW_READER=1 bash scripts/round3.sh newreader report
```

---

## 4. Upload kết quả (pod CPU, sau bước `report`)

```bash
bash scripts/round3.sh upload
```

Lệnh này đưa lên dataset `thanthienhai/ttcompress-main-eval` (đổi bằng `HF_EVAL_REPO=...`):

| Đích trên HF | Nội dung |
|---|---|
| `followup3/` | `bench_latency_sent.json`, `report.json`, `report.md`, `check_reader.json`, `round3_timings.tsv` |
| `followup3/eval_test/` | `answers_*.jsonl`, `selections_*.jsonl` (cần cho kiểm định ghép cặp) |
| `followup2/replication/` | `report.*`, `answers_*.jsonl` của lần lặp lại đợt 2 (hiện chỉ có ở local) |
| `followup/units/` | `report.*`, `answers_*.jsonl` của ablation units (hiện trên HF mới có bản `.md`) |
| (theo `run_pipeline.sh`) | Pruner của bản full (`STAGES=upload`). Lần pilot, bước này mất khoảng 2.5 giờ vì đẩy rất nhiều file: **chỉ chạy trên pod CPU**, không chạy trên pod GPU |

Thư mục nào không tồn tại thì được bỏ qua kèm dòng `skip (missing)`. Đường dẫn của ablation units mặc định là
`${RUN_ROOT}_units-sentence`; nếu trên HPS khác thì báo lại.

---

## 5. Báo lại cho nhóm nghiên cứu

1. Nội dung `$RUN_ROOT/logs/round3_timings.tsv`.
2. Toàn bộ output của bước `checkreader`.
3. Các dòng `!!` hoặc traceback nếu có lỗi, kèm tên bước.
4. Xác nhận upload xong (danh sách đích ở mục 4).

**Lưu ý khi đọc `report.md`:** Gemma không sinh nhãn nên được tính là reader giữ lại, và họ H3-heldout trong report
mới có thêm các cặp với Gemma. Kết quả đã đăng ký của lần chạy chính nằm trong `first_report.json` (được giữ
nguyên). Không trích số H1–H3 từ report mới.

---

## Phụ lục: tùy chỉnh

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `NEW_READER` | `google/gemma-3-27b-it` | Reader mới |
| `NEW_READER_TP` | `2` | Số GPU mỗi tiến trình của reader mới |
| `NEW_READER_PATTERN` | `27b` | Regex để `run_pipeline.sh` cho reader mới chạy tensor parallel (tên model viết thường nên pattern mặc định `(3[0-9]\|7[0-9])B` không khớp) |
| `FORCE_NEW_READER` | `0` | `1`: chạy `newreader` dù `checkreader` không đạt |
| `HF_EVAL_REPO` | `thanthienhai/ttcompress-main-eval` | Dataset nhận kết quả |
| `EXTRA_RATIOS_SINGLE` | `16,32` | Tỉ lệ thêm cho dữ liệu một bước (phải giữ nguyên như đợt follow-up) |

Các lệnh tương đương mà script gọi (khi cần chạy tay từng phần), với `M=$RUN_ROOT/models`:

```bash
export FOLLOWUP_ARMS=1 ROUND2_ARMS=1 EXTRA_RATIOS_SINGLE=16,32
# bench
EXTRA_ARMS= STAGES=bench BENCH_OUT=$RUN_ROOT/results/eval_test/bench_latency_sent.json \
BENCH_ARMS="ours_beta=pruner:$M/pruner_beta_primary,ours_sent=sent+pruner:$M/pruner_beta_primary,fuse_sent=sent+rrf:pruner:$M/pruner_beta_primary|pruner:$M/pruner_span,reranker=reranker:BAAI/bge-reranker-v2-m3" \
  bash run_pipeline.sh
# select (+ tải Gemma song song)
EXTRA_ARMS=llmlingua,longllmlingua STAGES=select bash run_pipeline.sh
EVAL_READERS="google/gemma-3-27b-it" EXTRA_ARMS= STAGES=prefetch bash run_pipeline.sh
# answer (bốn reader cũ, theo .env)
STAGES=answer bash run_pipeline.sh
# checkreader
python scripts/check_reader.py --eval-dir $RUN_ROOT/results/eval_test --reader-model google/gemma-3-27b-it --tp 2 --n 30
# newreader
EVAL_READERS="google/gemma-3-27b-it" LARGE_READER_PATTERN='27b' TP_LARGE=2 STAGES=answer bash run_pipeline.sh
# report
EVAL_READERS="Qwen/Qwen3-8B Qwen/Qwen3-1.7B aisingapore/Llama-SEA-LION-v3-8B Qwen/Qwen3-32B google/gemma-3-27b-it" \
REPORT_OURS=ours_beta,ours_ens,ours_fill,ours_sent,fuse_sent STAGES=report bash run_pipeline.sh
```
