# Runbook: thí nghiệm follow-up trên cluster (sau bản full 2026-09-29)

Mục tiêu: lấy kết quả đánh giá sớm mà **không train lại**, chỉ dùng lại nhãn, pruner và selections của bản full
(`RUN_ROOT=/mnt/hps/anhm-paper/ttscompress/runs/main`, `LABELS=/mnt/hps/anhm-paper/ttscompress/runs/labels_v2`).
Lý do của từng thí nghiệm: `METHOD_SPEC.md` §9. Việc còn lại: `TODO.md` §0.

Mọi lệnh dưới đây **không nạp XProvence**, vì patch cho XProvence trên transformers 5.x chưa được gộp vào repo.
Mọi lệnh chạy lại được: selections, câu trả lời và nhãn đã có trên đĩa sẽ được bỏ qua.

## Bước 0: đưa code mới lên cluster (~5 phút)

Code follow-up đã nằm trên `main` (commit `893673b` và các commit sau) và đã push lên `origin`.

Trên cluster: dùng đúng thư mục code của bản full, để giữ nguyên `.env`, `.baseline_site` và `.llmlingua_site`.

```bash
cd /mnt/hps/anhm-paper/ttcompress_new
git diff > /mnt/hps/anhm-paper/cluster_d936896_local.patch   # lưu patch XProvence + defusedxml để gộp sau
git stash push -m "full-run local patches (d936896)"
git fetch origin && git checkout main && git pull --ff-only origin main
git log --oneline -1                                          # phải là commit follow-up mới nhất
```

`.env` phải giữ nguyên, vì nó trỏ `LABELS` tới `runs/labels_v2`. Thiếu nó, pipeline sẽ tìm nhãn ở `runs/main/labels`
và không thấy nhãn cũ.

## Bước 1: kết quả chính (~1 giờ)

Nhánh chấm điểm theo câu (`sent+`) và lấp ngân sách bằng câu (`fill+`) cho ours, span_sup và reranker, cộng thêm tỉ lệ
16× và 32× trên dữ liệu một bước. Chỉ reader chính trả lời.

```bash
FOLLOWUP_ARMS=1 EXTRA_RATIOS_SINGLE=16,32 EXTRA_ARMS= EVAL_READERS=Qwen/Qwen3-8B \
  STAGES="select answer" bash run_pipeline.sh
REPORT_OURS=ours_beta,ours_ens,ours_fill,ours_sent STAGES=report bash run_pipeline.sh
```

- `EXTRA_ARMS=` bỏ qua các baseline đã công bố (selections 4×/8× của chúng đã có), trong đó có XProvence.
- `report` chạy riêng với danh sách reader mặc định. Nếu gộp chung với `EVAL_READERS=Qwen/Qwen3-8B`, Qwen3-32B sẽ bị
  coi là reader sinh nhãn và các cặp của nó chuyển từ H3-heldout sang H3, làm sai họ kiểm định.
- Báo cáo cũ được giữ lại thành `first_report.json` / `first_report.md`.

Cần xem trong `runs/main/results/eval_test/report.md`:

- Bảng Qwen3-8B của VIMQA, HotpotQA, 2Wiki ở 8×: `ours_fill`, `ours_sent` so với `exit` và `ours_beta`, và các dòng
  `ours_fill vs exit` trong bảng ghép cặp. Nếu `ours_fill` đuổi kịp EXIT, giả thuyết "nút thắt là đơn vị chọn" được
  xác nhận.
- Mục Diagnostics: tỉ lệ ngữ cảnh còn chứa đáp án, số đoạn chạm tới, phần ngân sách đã dùng của `ours_fill`.
- UIT-ViQuAD và XQuAD-vi ở 16× và 32×: `ours_beta` so với `reranker`, `embed`, `bm25`, `span_sup`. Khi ngân sách chỉ còn
  1–2 đoạn, dữ liệu một bước có hết bão hòa không.

## Bước 2: đo độ trễ riêng (~20 phút, một GPU, không chạy gì khác cùng lúc)

```bash
M=/mnt/hps/anhm-paper/ttscompress/runs/main/models
EXTRA_ARMS=exit,recomp STAGES=bench \
BENCH_ARMS="ours_beta=pruner:$M/pruner_beta_primary,ours_fill=fill+pruner:$M/pruner_beta_primary,reranker=reranker:BAAI/bge-reranker-v2-m3,embed=embed:BAAI/bge-m3,bm25,exit,recomp" \
  bash run_pipeline.sh
```

- Kết quả: `results/eval_test/bench_latency.md`. Số này thay cho khoảng 11–77 ms và mức "chậm hơn bao nhiêu lần"
  đang ghi trong paper.
- `EXTRA_ARMS=exit,recomp` để pipeline chỉ kiểm tra gói của EXIT (peft), không tải lại dữ liệu nltk/spaCy của Provence
  trên pod mới.

## Bước 3: `ORACLE_N=500` (~1 giờ)

```bash
ORACLE_N=500 EXTRA_ARMS= STAGES="labels fit" bash run_pipeline.sh
ORACLE_N=500 FOLLOWUP_ARMS=1 EXTRA_RATIOS_SINGLE=16,32 EXTRA_ARMS= EVAL_READERS=Qwen/Qwen3-8B \
  STAGES="select answer" bash run_pipeline.sh
REPORT_OURS=ours_beta,ours_ens,ours_fill,ours_sent STAGES=report bash run_pipeline.sh
```

- Stage `labels` chỉ đo mới 400 tài liệu test mỗi nguồn cho Qwen3-8B (~50 phút, theo tốc độ của bản full). Các thư mục
  nhãn đã đo xong được bỏ qua sau ~10 giây mỗi thư mục; cấu hình nhãn khớp nhờ patch SEA-LION đã gộp.
- Stage `fit` mất ~1 phút.
- Cần xem: họ H1-oracle trên 500 tài liệu, và bảng "ours_beta − oracle_beta by reader".

## Bước 4: bổ sung ba reader còn lại (~1 giờ)

```bash
REPORT_OURS=ours_beta,ours_ens,ours_fill,ours_sent STAGES="answer report" bash run_pipeline.sh
```

## Gửi kết quả về để phân tích

Sau mỗi bước, upload báo cáo vào dataset HF dưới tên mới, để không ghi đè bản gốc:

```bash
E=/mnt/hps/anhm-paper/ttscompress/runs/main/results/eval_test
huggingface-cli upload thanthienhai/ttcompress-main-eval $E/report.json followup/report.json --repo-type dataset
huggingface-cli upload thanthienhai/ttcompress-main-eval $E/report.md followup/report.md --repo-type dataset
huggingface-cli upload thanthienhai/ttcompress-main-eval $E/bench_latency.md followup/bench_latency.md --repo-type dataset
huggingface-cli upload thanthienhai/ttcompress-main-eval /mnt/hps/anhm-paper/cluster_d936896_local.patch run/patches/20-cluster-rest.patch --repo-type dataset
```

File thứ tư là patch XProvence / `defusedxml`. Sau khi nó được gộp vào repo, chạy các bước còn lại của
`scripts/followup.sh`: `hard` (nhiễu cùng bài báo), `units` (đơn vị câu cho multi-hop, cần đo lại nhãn và train lại
pruner) và LLMLingua theo số token (`*_tt`, nằm trong bước `arms`). Riêng bước `units` không dùng XProvence nên chạy
được ngay: `bash scripts/followup.sh units`.

---

# Đợt 2: củng cố kết quả chấm điểm theo câu (không train lại)

Mục tiêu: biến kết quả khám phá của đợt 1 (`ours_sent` ngang hoặc vượt EXIT) thành bằng chứng dùng được trong paper.
Bốn việc, thứ tự chạy dưới đây. Không việc nào nạp XProvence, nên chạy được khi patch XProvence chưa gộp. Nếu job
ablation `units` / `hard` đang chiếm GPU, chờ nó xong hoặc chạy trên một pod khác cùng HPS.

Tổng ước tính trên 4×H100: khoảng 3 giờ.

## Bước 2.0: lấy code mới

```bash
cd /mnt/hps/anhm-paper/ttcompress_new
git fetch origin && git checkout main && git pull --ff-only origin main
git log --oneline -3          # phải có commit đăng ký trước docs/PREREG_SENTENCE_REPLICATION.md
M=/mnt/hps/anhm-paper/ttscompress/runs/main/models
R=/mnt/hps/anhm-paper/ttscompress/runs/main
```

## Bước 2.1: đo riêng độ trễ của `ours_sent` (~10 phút, một GPU, không chạy gì khác cùng lúc)

```bash
EXTRA_ARMS= STAGES=bench BENCH_OUT=$R/results/eval_test/bench_latency_sent.json \
BENCH_ARMS="ours_beta=pruner:$M/pruner_beta_primary,ours_sent=sent+pruner:$M/pruner_beta_primary,reranker=reranker:BAAI/bge-reranker-v2-m3" \
  bash run_pipeline.sh
```

`ours_beta` và `reranker` được đo lại cùng lượt làm mốc so sánh. `BENCH_OUT` giữ nguyên file `bench_latency.json` của
đợt 1.

## Bước 2.2: ba reader còn lại và các seed của `ours_sent` trên tài liệu chính (~1–1.5 giờ)

```bash
FOLLOWUP_ARMS=1 EXTRA_RATIOS_SINGLE=16,32 EXTRA_ARMS= STAGES="select answer" bash run_pipeline.sh
REPORT_OURS=ours_beta,ours_ens,ours_fill,ours_sent STAGES=report bash run_pipeline.sh
```

- `select` chỉ tạo các nhánh mới `ours_sent_s1`, `ours_sent_s2` (mọi thứ khác đã có trên đĩa).
- `answer` chạy với danh sách reader mặc định: Qwen3-8B trả lời các nhánh seed mới; Qwen3-1.7B, SEA-LION và
  Qwen3-32B trả lời mọi nhánh của đợt 1 (`*_sent`, `*_fill`, 16×/32×) và các nhánh seed.
- Cần xem: bảng Qwen3-32B của các nguồn nhiều bước (`ours_sent` so với `exit`), và bảng "Training-seed variation"
  có thêm `ours_sent`.

## Bước 2.3: lặp lại đã đăng ký trước trên tài liệu mới (~1.5 giờ)

Giả thuyết, biên, tài liệu và phép thử đã cố định trong `docs/PREREG_SENTENCE_REPLICATION.md` và
`docs/prereg_sentence_replication.json` **trước** lần chạy này. Không sửa hai file đó sau khi chạy (nếu buộc phải
sửa, ghi ở mục 8 của file `.md`).

```bash
ARMS_REPL="full,ours_beta=pruner:@MODELS@/pruner_beta_primary,ours_sent=sent+pruner:@MODELS@/pruner_beta_primary"
ARMS_REPL="$ARMS_REPL,ours_sent_s1=sent+pruner:@MODELS@/pruner_beta_primary_s1,ours_sent_s2=sent+pruner:@MODELS@/pruner_beta_primary_s2"
ARMS_REPL="$ARMS_REPL,reranker_sent=sent+reranker:@BACKBONE@,span_sent=sent+pruner:@MODELS@/pruner_span,exit"

EVAL_DIR=$R/results/eval_replication EVAL_SOURCES=vimqa,hotpotqa,2wiki EVAL_N=2000 EVAL_OFFSET=500 ORACLE_N=0 \
  EXTRA_ARMS=exit FOLLOWUP_ARMS=0 EXTRA_RATIOS_SINGLE= ONLY_ARMS="$ARMS_REPL" \
  STAGES="select answer" bash run_pipeline.sh
EVAL_DIR=$R/results/eval_replication PREREG_FILE=docs/prereg_sentence_replication.json \
  REPORT_OURS=ours_sent,ours_beta STAGES=report bash run_pipeline.sh
```

- `EVAL_N=2000 EVAL_OFFSET=500`: mỗi nguồn lấy các tài liệu xếp hạng 501–2500, không tài liệu nào trùng 500 tài liệu
  của lần chạy chính (VIMQA chỉ còn 199). Thư mục riêng `results/eval_replication`, nên không đụng tới kết quả cũ.
- `EXTRA_ARMS=exit` chỉ để pipeline kiểm tra gói của EXIT (peft); danh sách nhánh do `ONLY_ARMS` quyết định.
- Cần xem: mục "Pre-registered families" đầu `results/eval_replication/report.md` (R1, R2, R3, kèm cột "every seed
  agrees").

## Gửi kết quả về

```bash
huggingface-cli upload thanthienhai/ttcompress-main-eval $R/results/eval_test/bench_latency_sent.json followup2/bench_latency_sent.json --repo-type dataset
huggingface-cli upload thanthienhai/ttcompress-main-eval $R/results/eval_test/report.json followup2/report.json --repo-type dataset
huggingface-cli upload thanthienhai/ttcompress-main-eval $R/results/eval_test/report.md followup2/report.md --repo-type dataset
huggingface-cli upload thanthienhai/ttcompress-main-eval $R/results/eval_replication/report.json followup2/replication/report.json --repo-type dataset
huggingface-cli upload thanthienhai/ttcompress-main-eval $R/results/eval_replication/report.md followup2/replication/report.md --repo-type dataset
```
