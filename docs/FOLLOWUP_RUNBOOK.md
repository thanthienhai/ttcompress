# Runbook: thí nghiệm follow-up trên cluster (sau bản full 2026-09-29)

Mục tiêu: lấy kết quả đánh giá sớm mà **không train lại**, chỉ dùng lại nhãn, pruner và selections của bản full
(`RUN_ROOT=/mnt/hps/anhm-paper/ttscompress/runs/main`, `LABELS=/mnt/hps/anhm-paper/ttscompress/runs/labels_v2`).
Lý do của từng thí nghiệm: `METHOD_SPEC.md` §9. Việc còn lại: `TODO.md` §0.

Mọi lệnh dưới đây **không nạp XProvence**, vì patch cho XProvence trên transformers 5.x chưa được gộp vào repo.
Mọi lệnh chạy lại được: selections, câu trả lời và nhãn đã có trên đĩa sẽ được bỏ qua.

## Bước 0: đưa code mới lên cluster (~5 phút)

Ở máy local: commit và push các thay đổi follow-up.

```bash
git checkout -b followup-2026-09-30
git add -A ttcompress evaluate.py generate_labels.py run_pipeline.sh scripts tests paper METHOD_SPEC.md README.md TODO.md docs/FOLLOWUP_RUNBOOK.md
git commit -m "Follow-up: sentence/fill arms, extra single-hop ratios, sentence units, bench, diagnostics"
git push -u origin followup-2026-09-30
```

Trên cluster: dùng đúng thư mục code của bản full, để giữ nguyên `.env`, `.baseline_site` và `.llmlingua_site`.

```bash
cd /mnt/hps/anhm-paper/ttcompress_new
git diff > /mnt/hps/anhm-paper/cluster_d936896_local.patch   # lưu patch XProvence + defusedxml để gộp sau
git stash push -m "full-run local patches (d936896)"
git fetch origin && git checkout followup-2026-09-30
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
