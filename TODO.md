# TODO

Các việc còn lại so với `METHOD_SPEC.md`, rà soát 2026-10-01. Hạn nộp ARR: **2026-10-12**.

Hiện trạng: bản full 3000/300/500 đã chạy xong trên cluster (29/09, code `d936896` + 2 patch;
`reports/RUN_REPORT_2026-09-30_full_h100x4.md`; dữ liệu eval trên HF `thanthienhai/ttcompress-main-eval`). Code
follow-up đã commit và push lên `main`; các bước follow-up không train lại đã chạy (30/09), hai ablation đang chạy.

## 0. Follow-up sau bản full (2026-09-30)

Phân tích theo từng tài liệu và các thay đổi code: `METHOD_SPEC.md` §9.

- [x] Gộp patch prompt base của SEA-LION (`run/patches/10-sealion-base-prompt.patch` trong dataset HF): `\n` sau
  `### Đáp án:`, `BASE_PROMPT_VERSION = 2`, `base_prompt_version` trong `measure_config.json` của reader base.
  Cấu hình code ghi ra khớp cả 23 thư mục nhãn của bản full; 98/98 test pass.
- [x] Đưa code mới lên cluster (`c4967c3`), giữ `.env` của bản full.
- [x] Code: nhánh `sent+` / `fill+`, `llmlingua_tt` / `longllmlingua_tt`, `EXTRA_RATIOS_SINGLE`, `CONFIRM_RATIOS`,
  `MULTIHOP_UNITS=sentence`, `STAGES=bench` (`scripts/bench_latency.py`), mục Diagnostics + oracle gap theo reader
  trong `report.md`, hình `tokens.pdf`, `scripts/followup.sh`; 97/97 test pass. Báo cáo tạo lại từ dữ liệu HF khớp
  từng test của 6 họ giả thuyết.
- [x] Paper: điền số của bản full (mọi dòng bảng chính, dấu Holm, bảng chi phí, bảng Qwen3-32B), mục phân tích theo
  từng tài liệu (bảng độ phủ, bảng khoảng cách oracle theo reader), chất lượng nhãn, 4 hình dữ liệu trong
  `paper/figures/`; biên dịch sạch bằng pdfLaTeX và XeLaTeX.
- [x] Chạy trên cluster các bước không train lại của `docs/FOLLOWUP_RUNBOOK.md` (30/09;
  `reports/FOLLOWUP_PROGRESS_REPORT.md`; kết quả trên HF `followup/`): nhánh `sent+` / `fill+` và 16×/32× (chỉ
  Qwen3-8B), benchmark độ trễ, `ORACLE_N=500`. Kết quả xác nhận của bản full được giữ nguyên (H1, H2, H3 trùng từng
  test); H1-oracle 1/10 với 500 tài liệu.
- [ ] Ba reader còn lại (Qwen3-1.7B, SEA-LION, Qwen3-32B) trả lời các nhánh mới và các tỉ lệ 16×/32×.
  *Chưa làm:* báo cáo tiến độ ghi là đã xong, nhưng `followup/report.json` chỉ có câu trả lời của Qwen3-8B cho các
  lựa chọn mới. Lệnh: Bước 2.2 của runbook.
- [ ] `llmlingua_tt` / `longllmlingua_tt`. *Chưa chạy:* báo cáo tiến độ ghi là đã xong, nhưng Bước 1 đặt `EXTRA_ARMS=`
  nên hai nhánh này không được thêm (không có trong `report.json`). Cần `EXTRA_ARMS` có `llmlingua,longllmlingua`.
- [ ] Đo riêng độ trễ của `ours_sent` (benchmark mới có `ours_beta` và `ours_fill`). Lệnh: Bước 2.1 của
  `docs/FOLLOWUP_RUNBOOK.md`.
- [ ] Các seed của `ours_sent` (`ours_sent_s1`, `_s2`) trên tài liệu chính, cùng lượt với ba reader còn lại: Bước 2.2.
- [x] Đăng ký trước lần lặp lại `ours_sent` so với EXIT / `reranker_sent` trên tài liệu mới
  (`docs/PREREG_SENTENCE_REPLICATION.md`, `docs/prereg_sentence_replication.json`); code: `select --offset`,
  `report --prereg`, `EVAL_N` / `EVAL_OFFSET` / `ONLY_ARMS` / `PREREG_FILE` / `BENCH_OUT` trong `run_pipeline.sh`.
- [x] Phụ lục đăng ký trước R4–R7 (2026-10-01, trước khi chạy; `docs/PREREG_SENTENCE_REPLICATION.md` §9, JSON chỉ
  thêm họ, R1–R3 giữ nguyên): R4 `ours_beta` > `reranker` (theo đoạn, 4×/8×), R4b > `xprovence` (tùy chọn), R5
  `ours_sent` > `span_ans_sent`, R6 `ours_sent` không kém `span_sent` quá 0.02, R7 `fuse_sent` > `span_sent`. Code:
  `--label-source answer` (`pruner_span_ans`, nhãn đoạn chứa chuỗi đáp án, không đọc câu hỗ trợ kể cả lúc chọn
  epoch), arm `rrf:a|b` (reciprocal rank fusion, k = 60), `ROUND2_ARMS=1` trong `run_pipeline.sh`, report ghi
  "Amended"; runbook Đợt 2 thêm Bước 2.1b và các nhánh mới; 104/104 test pass.
- [ ] Chạy Đợt 2 (Bước 2.1, 2.1b, 2.2, 2.3 của runbook, ~5 giờ); rồi điền R1–R7 vào paper (`\todo{kết quả R1--R7.}`).
- [x] Ablation `units` (`MULTIHOP_UNITS=sentence`, đo lại nhãn + train lại, 4 reader): xong 30/09, kiểm 01/10 từ
  `followup/abl_units_report.md` (HF chỉ có bản `.md`). Kết quả (khám phá): `ours_beta` mức câu hơn `ours_sent`
  không quá +0.028 F1; hơn EXIT bản chính ở 8× +0.040 / +0.004 / +0.112 (VIMQA / HotpotQA / 2Wiki, Qwen3-8B) và
  +0.022 / +0.004 / +0.108 (Qwen3-32B); H1 28/30, H1-oracle 0/6, H2a 2/12 (cả hai là so với `oracle_span` trên
  VIMQA; so với `span_sup` cả 6 ô âm, −0.024..−0.003), H3 0/16, H3-heldout 0/16; R² surrogate giảm 0.56/0.54 →
  0.26. Đã vào paper (Bảng `tab:units`, (b)).
  *Báo cáo tiến độ `ABL_UNITS_PROGRESS_REPORT_2026-09-30.md` sai ở §2.2–2.5:* các cột "F1" là answer-in-context
  (ví dụ 2Wiki 8× "0.726 vs 0.282", `oracle_span` "1.000"); F1 thật 0.568 vs 0.347. Δ với `span_sup` ghi dương
  trên 2Wiki nhưng thật ra âm; "H1 62% → 93%" so hai họ khác cỡ (cùng 30 phép kiểm định: 27/30 → 28/30); "*_tt
  đã xong" vẫn sai; "`ours_fill` thu hẹp 25–35% khoảng cách tới EXIT" sai (8×: 37% VIMQA, 19% 2Wiki, âm trên
  HotpotQA).
- [ ] Upload `report.json` (và `answers_*.jsonl`) của `units-sentence` lên HF: cần cho kiểm định ghép cặp theo
  `doc_id` giữa hai lần chạy (`ours_beta` mức câu vs EXIT / `ours_sent`); hiện các so sánh đó trong paper chỉ là mô tả.
  Tài liệu test trùng theo `doc_id` (cùng hàng, `sources.multihop_row_to_doc`), nên ghép cặp được.
- [ ] Ablation `hard`: đang chạy (job `ttcompress-ablation-h100x4`, 30/09 16:21 UTC còn ở bước labels, 112/750 tài
  liệu mỗi shard); sau đó `compare`. Điền (d) trong paper.
- [x] Điền kết quả follow-up vào `paper/main.tex`: bảng `tab:sentence`, `tab:units`, mục Thí nghiệm bổ sung (a), (b),
  (c), (e), (f), bảng chi phí đo riêng, H1-oracle với 500 tài liệu, hình pareto (thời gian đo riêng) và tokens (có
  `ours_fill` / `ours_sent`, 16×/32×); tóm tắt, giới thiệu, kết luận, hạn chế nhắc (b). Còn 6 `\todo`: ba reader
  cho (a)/(c), bench `ours_sent`, R1–R3 (đều chờ Đợt 2), (d) chờ `abl_hard`, (g) `*_tt` chưa chạy, giấy phép.
  Biên dịch sạch bằng pdfLaTeX (thân bài tới trang 16).
- [x] Kiểm `span_sup` của bản full: train trên 9000 tài liệu, `ours_beta` trên 8628 (372 tài liệu không mang thông
  tin, 4.1%, chỉ `span_sup` giữ; log `run/logs/train_pruner_*.log` trên HF). Lợi thế nhỏ nghiêng về đối chứng, nên
  kết quả âm của H2 là thận trọng; đã ghi vào paper. Train lại `pruner_span` là tùy chọn.
- [ ] Gộp patch XProvence / `defusedxml` vào repo: đã có trên HF (`run/patches/20-cluster-rest.patch`). Cần người
  duyệt và áp dụng (01/10 Claude không được phép áp patch tải từ HF). Là điều kiện để chạy R4b; quyết định trước khi
  chạy Bước 2.3.

## 1. Chạy trên cluster

Tất cả các bước dưới đây nằm trong **một job**: `python scripts/build_cluster_job.py` → gửi
`dist/ttcompress_job.sh` (`bash ttcompress_job.sh`). Theo dõi ở dataset HF riêng tư
`<token user>/ttcompress-campaign-coling2027` (`STATUS.md`). Gửi lại cùng file để chạy tiếp.

- [x] Code: `scripts/cluster_campaign.sh` (cài gói Python thiếu vào `$BASE/pylibs` qua mirror `hub.fci.vn`, gói
  baseline vào `$BASE/sites/*`), `scripts/campaign_status.py`, `scripts/build_cluster_job.py`; test bằng stub
  (`tests/test_campaign.py`).
- Chạy thật trên cluster: full 3000/300/500 → `abl_hard` → so sánh → upload model + eval của bản full.
  - [x] Bản full 3000/300/500: job `ttcompress-full-h100x4`, xong 29/09 (chạy thẳng `run_pipeline.sh` với
    `RUN_ROOT=runs/main`, không qua campaign). Smoke đã chạy 28/09, pilot 26/09.
  - [x] Upload eval + nhãn của bản full: `thanthienhai/ttcompress-main-eval`.
  - [ ] `abl_hard`. *Đang chạy* (bước `hard` của `scripts/followup.sh`, xem §0).
  - [ ] Upload pruner của bản full lên HF. *Chưa làm:* HF mới có pruner của pilot; cần cho việc kiểm `span_sup` (§0)
    và để công bố. Chạy `STAGES=upload` từ một pod CPU.
- [x] Kiểm tra các baseline đã công bố: cả 7 chạy đủ 500/500 tài liệu ở mọi ô của bản full, không tài liệu nào lỗi
  (kiểm trực tiếp trên `report.json` thay cho `STATUS.md`, vì bản full không chạy qua campaign).
  - [x] Provence / XProvence: chạy được; XProvence cần một patch cho transformers 5.x (chưa commit, xem §0).
  - [x] LLMLingua-2: chạy được, cờ `truncated` hoạt động (bị cắt tới 35% tài liệu, nhiều nhất ở dữ liệu một bước).
  - [x] EXIT: chạy được (tài khoản của `HF_TOKEN` đã chấp nhận license Gemma).
  - [x] RECOMP, và pass select thứ hai của LLMLingua / LongLLMLingua với transformers 4.46.3: chạy được
    (LLMLingua / LongLLMLingua bị cắt đuôi ở 34--51% tài liệu nhiều bước ở 8x; bản sửa là nhánh `*_tt`).

## 2. Code, sửa trước bản full

- [x] `generate_labels.py`: `max_new_tokens` của shard rỗng giờ lấy theo hop của source (có test).
- [x] Đối chứng RQ2: `pruner_span` giờ train trên đúng tập doc của `pruner_beta_primary` (bỏ doc không mang
  thông tin cho mọi pruner). Lưu ý: bản sửa ở `561254e`, bản full chạy `d936896` (xem §0).
- [x] ListNet cộng tổng / MSE trung bình: không phải lỗi, gradient hai số hạng cùng cỡ với mọi số chunk; đã ghi
  vào spec §4 và comment `listnet_loss`.

## 3. Code, nhỏ

- [x] Arm theo chunk: separator được tính vào ngân sách, văn bản nối được đếm lại và bỏ chunk yếu nhất đến khi
  vừa (`select_by_scores`), nên `kept_tokens` luôn ≤ ngân sách.
- [x] Baseline đã công bố mang nhãn lạ: cảnh báo lúc `select`, và `report.md` liệt kê các arm không thuộc họ giả
  thuyết nào (`outside_families`).
- [x] Bảng Cost trong run ablation: mỗi (reader, tập label) lấy từ thư mục label đầu tiên có nó (label của
  ablation thắng), không còn dòng trùng.
- [x] Thiết lập đo label (`tp`, `docs_per_call`, `batch_size`, `num_shards`, phiên bản vLLM/torch/transformers)
  ghi vào `measure_runs.jsonl` để truy nguồn gốc; cố ý không đưa vào phép so khớp `measure_config.json` (để
  label dùng lại được giữa các cấu hình GPU).
- [x] `2wiki` không còn split train (`AVAILABLE_SPLITS`), giống `xquad_vi`.
- [x] `report.md` có cột answer recall.
- [x] Hình theo độ sâu needle vẽ thêm `abl_posadj` (nét đứt, cùng màu Ours-β).
- [x] Docstring `reader.py`: 64 / 48.
- [x] Toàn bộ `tests/` chạy sau `prefetch` trong một lần chạy có `preflight` (dataset và model test đã được cache;
  model test tí hon được thêm vào prefetch); test gọi bash không bị ảnh hưởng bởi biến môi trường của run bao ngoài.

## 4. Test còn thiếu

- [ ] Backend vLLM (`VLLMReader`). *Chưa làm vì:* cần Linux + GPU, máy phát triển là Windows không GPU. Đã chạy
  thật trong smoke, pilot và bản full.
- [ ] Nhánh BCE của `pruner_span`; vòng train và chọn epoch theo dev. *Chưa làm vì:* ưu tiên thấp hơn follow-up;
  bản full đã train 8 pruner thành công. Viết được trên CPU với model tí hon (`TINY_ENCODER`).
- [ ] Mỗi cửa sổ đều lặp lại câu hỏi khi tài liệu dài hơn `--max-len`. *Chưa làm vì:* ưu tiên thấp;
  `test_pack_windows_spans_are_exact_and_every_chunk_appears_once` đã kiểm nhiều cửa sổ, token mở/đóng và khoảng
  của từng chunk, chỉ chưa kiểm câu hỏi ở đầu mỗi cửa sổ.
- [ ] Các compressor Provence / LLMLingua. *Chưa làm vì:* test cần tải model remote-code, spaCy `xx_sent_ud_sm`
  và transformers 4.46.3; cả hai đã chạy đủ 500/500 tài liệu mỗi ô trong bản full. Đường `*_tt` (mục tiêu số
  token) có test bằng compressor giả lập.

## 5. `METHOD_SPEC.md`

- [x] §7: lệnh pilot thêm `RUN_ROOT=runs/pilot` (và `run_pipeline.sh` có guard `run_config.txt`).
- [x] §2: UIT-ViQuAD 2.0 thay cho số liệu v1.
- [x] §4: input định dạng cặp `<s> q </s></s> chunks </s>`, cắt câu hỏi 96 / chunk 512 token, `pos_weight` của BCE.
- [x] §2: ai đọc `gold_chunks` (oracle, `span_sup`, chẩn đoán).
- [x] §3: mask mô tả đúng như code (Bernoulli độc lập, rồi lật 1 chunk nếu mask rỗng / giữ tất cả), kèm tần suất
  theo C (C = 2: mọi mask giữ đúng 1 chunk). Code giữ nguyên để không trộn hai quy trình mask trong một thư mục label.
- [x] §9: cập nhật hiện trạng 2026-09-29 (những gì đã / chưa chạy trên cluster, probe của job kiểm chứng baseline).
- [x] §9: hiện trạng 2026-09-30 sau bản full (kết quả các họ, phân tích theo từng tài liệu, các thay đổi follow-up).
- [ ] §9: ghi kết quả các baseline đã công bố trong bản full (cả 7 chạy đủ; XProvence cần patch cho transformers
  5.x; LLMLingua / LongLLMLingua bị cắt đuôi 34--51% ở 8x). *Chưa làm vì:* mới ghi ở §1 của file này, chưa chép
  vào §9.

## 6. Bài báo và nộp

- [x] Điền kết quả từ bản full vào `paper/main.tex` (mục Kết quả, Phân tích, Kết luận, Hạn chế, phụ lục Qwen3-32B)
  và số tài liệu mỗi tập (chú thích Bảng dữ liệu).
- [ ] Dịch bản nháp sang tiếng Anh: bản nộp ARR phải là tiếng Anh (`paper/main.tex:2`). *Chưa làm vì:* nội dung
  còn đổi khi có kết quả follow-up; dịch sau để khỏi dịch hai lần.
- [ ] Trích dẫn UIT-ViQuAD 2.0 (bài VLSP 2021 ViMRC): `paper/main.tex` mới chỉ cite bản v1 (`nguyen2020viquad`),
  nhưng code dùng 2.0. *Chưa làm vì:* cần tra thông tin xuất bản chính xác (tác giả, venue, trang); không điền
  bib từ trí nhớ.
- [ ] Xác nhận giấy phép UIT-ViQuAD (chỉ dùng cho nghiên cứu) và VIMQA (user agreement) (`\todo` trong mục Cân nhắc
  đạo đức). *Chưa làm vì:* cần người đọc điều khoản hoặc liên hệ nhóm phát hành dữ liệu.
- [ ] Kiểm tra tác giả / venue các mục `TODO` trong `paper/references.bib`: `ecorag2025`, `looComp2026`, `poc2026`,
  `readerScaling2026`, `coreRag2026`, `hwang2025exit`, `vimqa2022`, `xprovence2026`, `sealion2024`. *Chưa làm vì:*
  cần tra từng bài trên arXiv / ACL Anthology.
- [ ] Rà lại novelty ngay trước khi nộp ("F1 surrogate pseudo-label pruner", "amortized attribution compression").
  *Chưa làm vì:* theo kế hoạch, làm sát ngày nộp.
- [ ] Xác nhận giới hạn trang và danh sách area trên 2027.coling-iccl.org. *Chưa làm vì:* cần kiểm trên trang CFP.
- [ ] Nộp ARR trước **2026-10-12**.

## Ngoài phạm vi (ghi vào Limitations, không làm)

CORE-RAG (không có checkpoint), CompAct / RECOMP abstractive (không kiểm soát được độ dài), ContextCite lúc
inference (`oracle_beta` là bản tương đương cùng ngân sách), surrogate có tương tác cặp giữa các chunk.
