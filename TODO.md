# TODO

Các việc còn lại so với `METHOD_SPEC.md`, rà soát 2026-09-30. Hạn nộp ARR: **2026-10-12**.

Hiện trạng: bản full 3000/300/500 đã chạy xong trên cluster (29/09, code `d936896` + 2 patch chưa commit;
`reports/RUN_REPORT_2026-09-30_full_h100x4.md`; dữ liệu eval trên HF `thanthienhai/ttcompress-main-eval`). Code
follow-up của 30/09 đã viết, 97/97 test pass, chưa commit và chưa chạy trên cluster.

## 0. Follow-up sau bản full (2026-09-30)

Phân tích theo từng tài liệu và các thay đổi code: `METHOD_SPEC.md` §9.

- [x] Gộp patch prompt base của SEA-LION (`run/patches/10-sealion-base-prompt.patch` trong dataset HF): `\n` sau
  `### Đáp án:`, `BASE_PROMPT_VERSION = 2`, `base_prompt_version` trong `measure_config.json` của reader base.
  Cấu hình code ghi ra khớp cả 23 thư mục nhãn của bản full; 98/98 test pass.
- [ ] **Chặn: lấy phần patch còn lại từ cluster và gộp**: XProvence trên transformers 5.x (`ttcompress/selection.py`)
  và `defusedxml` (`run_pipeline.sh`). *Chưa làm vì:* không được lưu kèm dataset HF, chỉ có trong thư mục code
  trên cluster `/mnt/hps/anhm-paper/ttcompress_new` (HEAD `d936896`). Lấy bằng
  `git -C /mnt/hps/anhm-paper/ttcompress_new diff -- ttcompress/selection.py run_pipeline.sh > cluster_rest.patch`.
  Thiếu nó, các bước `bench`, `arms` (pass 16x/32x) và `hard` sẽ dừng khi nạp XProvence.
- [ ] Đưa code mới lên cluster, giữ nguyên `.env` của bản full (`RUN_ROOT=.../runs/main`, `LABELS=.../runs/labels_v2`).
- [x] Code: nhánh `sent+` / `fill+`, `llmlingua_tt` / `longllmlingua_tt`, `EXTRA_RATIOS_SINGLE`, `CONFIRM_RATIOS`,
  `MULTIHOP_UNITS=sentence`, `STAGES=bench` (`scripts/bench_latency.py`), mục Diagnostics + oracle gap theo reader
  trong `report.md`, hình `tokens.pdf`, `scripts/followup.sh`; 97/97 test pass. Báo cáo tạo lại từ dữ liệu HF khớp
  từng test của 6 họ giả thuyết.
- [x] Paper: điền số của bản full (mọi dòng bảng chính, dấu Holm, bảng chi phí, bảng Qwen3-32B), mục phân tích theo
  từng tài liệu (bảng độ phủ, bảng khoảng cách oracle theo reader), chất lượng nhãn, 4 hình dữ liệu trong
  `paper/figures/`; biên dịch sạch bằng pdfLaTeX và XeLaTeX.
- [ ] Chạy trên cluster: trước tiên các bước không train lại trong `docs/FOLLOWUP_RUNBOOK.md` (nhánh câu / lấp ngân
  sách, 16×/32×, độ trễ, `ORACLE_N=500`); sau đó phần còn lại của `bash scripts/followup.sh` (`hard`, `units`, `*_tt`). *Chưa làm vì:* cần GPU cluster, và phụ thuộc mục chặn ở trên.
- [ ] Điền kết quả follow-up vào `paper/main.tex` (14 chỗ `\todo`). *Chưa làm vì:* chờ kết quả bước trên.
- [ ] Xác nhận `span_sup` của bản full train trên đúng tập doc của `ours_beta` (bản sửa ở `561254e`, bản full chạy
  `d936896`); nếu không, train lại `pruner_span` và chọn lại `span_sup`. *Chưa làm vì:* pruner của bản full chưa
  được upload lên HF (chỉ có pruner của pilot), nên không đọc được từ máy này. Cách kiểm: so `n_train` trong
  `runs/main/models/pruner_span/train_log.json` với `pruner_beta_primary/train_log.json` trên cluster.
- [ ] Commit các thay đổi follow-up. *Chưa làm vì:* chưa có yêu cầu commit; nên commit cùng lúc với 2 patch ở mục chặn.

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
  - [ ] `abl_hard`. *Chưa chạy:* chuyển sang bước `hard` của `scripts/followup.sh`.
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
