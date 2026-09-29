# TODO

Các việc còn lại so với `METHOD_SPEC.md`, cập nhật 2026-09-29. Hạn nộp ARR: **2026-10-12**.

Hiện trạng: code §1–§7 gần đủ, 78/78 test pass. Pipeline mới chạy ở quy mô smoke (28/09, commit `80975e9`) và
pilot (26/09, code cũ hơn). Commit hiện tại (`b879b37`, thêm RECOMP / EXIT / LLMLingua / LongLLMLingua) chưa
chạy trên cluster.

## 1. Chạy trên cluster

Tất cả các bước dưới đây nằm trong **một job**: `python scripts/build_cluster_job.py` → gửi
`dist/ttcompress_job.sh` (`bash ttcompress_job.sh`). Theo dõi ở dataset HF riêng tư
`<token user>/ttcompress-campaign-coling2027` (`STATUS.md`). Gửi lại cùng file để chạy tiếp.

- [x] Code: `scripts/cluster_campaign.sh` (cài gói Python thiếu vào `$BASE/pylibs` qua mirror `hub.fci.vn`, gói
  baseline vào `$BASE/sites/*`), `scripts/campaign_status.py`, `scripts/build_cluster_job.py`; test bằng stub
  (`tests/test_campaign.py`).
- [ ] Gửi job. Chạy thật lần đầu trên cluster: smoke → probe từng baseline → pilot 300/30/50 → full 3000/300/500
  → `abl_hard`, `abl_pad` (20000 ký tự) → so sánh → upload model + eval của bản full.
- [ ] Xem phần "Published compressors" trong `STATUS.md`: baseline nào `failed` và vì sao. Các điểm dễ hỏng:
  - [ ] Provence / XProvence: API `process()` (trả về `reranking_score` lồng nhau), spaCy `xx_sent_ud_sm` tải từ GitHub.
  - [ ] LLMLingua-2: `rate` tính theo token XLM-R, cờ `truncated`.
  - [ ] EXIT: tài khoản của `HF_TOKEN` phải chấp nhận license Gemma (`google/gemma-2b-it`) trên huggingface.co.
  - [ ] RECOMP, và pass select thứ hai của LLMLingua / LongLLMLingua với transformers 4.46.3.
- [ ] Sau pilot: đọc "Projection from the pilot" trong `STATUS.md`. Nếu bản full không kịp hạn, hủy job và gửi lại
  với `--set MAIN_SIZES="<n_train> <n_dev> <n_test>"` (label đã đo được dùng lại).

## 2. Code, sửa trước bản full

- [ ] Kiểm tra "mọi seed đồng thuận" chưa áp dụng cho H3 / H3-heldout (`evaluate.py:586` chỉ xét test có khóa
  `reader`), nên `ours_ens_s<k>` chỉ vào bảng SD.
- [x] `generate_labels.py`: `max_new_tokens` của shard rỗng giờ lấy theo hop của source (có test).
- [x] Đối chứng RQ2: `pruner_span` giờ train trên đúng tập doc của `pruner_beta_primary` (bỏ doc không mang
  thông tin cho mọi pruner).
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

- [ ] Backend vLLM (`VLLMReader`).
- [ ] Nhánh BCE của `pruner_span`; vòng train và chọn epoch theo dev.
- [ ] Mỗi cửa sổ đều lặp lại câu hỏi khi tài liệu dài hơn `--max-len`.
- [ ] Các compressor Provence / LLMLingua.

## 5. `METHOD_SPEC.md`

- [x] §7: lệnh pilot thêm `RUN_ROOT=runs/pilot` (và `run_pipeline.sh` có guard `run_config.txt`).
- [x] §2: UIT-ViQuAD 2.0 thay cho số liệu v1.
- [x] §4: input định dạng cặp `<s> q </s></s> chunks </s>`, cắt câu hỏi 96 / chunk 512 token, `pos_weight` của BCE.
- [x] §2: ai đọc `gold_chunks` (oracle, `span_sup`, chẩn đoán).
- [x] §3: mask mô tả đúng như code (Bernoulli độc lập, rồi lật 1 chunk nếu mask rỗng / giữ tất cả), kèm tần suất
  theo C (C = 2: mọi mask giữ đúng 1 chunk). Code giữ nguyên để không trộn hai quy trình mask trong một thư mục label.
- [x] §9: cập nhật hiện trạng 2026-09-29 (những gì đã / chưa chạy trên cluster, probe của job kiểm chứng baseline).
- [ ] §9: điền kết quả probe (baseline nào `ok` / `failed`, lỗi gì) từ `STATUS.md` sau khi job chạy.

## 6. Bài báo và nộp

- [ ] Điền kết quả từ bản full vào `paper/main.tex` (§Kết quả, `\todo` dòng 540) và số tài liệu mỗi tập (dòng 466).
- [ ] Dịch bản nháp sang tiếng Anh: bản nộp ARR phải là tiếng Anh (`paper/main.tex:2`).
- [ ] Trích dẫn UIT-ViQuAD 2.0 (bài VLSP 2021 ViMRC): `paper/main.tex:437` mới chỉ cite bản v1 (`nguyen2020viquad`),
  nhưng code dùng 2.0.
- [ ] Xác nhận giấy phép UIT-ViQuAD (chỉ dùng cho nghiên cứu) và VIMQA (user agreement) (`\todo` dòng 695).
- [ ] Kiểm tra tác giả / venue các mục `TODO` trong `paper/references.bib`: `ecorag2025`, `looComp2026`, `poc2026`,
  `readerScaling2026`, `coreRag2026`, `hwang2025exit`, `vimqa2022`, `xprovence2026`, `sealion2024`.
- [ ] Rà lại novelty ngay trước khi nộp ("F1 surrogate pseudo-label pruner", "amortized attribution compression").
- [ ] Xác nhận giới hạn trang và danh sách area trên 2027.coling-iccl.org.
- [ ] Nộp ARR trước **2026-10-12**.

## Ngoài phạm vi (ghi vào Limitations, không làm)

CORE-RAG (không có checkpoint), CompAct / RECOMP abstractive (không kiểm soát được độ dài), ContextCite lúc
inference (`oracle_beta` là bản tương đương cùng ngân sách), surrogate có tương tác cặp giữa các chunk.
