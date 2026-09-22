# Outcome-Supervised Relevance — Pipeline 3 Giai đoạn — Method Spec (v2, dùng public dataset)

> **Trạng thái:** v2 — thay toàn bộ dữ liệu nền nội bộ (`data/vncompress_vi_v2`, `qa_combined.jsonl`) bằng public dataset `UIT-ViQuAD 2.0`. **`vcc_bench_v2.json` (test set) giữ nguyên** làm benchmark báo cáo cuối — xem giả định ở §0.
> **Ngữ cảnh dự án:** LACC/VNCompress. Xem `WAVE4_REPORT.md` cho dữ liệu nền (E4/E6/tone-probe/truncation). Xem `PCS_METHOD_SPEC.md` cho công thức `g(i,L)` và hạ tầng sweep `λ` — pipeline này **tái dùng**, không viết lại.
> **⚠️ Đổi khung ngân sách so với PCS:** PCS là bolt-on, 0 GPU-hour train. Pipeline này chủ động dỡ ràng buộc "không train" — phần đắt là **Giai đoạn A: sinh label** (hàng nghìn lượt gọi reader 8B). Phải xác nhận ngân sách GPU trước khi chạy full-scale (§7).

---

## 0. Giả định cần xác nhận trước khi code

**Test set không đổi:** `relevance_v2`/`relevance_v3` vẫn được đánh giá cuối cùng trên `vcc_bench_v2.json` (qua PCS sweep, §6) để so sánh được với `encoder`/`truncation`/`h2o`/`snapkv` trong cùng bảng kết quả của paper. Chỉ **nguồn dữ liệu train** (Giai đoạn A) đổi sang public. Nếu sai giả định này — báo lại trước khi coding agent bắt đầu.

---

## 1. Dữ liệu nền — public, thay thế toàn bộ nguồn nội bộ

| Trước (nội bộ, đã bỏ) | Sau (public, dùng thay) |
|---|---|
| `qa_combined.jsonl` (32.951 mẫu/3.155 doc) | **`taidng/UIT-ViQuAD2.0`** (HuggingFace) — Train 28.457 câu hỏi / Dev ~5.700 / Public test ~3.821, dựa trên 5.109 đoạn văn / 174 bài Wikipedia tiếng Việt (Nguyen et al., COLING 2020) |
| `data/vncompress_vi_v2` (corpus thô) | Các đoạn văn (`context`) đi kèm trong chính UIT-ViQuAD 2.0 — không cần corpus riêng |
| `data/benchmark/vcc_bench_v2.json` | **Giữ nguyên, không đổi** — vẫn là test set cuối (§0) |

**⚠️ License — cần bạn tự xác nhận trước khi công bố paper:** điều khoản gốc từ nhóm UIT (qua bản chị em UIT-ViWikiQA) ghi "chỉ dùng cho nghiên cứu phi lợi nhuận" — khác với license MIT ghi trên một số model-card mirror trên HF. Kiểm tra lại trang chính thức của nhóm NLP@UIT trước khi coding agent tải dataset, và trích dẫn đúng paper gốc trong Related Work.

**Việc đầu tiên của coding agent:** kiểm tra giao giữa nội dung UIT-ViQuAD 2.0 (title/context Wikipedia) và 14 document trong `vcc_bench_v2.json` — phải rỗng hoặc gần rỗng (khác nguồn corpus nên rủi ro thấp, nhưng vẫn phải log kết quả kiểm tra, đúng kỷ luật đã áp dụng ở `WAVE4_REPORT.md §4.3`).

---

## 2. Giai đoạn A — Sinh label bằng outcome đo được

### 2.1 Đơn vị: chunk

⚠️ **[CẦN XÁC NHẬN — chặn tiến độ, không đổi từ v1]** Đơn vị chọn của E6 (token hay chunk) chưa xác nhận. Pipeline này giả định chunk. Kiểm tra interface thật của E6 trong repo trước khi code.

### 2.2 Xây dựng document dài từ UIT-ViQuAD (mới trong v2 — bắt buộc vì UIT-ViQuAD gốc là đoạn văn ngắn, không phải document nhiều chunk)

UIT-ViQuAD 2.0 mỗi mẫu chỉ có **1 đoạn văn ngắn** (context) + câu hỏi — không tự nhiên có cấu trúc "nhiều chunk" như `qa_combined.jsonl` cũ. Cần dựng lại document dài kiểu needle-in-haystack:

1. **Needle** = đoạn văn (context) gốc chứa câu trả lời của mẫu UIT-ViQuAD đang xét.
2. **Haystack** = ghép thêm `M` đoạn văn khác từ UIT-ViQuAD (không chứa câu trả lời của câu hỏi hiện tại — kiểm tra loại trừ) làm nhiễu, chèn needle vào vị trí ngẫu nhiên trong chuỗi đoạn nhiễu này.
3. Toàn bộ chuỗi (haystack + needle) chia thành `C` chunk = từng đoạn văn (mỗi đoạn UIT-ViQuAD gốc = 1 chunk).

⚠️ **[CẦN XÁC NHẬN]** Cách chọn `M` đoạn nhiễu (ngẫu nhiên toàn bộ tập, hay cùng chủ đề/bài Wikipedia để tăng độ khó?) nên **khớp đúng cách task `needle_in_haystack` nội bộ đang dùng** (đã có sẵn trong benchmark hiện tại, dùng nội dung nội bộ) — coding agent cần đọc code/config của task `needle_in_haystack` hiện có trong repo (không phải dữ liệu, chỉ **phương pháp dựng haystack**) và áp cùng logic lên nguồn UIT-ViQuAD, để độ khó hai bộ dữ liệu (train mới vs test `vcc_bench_v2.json`) tương đương nhau. Không tự chọn cách dựng nhiễu tuỳ tiện.

### 2.3 Sinh mask

Không đổi so với v1: `K` mask nhị phân ngẫu nhiên trên `C` chunk, tập trung xác suất giữ quanh `1/4` hoặc `1/8` (khớp ratio 4x/8x đang eval). Ghép chunk giữ lại theo đúng thứ tự gốc trong document đã dựng ở §2.2, tái dùng hàm ghép có sẵn (`vncompress/compression.py`).

### 2.4 Đo outcome

Không đổi so với v1: đưa văn bản đã ghép + câu hỏi (từ mẫu UIT-ViQuAD gốc) vào reader (Qwen3-8B, không train), đo `token_f1` (dùng lại hàm metric có sẵn). Lưu `(mask, token_f1)` cho từng document ở dạng `outcome_labels_raw/{doc_id}.jsonl`.

### 2.5 Ước lượng điểm đóng góp từng chunk

Không đổi so với v1:

```
token_f1(m_k) ≈ β_0 + Σ_c β_c · m_k[c]
```

Ridge regression, chọn `α` bằng CV trên dev split (dùng chính split `Dev` chính thức của UIT-ViQuAD 2.0 — không tự cắt split mới).

### 2.6 Kiểm tra ổn định label

Không đổi so với v1: bootstrap resample mask, tính CI cho `β_c`, gắn cờ `low_confidence` cho document có CI quá rộng.

---

## 3. Giai đoạn B — Tách content-signal khỏi position-signal

Không đổi so với v1:

```
β_c = γ_0 + γ_1 · g(i_c, L) + residual_c
```

`g(i_c, L)` dùng đúng công thức trong `PCS_METHOD_SPEC.md §2`, `i_c` là vị trí chunk trong document đã dựng ở §2.2 (không phải vị trí trong bài Wikipedia gốc — vì document dài là chuỗi haystack+needle mới dựng, vị trí phải tính trên chuỗi này).

Giữ lại 2 phiên bản label: **thô** (`β_c`) và **residual** (`residual_c`) — train song song 2 classifier.

---

## 4. Giai đoạn C — Train classifier

Không đổi so với v1: tái dùng kiến trúc E6, loss MSE trên label chuẩn hoá per-document, train 2 checkpoint (`relevance_v2` từ label thô, `relevance_v3` từ label residual).

**Train/dev split:** dùng đúng split chính thức `Train`/`Dev` của UIT-ViQuAD 2.0 (28.457 / ~5.700 câu hỏi) — không tự cắt lại tỉ lệ khác, giữ được khả năng so sánh với các công bố khác từng dùng chính dataset này.

---

## 5. Train/dev/test discipline

- Giai đoạn A chỉ chạy trên **Train split chính thức của UIT-ViQuAD 2.0**.
- Dev split chính thức của UIT-ViQuAD 2.0 dùng để: chọn `α` ridge (§2.5), sau này chọn `λ` khi cắm vào PCS (§6).
- **`vcc_bench_v2.json` chỉ dùng đúng một lần** cho con số chốt cuối cùng qua PCS sweep — không chạm vào trong Giai đoạn A/B/C (đúng giả định §0).
- **Public test split của UIT-ViQuAD 2.0** (~3.821 câu hỏi) — không dùng trong pipeline này, giữ nguyên chưa động tới, để dành cho việc kiểm tra chéo/tái lập độc lập sau này nếu cần (ví dụ nếu có phản biện paper yêu cầu).

---

## 6. Kết nối lại với PCS đã có

Không đổi so với v1:

```
score(i) = (1−λ)·relevance_v2(i) + λ·g(i,L)      # rồi lặp lại với relevance_v3(i)
```

Chạy trên **`vcc_bench_v2.json`** (test set nội bộ, giữ nguyên theo §0) — đây là điểm nối giữa nhánh train-trên-public và nhánh báo-cáo-trên-benchmark-nội-bộ. Tái dùng nguyên hạ tầng sweep `λ`, unit test tương đương `λ=1`, document-clustered bootstrap CI.

---

## 7. [CẦN XÁC NHẬN] — Số liệu chi phí

Không đổi so với v1 — vẫn thiếu:

1. Thời gian trung bình/sample khi reader Qwen3-8B generate câu trả lời (batch size 16) — lấy từ log thật, không suy đoán.
2. Với document đã dựng ở §2.2 — số chunk trung bình mỗi document (`M+1`, phụ thuộc `M` chọn ở §2.2) — quyết định `K` hợp lý.

**Coding agent không tự chọn `N` (số document dùng sinh label, giờ giới hạn bởi Train split UIT-ViQuAD 2.0 ≈ tối đa 28.457 câu hỏi, không phải 3.155 document như bản nội bộ cũ — quy mô khả dụng lớn hơn) hay `K` trước khi có 2 số trên.** Chạy pilot nhỏ (`N=10, K=15`) đo lại thời gian thật trước khi scale.

---

## 8. Rủi ro cần nêu trong report

- **License UIT-ViQuAD 2.0** cần xác nhận trước khi công bố (§1) — không phải rủi ro kỹ thuật nhưng là rủi ro publication.
- **Domain shift:** UIT-ViQuAD 2.0 là Wikipedia (bách khoa), trong khi `vcc_bench_v2.json` có thể thuộc domain khác — nếu khác domain, `relevance_v2/v3` học trên Wikipedia có thể không tổng quát hoá tốt sang domain test. Đây tự nó là một biến số cần bàn trong Discussion, không phải lỗi cần sửa — nhưng phải nêu rõ, không được ỉm đi.
- **Haystack tự dựng (§2.2) có thể dễ/khó hơn `needle_in_haystack` nội bộ** nếu cách chọn nhiễu không khớp — đây là lý do §2.2 đánh dấu [CẦN XÁC NHẬN], không phải chi tiết phụ.
- E4 selectivity −0.877 (`WAVE4_REPORT.md`) và departure khỏi trục resource+negative-result — không đổi so với v1, vẫn cần nêu.
- Không claim equivalence (TOST) cho bất kỳ so sánh nào trong scope này.

---

## 9. Checklist thực thi cho coding agent

- [ ] Xác nhận license UIT-ViQuAD 2.0 cho phép dùng trong nghiên cứu công bố (§1)
- [ ] Tải `taidng/UIT-ViQuAD2.0`, dùng đúng split Train/Dev/Public-test chính thức
- [ ] Xác nhận đơn vị chọn của E6 (token/chunk) trước khi code (§2.1)
- [ ] Đọc code/config task `needle_in_haystack` nội bộ hiện có, xác nhận cách dựng nhiễu để áp đúng logic lên UIT-ViQuAD (§2.2) — không tự chọn cách dựng nhiễu
- [ ] Viết + chạy script kiểm tra giao giữa UIT-ViQuAD và 14 document `vcc_bench_v2.json`, log kết quả (§1)
- [ ] Lấy 2 số liệu chi phí còn thiếu (§7), chạy pilot `N=10, K=15` trước khi scale
- [ ] Implement dựng document dài (needle+haystack) từ UIT-ViQuAD (§2.2)
- [ ] Implement sinh mask, ridge regression `β_c`, bootstrap kiểm tra ổn định (§2.3–2.6) — logic không đổi so với v1
- [ ] Implement tách `γ_1·g(i_c,L)` khỏi `β_c`, xuất label thô + residual (§3)
- [ ] Train 2 checkpoint (`relevance_v2`, `relevance_v3`) trên Train/Dev split chính thức của UIT-ViQuAD 2.0 (§4)
- [ ] Cắm vào công thức PCS, chạy sweep `λ` trên **`vcc_bench_v2.json`** (test set nội bộ, không đổi) cho cả hai checkpoint (§6)
- [ ] Ghi log: kết quả kiểm tra giao dữ liệu, số liệu chi phí thật, tỉ lệ document flag low-confidence, hai đường cong `λ` so với hai đầu mút đã có trên VCC-Bench
