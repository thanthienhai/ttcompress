# Position-Calibrated Selection (PCS) — Method Spec (v2, đã khớp code thật)

> **Trạng thái:** v2 — đã thay số giả định bằng số thật từ `TruncationCompressor` (`vncompress/compression.py:322-376`). Còn **2 mục [CẦN XÁC NHẬN]** trước khi chạy số báo cáo — xem §4 và §10.
> **Ngữ cảnh dự án:** LACC/VNCompress, trục paper = resource + negative-result, VCC-Bench là đóng góp chính. Xem `WAVE4_REPORT.md` để có toàn bộ dữ liệu nền (số liệu E4/E6/tone-probe/truncation đã có).
> **Ràng buộc cứng, không được vi phạm khi implement:**
> 1. Bolt-on only — **không** train lại E6 encoder. Chỉ dùng checkpoint đã có, forward-pass để lấy điểm relevance.
> 2. Toàn bộ cơ chế mới phải **query-agnostic** (không đọc câu hỏi ở bất kỳ bước nào).
> 3. `λ` là **một scalar duy nhất, toàn cục** — không được fit riêng theo task/ratio.
> 4. Không claim equivalence (TOST) cho bất kỳ so sánh nào trong scope này.

---

## 0. Đổi khung so với v1 — vì sao

Ở v1, PCS là công thức cộng: `s_i = p_i + λ·g(i,L)`, và `truncation` được coi là một baseline **ngoài**, tách biệt.

Với số thật vừa xác nhận, `g(i,L)` khớp **chính xác** cách `truncation` chọn token (xem §2). Điều này cho phép viết lại thành **một họ điểm số duy nhất**, trong đó `encoder` (E6 gốc) và `truncation` là hai đầu mút của cùng một tham số:

```
score(i) = (1 − λ)·relevance(i) + λ·g(i, L),   λ ∈ [0, 1]
```

- **λ = 0** → `score(i) = relevance(i)` → **đúng arm `encoder` đã có**, không đổi gì.
- **λ = 1** → `score(i) = g(i, L)` → **đúng arm `truncation` đã có**, tái tạo token-for-token (đã kiểm chứng ở §2).
- **0 < λ < 1** → vùng mới, chưa ai đo — đây là phần PCS thật sự đóng góp.

Nghĩa quan trọng cho paper: **`truncation` không còn là baseline rời rạc, mà là trường hợp riêng (λ=1) của cùng một công thức.** Đây là điểm nên nêu trong Method section — không phải "chúng tôi so với truncation", mà "truncation là điểm cực trị của không gian tham số chúng tôi quét".

---

## 1. Giả thuyết (pre-registered, không đổi so với v1)

Ưu thế của `truncation` so với mọi compressor ngữ nghĩa đã thử (E6, `lacc_tone`, `lacc_ppl_*`, `lacc_classprop`, ...) không đến từ thiếu tín hiệu ngữ nghĩa tốt, mà từ việc **vị trí token trong tài liệu** mang phần lớn thông tin hữu ích cho task `needle_in_haystack`.

Với khung mới (§0), giả thuyết trở thành một câu hỏi cụ thể hơn: **quét `λ` từ 0 → 1, token_f1 có tăng đơn điệu (nghĩa là càng thiên vị trí càng tốt, đỉnh nằm tại λ=1) hay đạt đỉnh ở đâu đó giữa hai đầu mút (nghĩa là kết hợp semantic + position tốt hơn cả hai đầu mút riêng lẻ)?**

Ba kết cục vẫn đều hợp lệ cho paper:

- **Đỉnh tại hoặc gần λ=1** → phần semantic của E6 không cộng thêm giá trị ngoài vị trí; củng cố luận điểm trung tâm.
- **Đỉnh ở giữa (0<λ<1), rõ ràng cao hơn cả hai đầu mút** → phát hiện dương tính thật sự: kết hợp đúng cách thắng cả truncation lẫn E6 thuần — đây sẽ là kết quả mạnh nhất có thể có từ dự án.
- **Đỉnh tại λ=0 hoặc đường cong phẳng/nhiễu** → semantic của E6 vẫn vô dụng dù kết hợp thế nào — kết quả âm mạnh hơn nữa.
- **H2O/SnapKV cũng thua điểm λ=1** → phát hiện lan sang họ attention-optimization (cache-level), lấp đúng gap mà `WAVE4_REPORT.md §6.3` đã tự flag.

---

## 2. Định nghĩa `g(i, L)` — đã khớp code thật, có kiểm chứng

**Nguồn xác nhận:** `TruncationCompressor`, `vncompress/compression.py:322-376`.

```python
keep_k = ceil(n / ratio)          # ratio = original/compressed (QUY ƯỚC REPO — xem cảnh báo dưới)
head_k = keep_k // 2              # floor
tail_k = keep_k - head_k          # tail nhận token lẻ ⇒ tail_k ≥ head_k, chênh ≤ 1
kept   = ids[:head_k] + ids[n - tail_k:]
```

Đặt khoảng cách tới đầu gần nhất: `d(i) = min(i, L−1−i)`. `truncation` chính là **top-`keep_k` theo `d` nhỏ nhất**. Chuẩn hoá thành điểm trong `[0,1]`:

```
g(i, L) = 1 − 2·min(i, L−1−i) / (L−1)
```

- `g = 1` ở hai đầu (`i=0`, `i=L−1`), `g = 0` ở giữa, hình chữ U đối xứng.
- **Đã kiểm chứng khớp token-for-token:** head đóng góp các `d ∈ {0, ..., head_k−1}`, tail đóng góp `d ∈ {0, ..., tail_k−1}`; khi `keep_k` lẻ, `tail_k = head_k + 1` nên token dư rơi vào tail — đúng thứ tự tie-break của code gốc.
- **Không còn tham số `r*`, `w` cần đoán** — mục [CẦN XÁC NHẬN #1] của v1 đã đóng.

### ⚠️ Cảnh báo bắt buộc — quy ước `ratio`

`ratio` trong repo = **original/compressed** (dòng 336, 339 của `compression.py`). "8x" nghĩa là `ratio=8`, giữ 1/8. **Không được đảo ngược** ở bất kỳ chỗ nào tính `keep_k`, `g`, hay adaptive-budget — đảo ngược sẽ làm sai toàn bộ so sánh mà không báo lỗi (silent bug).

### ⚠️ An toàn implementation cho tie-break

`g(i,L)` là số thực — khi chọn top-k tổng quát bằng `argsort`/`topk` chuẩn, **ties có thể vỡ khác quy tắc "nghiêng tail" của code gốc**, đặc biệt quanh biên giữa vùng head/tail. Để đảm bảo λ=1 tái tạo **chính xác** arm `truncation` đã có (đây là unit test bắt buộc, không phải tuỳ chọn):

- **Không** dùng `g(i,L)` tính bằng floating-point rồi top-k tổng quát cho trường hợp λ=1.
- Viết `score_to_rank_key(i) = (d(i), tie_break)` với `tie_break` ưu tiên chỉ số lớn hơn (nghiêng tail) khi `d` bằng nhau, hoặc đơn giản hơn: **tại λ=1, gọi thẳng lại hàm `TruncationCompressor` có sẵn thay vì đi qua top-k tổng quát**, để loại rủi ro lệch do làm tròn số thực.
- Unit test bắt buộc: `score(i, λ=1)` phải cho ra đúng tập token mà `TruncationCompressor` chọn, với mọi `n`, `ratio` đã test trong benchmark hiện có (đối chiếu trực tiếp, không chỉ kiểm tra "gần giống").

---

## 3. `relevance(i)` — không đổi, vẫn 100% tái dùng

`relevance(i) = p_i = σ(f_θ(x_i))`, lấy thẳng từ checkpoint E6 hiện có (`models/encoder_compressor/`). Chỉ **inference**, không forward-pass với gradient, không train lại.

---

## 4. [CẦN XÁC NHẬN] — Đơn vị chọn của E6: token hay câu/chunk?

`g(i,L)` ở §2 định nghĩa theo **token index**. Nếu E6 chọn theo **câu/chunk** (không phải token), cần biết trước khi code:

- Nếu E6 vốn hoạt động ở token-level → dùng thẳng `g(i,L)` như trên, không cần đổi gì.
- Nếu E6 hoạt động ở sentence/chunk-level → `g` cho một chunk/span cần tính theo **khoảng cách nhỏ nhất từ hai đầu span tới hai đầu văn bản** (hoặc midpoint của span) — công thức không đổi, chỉ đổi đơn vị `i, L` từ token sang chunk. Cần xác nhận quy ước này trước khi implement để không tạo ra một biến thể "PCS-chunk" khác PCS-token mà không ai chủ đích thiết kế.

Coding agent: kiểm tra trực tiếp interface của `encoder` arm (input/output shape) trong repo trước khi code phần trộn điểm — nếu không rõ, dừng lại hỏi thay vì đoán.

---

## 5. Bảo toàn ràng buộc dự án

| Ràng buộc | Cách tuân thủ |
|---|---|
| Query-agnostic | `g(i,L)` chỉ phụ thuộc vị trí, không nhìn câu hỏi. `relevance(i)` từ E6 vốn query-agnostic. |
| Bolt-on, không train-from-scratch | 0 GPU-hour train encoder mới. Chi phí chỉ là inference đã có (tái dùng) + quét `λ` (không cần GPU, hoặc rất ít). |
| Resource + negative-result | Mọi vị trí đỉnh của đường cong λ (kể cả tại 0 hoặc 1) đều là finding hợp lệ. |

---

## 6. Phạm vi thực nghiệm

| Arm | Task | Ratio | Trạng thái |
|---|---|---|---|
| `encoder` (λ=0, đã có) | `needle_in_haystack` | 4x, 8x | Đã có — không chạy lại, chỉ dùng làm điểm neo của đường cong |
| `truncation` (λ=1, đã có) | `needle_in_haystack` | 4x, 8x | Đã có — không chạy lại, dùng làm điểm neo còn lại |
| **`encoder_pcs`** (0<λ<1, mới) | `needle_in_haystack` | 4x, 8x | Cần code: quét `λ`, bench từng điểm |
| `h2o` | `needle_in_haystack` | 8x | Đã đăng ký, chưa từng chạy — xem §10 |
| `snapkv` | `needle_in_haystack` | 8x | Như trên |

**Lưới quét `λ`:** `{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}` — 9 điểm nội suy, cộng 2 điểm neo có sẵn (0 và 1) = 11 điểm cho toàn đường cong. Chạy đủ cả lưới trên **tập nhỏ riêng (dev split)** trước, báo cáo:

1. Đường cong đầy đủ token_f1 theo `λ` (0→1) trên dev — cho thấy hình dạng thật của tradeoff, không chỉ 1 điểm.
2. Chọn **một** `λ` duy nhất (đỉnh đường cong dev, hoặc điểm gần đỉnh nhất trong lưới) để chạy chính thức trên test set VCC-Bench và báo cáo CI — đây là con số "chốt" cho paper, tránh chọn theo test set (leak).

**Cố tình loại khỏi scope này** (không đổi so với v1):

- Không chạy 2x — ở 2x "almost everything survives" (§10.2 `WAVE4_REPORT.md`), không có gì để đo.
- Không chạy H2O/SnapKV ở 4x — chỉ 8x để giữ ngân sách compute (`output_attentions=True` trên reader 8B là O(S²)/layer).
- Không mở rộng sang 5-task matrix — chỉ `needle_in_haystack`, task duy nhất phân tách được arm rõ ràng.
- Không tune `λ` riêng theo task/ratio — một giá trị toàn cục duy nhất cho con số báo cáo chính; đường cong quét trên dev là để hiểu hình dạng, không phải để có nhiều `λ` khác nhau cho từng ratio.

---

## 7. Giao thức thống kê

- Dùng lại **document-clustered bootstrap đã fix ở `WAVE4_REPORT.md §9`** — import function có sẵn, không viết lại.
- Chỉ báo cáo **point estimate + 95% CI** cho `encoder_pcs` tại `λ` đã chọn. **Không** chạy TOST/equivalence trong scope này.
- Quét `λ` để chọn đỉnh chạy trên **dev split tách riêng khỏi test set VCC-Bench** — tránh lặp lỗi contamination đã phát hiện ở `WAVE4_REPORT.md §4.3`.

---

## 8. Tiêu chí thành/bại (chốt trước khi chạy, không đổi sau khi thấy số)

- **Đỉnh đường cong nằm có ý nghĩa thống kê trên cả hai đầu mút** (CI của đỉnh không phủ CI của cả `encoder` lẫn `truncation`) → phát hiện dương tính rõ.
- **Đỉnh đường cong ≈ λ=1 (trong sai số CI)** → semantic của E6 không đóng góp; củng cố luận điểm trung tâm.
- **Đường cong phẳng/nhiễu, không đỉnh rõ** → cả relevance lẫn kết hợp đều không giúp gì so với truncation thuần; kết quả âm mạnh nhất.
- **H2O/SnapKV:** so trực tiếp với điểm `λ=1` (`truncation`) bằng cùng chuẩn CI — không cần thắng, chỉ cần có số.

---

## 9. Ước lượng chi phí

| Bước | Chi phí ước tính |
|---|---|
| Quét 9 điểm `λ` trên dev split | Không cần GPU mới — chỉ trộn điểm đã có sẵn từ inference E6 (đã chạy) với `g(i,L)` (tính tay), rồi chọn top-k. Chi phí là CPU + thời gian tính CI, không phải training |
| Bench `encoder_pcs` (λ đã chọn, needle, 2 ratio × 120 sample) | Tương đương chi phí bench `encoder` hiện có × 1 lượt |
| Unit test tương đương λ=1 với `truncation` | Vài phút, bắt buộc trước khi tin bất kỳ số nào khác |
| H2O/SnapKV (needle @ 8x, 120 sample, `output_attentions=True` trên reader 8B) | Khoản đắt nhất trong scope này — xem §10 |

---

## 10. [CẦN XÁC NHẬN] — H2O/SnapKV: đã có implementation chạy được chưa?

`WAVE4_REPORT.md §6.3` mục 5: các arm này **"registered but deliberately out of the default sweep"**. Cần xác định:

- Có code thật thực thi H2O/SnapKV eviction logic chưa, hay mới là tên trong registry chưa nối pipeline?
- Nếu chưa có: việc riêng cần làm trước, coding agent báo cáo lại trạng thái thật trước khi ước lượng thời gian.
- Nếu đã có: chạy scoped đúng §6 (needle @ 8x, không mở rộng).

---

## 11. Checklist thực thi cho coding agent

- [ ] Xác nhận đơn vị chọn của E6 (token hay chunk) — §4 — dừng hỏi nếu không rõ, không tự đoán
- [ ] Implement `g(i,L) = 1 − 2·min(i, L−1−i)/(L−1)` là hàm thuần, không tham số học
- [ ] Implement `score(i) = (1−λ)·relevance(i) + λ·g(i,L)`, tái dùng `relevance(i)` từ checkpoint E6 có sẵn (chỉ inference)
- [ ] Viết unit test: `score(i, λ=1)` → chọn đúng tập token của `TruncationCompressor` hiện có, mọi `n`/`ratio` đã test — chạy PASS trước khi tin số khác
- [ ] Implement quét `λ` trên 9 điểm nội bộ + 2 điểm neo (encoder, truncation) trên dev split riêng
- [ ] Vẽ/log đường cong token_f1 theo `λ` trên dev — chọn 1 `λ` cho bench chính thức
- [ ] Đăng ký arm `encoder_pcs` (λ đã chọn) vào registry, dùng lại luật chọn top-K đã có
- [ ] Kiểm tra trạng thái thật của H2O/SnapKV implementation (§10) trước khi ước lượng thời gian chạy
- [ ] Chạy bench chính thức: `encoder_pcs` @ {4x, 8x}; `h2o`, `snapkv` @ {8x} — task `needle_in_haystack` duy nhất, test set VCC-Bench
- [ ] Tính CI bằng document-clustered bootstrap đã fix (`WAVE4_REPORT.md §9`), không claim TOST
- [ ] Ghi log đầy đủ: đường cong quét λ trên dev, `λ` cuối cùng chọn, kết quả unit test tương đương λ=1, và xác nhận quy ước `ratio` không bị đảo ngược ở đâu trong code
