# Đăng ký trước: lặp lại kết quả chấm điểm theo câu trên tài liệu mới

**Cố định:** 2026-09-30, trước khi bất kỳ tài liệu nào dưới đây được đánh giá. Commit chứa file này là mốc thời
gian; mọi thay đổi sau lần chạy phải ghi rõ ở cuối file, không sửa phần bên trên.
**Bản máy đọc được:** `docs/prereg_sentence_replication.json` (báo cáo đọc nó qua `evaluate.py report --prereg`).

## 1. Vì sao cần lặp lại

Trong đợt follow-up ngày 30/09 (`reports/FOLLOWUP_PROGRESS_REPORT.md`, `paper/main.tex` §Thí nghiệm bổ sung),
`ours_sent` (bộ tỉa `pruner_beta_primary` đã chưng cất trên đoạn, dùng để chấm điểm từng câu, không huấn luyện lại)
đạt, ở 8× với reader Qwen3-8B trên 500 tài liệu test đầu của mỗi nguồn:

| nguồn | `ours_sent` − `exit` | `ours_sent` − `reranker_sent` |
|---|---|---|
| VIMQA | +0.019 (SE ≈ 0.013) | +0.027 (SE ≈ 0.013) |
| HotpotQA | −0.012 (SE ≈ 0.016) | +0.041 (SE ≈ 0.017) |
| 2Wiki | +0.092 (SE ≈ 0.024) | +0.102 (SE ≈ 0.025) |

Kết quả đó là khám phá: thiết kế sau khi thấy kết quả chính, đánh giá trên chính các tài liệu đã dùng để chẩn
đoán, và so sánh trong một bảng 3008 phép thử. Lần lặp lại này kiểm định cùng các khẳng định với giả thuyết, biên,
tài liệu và phép thử cố định trước.

## 2. Tài liệu

- Split `test`, các nguồn nhiều bước VIMQA, HotpotQA, 2Wiki.
- Mỗi nguồn: các tài liệu xếp hạng 501–2500 theo thứ tự băm (`select --n 2000 --offset 500`), tức **không tài liệu
  nào** trong 500 tài liệu của lần chạy chính và đợt follow-up. Thứ tự băm cố định từ trước (`sources.take_n`).
- Số tài liệu: HotpotQA 2000, 2Wiki 2000, VIMQA 199 (tập test của VIMQA chỉ có 699 tài liệu).
- Không có tài liệu nào trong số này được dùng để huấn luyện, chọn epoch hay chọn α.

## 3. Nhánh (không huấn luyện lại; bộ tỉa của lần chạy chính)

| nhánh | spec |
|---|---|
| `full` | toàn ngữ cảnh (tham chiếu) |
| `ours_beta` | `pruner:…/pruner_beta_primary` |
| `ours_sent` | `sent+pruner:…/pruner_beta_primary` |
| `ours_sent_s1`, `ours_sent_s2` | `sent+pruner:…/pruner_beta_primary_s1`, `_s2` (kiểm "mọi seed đồng ý") |
| `reranker_sent` | `sent+reranker:BAAI/bge-reranker-v2-m3` |
| `span_sent` | `sent+pruner:…/pruner_span` (mô tả, không kiểm định) |
| `exit` | EXIT, như lần chạy chính |

Ngân sách, bộ đếm token, quy tắc chọn câu, prompt và reader giống hệt lần chạy chính. Tỉ lệ 4× và 8×; chỉ 8× được
kiểm định. Reader: Qwen3-8B, Qwen3-1.7B, Llama-SEA-LION-v3-8B, Qwen3-32B.

## 4. Giả thuyết và phép thử

Mỗi họ: bootstrap cụm ghép cặp một phía (5000 lần, như lần chạy chính), Holm trong họ, α = 0.05.

| họ | reader | khẳng định | phép thử | ô |
|---|---|---|---|---|
| **R1** | Qwen3-8B | `ours_sent` không kém `exit` quá 0.02 F1 | không-kém-hơn, Δ > −0.02 | VIMQA, HotpotQA, 2Wiki × 8× |
| **R2** | Qwen3-8B | `ours_sent` > `reranker_sent` | vượt trội, Δ > 0 | VIMQA, HotpotQA, 2Wiki × 8× |
| **R3** | Qwen3-32B (không gán nhãn) | như R1 | không-kém-hơn, Δ > −0.02 | VIMQA, HotpotQA, 2Wiki × 8× |

- **Biên 0.02 F1** là biên tương đương đã dùng cho H2b trong các giả thuyết đăng ký trước của bài; không chọn theo
  kết quả khám phá.
- **Mọi seed đồng ý:** một ô chỉ được gọi là bền nếu đạt và `ours_sent_s1`, `ours_sent_s2` cũng cho ước lượng điểm
  thỏa cùng khẳng định.
- Báo cáo mọi ô, đạt hay không. 4×, `span_sent`, `ours_beta` và các reader Qwen3-1.7B, SEA-LION là mô tả.

## 5. Quy tắc diễn giải (cố định trước)

- R1 đạt ở một nguồn → paper được viết "trên nguồn đó, chấm điểm theo câu bằng bộ tỉa không kém EXIT quá 0.02 F1
  ở 1/8 ngân sách, với chi phí thấp hơn khoảng một bậc độ lớn". Nguồn không đạt được báo cáo là chưa kết luận, kèm
  ước lượng và khoảng tin cậy.
- R2 đạt ở một nguồn → "nhãn mức đóng góp vượt độ liên quan ở mức câu" trên nguồn đó.
- R3 là điều kiện để gọi kết quả R1 là không phụ thuộc reader gán nhãn.
- Nếu không họ nào đạt, kết quả của đợt follow-up giữ vai trò khám phá trong paper, và lần lặp lại được báo cáo
  như một kết quả âm.

## 6. Dự đoán và power (ước lượng trước khi chạy)

Nếu các hiệu ứng khám phá ở mục 1 là thật (thường là lạc quan: ước lượng khám phá có xu hướng bị thổi phồng), với
SE co theo 1/√n và ngưỡng một phía từ z = 1.645 (không Holm) đến z = 2.13 (Holm với 3 phép thử):

| nguồn | n | R1 | R2 |
|---|---|---|---|
| VIMQA | 199 | ~0.4–0.6 | ~0.2–0.4 |
| HotpotQA | 2000 | ~0.1–0.25 | ~1.0 |
| 2Wiki | 2000 | ~1.0 | ~1.0 |

Tức là dự đoán: R1 đạt trên 2Wiki, khó đạt trên HotpotQA (ước lượng khám phá −0.012 đã gần biên), không chắc trên
VIMQA vì cỡ mẫu nhỏ; R2 đạt trên HotpotQA và 2Wiki.

## 7. Giới hạn đã biết

- VIMQA chỉ còn 199 tài liệu test mới. 400 tài liệu dev chưa dùng (xếp hạng 301–700) có thể bổ sung như một phân tích
  mô tả tách riêng (`--split dev --offset 300`); chúng không thuộc các họ ở trên.
- `ours_sent` dùng bộ tỉa được chưng cất trên đoạn; nhãn đo ở mức câu là ablation `units` (đang chạy), không thuộc
  lần lặp lại này.

## 8. Thay đổi sau khi chạy

(chưa có)

## 9. Phụ lục, cố định trước khi chạy (2026-10-01)

Thêm năm họ, viết **trước khi** lần lặp lại (mục 2) chạy, và trước khi ba nhánh mới dưới đây được đánh giá trên bất
kỳ tài liệu nào. Phụ lục này được viết sau khi đã thấy kết quả của lần chạy chính, đợt follow-up và ablation `units`
trên 500 tài liệu test đầu, nên các họ dưới đây chỉ được xác nhận bởi các tài liệu 501–2500. R1–R3, tài liệu, phép
thử và mục 5 giữ nguyên; Holm tính riêng trong từng họ, nên thêm họ không làm đổi R1–R3.

### 9.1. Nhánh mới (không đo thêm nhãn)

| nhánh | spec | là gì |
|---|---|---|
| `reranker` | `reranker:@BACKBONE@` | xương sống chưa huấn luyện, chọn theo đoạn (như lần chạy chính) |
| `ours_beta_s1`, `ours_beta_s2` | `pruner:…/pruner_beta_primary_s1`, `_s2` | kiểm "mọi seed đồng ý" cho R4 |
| `span_ans_sent` | `sent+pruner:…/pruner_span_ans` | đối chứng không cần gán tay, chấm theo câu |
| `fuse_sent` | `sent+rrf:pruner:…/pruner_beta_primary\|pruner:…/pruner_span` | gộp thứ hạng của `ours_sent` và `span_sent` |
| `xprovence` (tùy chọn) | `provence:naver/xprovence-reranker-bgem3-v1` | chỉ cho R4b, xem dưới |

- **`pruner_span_ans`**: cùng xương sống, cùng tập tài liệu huấn luyện và dev với `pruner_beta_primary` (cùng thư mục
  fit, bỏ cùng các tài liệu không mang thông tin), cùng số epoch và siêu tham số với `pruner_span`, hạt giống 0. Nhãn:
  1 nếu đoạn chứa chuỗi đáp án (`sources.contains_answer`, đúng phép thử của `oracle_span`), 0 nếu không; loss BCE.
  Tài liệu không có đoạn nào chứa chuỗi đáp án bị bỏ (không có nhãn dương). Chọn epoch theo recall của các đoạn chứa
  đáp án trên dev. Không dùng câu hỗ trợ gán tay ở bất kỳ bước nào. Huấn luyện một lần, không chỉnh siêu tham số. Lệnh:
  `ROUND2_ARMS=1 STAGES=train bash run_pipeline.sh`.
- **`fuse_sent`**: reciprocal rank fusion (Cormack et al., SIGIR 2009) của hai bộ tỉa của lần chạy chính, mỗi bộ chấm
  từng câu như `ours_sent` / `span_sent`. Câu $i$ nhận $\sum_j 1/(k + \mathrm{rank}_j(i))$ với $k = 60$ (giá trị của bài
  gốc, không chỉnh), rồi chọn câu trong ngân sách như `ours_sent`.
- **R4b** cần XProvence, mà bản sửa để XProvence chạy được trên transformers 5.x chưa có trong repo. Nếu nhánh
  `xprovence` không chạy trong lần lặp lại này, R4b được báo cáo là "không chạy". Quyết định chạy hay không được đưa ra
  trước khi có kết quả, và không phụ thuộc vào kết quả.

### 9.2. Họ mới

Cùng giao thức như mục 4: bootstrap cụm ghép cặp một phía (5000 lần), Holm trong họ, α = 0.05; tài liệu 501–2500 của
VIMQA, HotpotQA và 2Wiki; reader Qwen3-8B.

| họ | khẳng định | phép thử | tỉ lệ | ô |
|---|---|---|---|---|
| **R4** | `ours_beta` > `reranker`: nhãn mức đóng góp vượt độ liên quan của chính xương sống, chọn theo đoạn | vượt trội | 4×, 8× | 6 |
| **R4b** | `ours_beta` > `xprovence` (cùng họ khởi tạo), chọn theo đoạn | vượt trội | 4×, 8× | 6 |
| **R5** | `ours_sent` > `span_ans_sent`: β vượt nhãn "đoạn chứa đáp án" cũng không cần gán tay | vượt trội | 8× | 3 |
| **R6** | `ours_sent` không kém `span_sent` quá 0.02 F1: β không cần gán câu hỗ trợ mà gần ngang giám sát bằng câu hỗ trợ | không-kém-hơn, Δ > −0.02 | 8× | 3 |
| **R7** | `fuse_sent` > `span_sent`: β mang thông tin bổ sung cho nhãn câu hỗ trợ | vượt trội | 8× | 3 |

- "Mọi seed đồng ý" dùng `ours_beta_s1/_s2` (R4, R4b) và `ours_sent_s1/_s2` (R5, R6); R7 không có seed.
- Biên 0.02 của R6 là biên của H2b và R1, không chọn theo dữ liệu.
- Mô tả, không kiểm định: 4× của R5–R7, ba reader còn lại, và các nhánh mới trên 500 tài liệu đầu (chạy cùng đợt,
  `ROUND2_ARMS=1`).

### 9.3. Quy tắc diễn giải (cố định trước)

- H1 như đã đăng ký vẫn được báo cáo là không đạt (74/120). R4 không thay H1; R4 đạt ở một nguồn thì paper được viết
  "trên nguồn đó, bộ tỉa vượt xương sống chưa huấn luyện của nó trên tài liệu chưa dùng (lặp lại đăng ký trước)".
  Tương tự với R4b và XProvence.
- R5 đạt ở một nguồn: "β vượt nhãn chứa đáp án, vốn cũng không cần gán tay" trên nguồn đó. R6 đạt ở một nguồn:
  "β không kém giám sát bằng câu hỗ trợ quá 0.02 F1" trên nguồn đó. Chỉ khi cả R5 và R6 đạt ở cùng một nguồn mới
  được viết "β thay được nhãn câu hỗ trợ gán tay" cho nguồn đó.
- R7 đạt ở một nguồn: "β mang thông tin bổ sung cho nhãn câu hỗ trợ" trên nguồn đó; không đạt thì không có bằng chứng
  về tính bổ sung. Không họ nào ở đây thay đổi kết luận của H2a trên lần chạy chính.
- Mọi ô được báo cáo, đạt hay không, kèm ước lượng và khoảng tin cậy.

### 9.4. Dự đoán và power (trước khi chạy)

Từ các so sánh ghép cặp của lần chạy chính (Qwen3-8B, 500 tài liệu; thường bị thổi phồng), SE co theo 1/√n, ngưỡng một
phía từ z = 1.645 tới ngưỡng Holm đầu tiên:

| họ | VIMQA (n = 199) | HotpotQA (n = 2000) | 2Wiki (n = 2000) |
|---|---|---|---|
| R4, 4× / 8× (Δ khám phá +0.048 / +0.058, +0.054 / +0.134, +0.080 / +0.035) | ~0.5–0.75 / ~0.65–0.9 | ~1.0 / ~1.0 | ~1.0 / ~1.0 |
| R4b, 4× / 8× (+0.055 / +0.060, +0.058 / +0.109, +0.070 / +0.026) | ~0.6–0.85 / ~0.7–0.9 | ~1.0 / ~1.0 | ~1.0 / ~0.8–0.95 |
| R6, 8× (Δ khám phá −0.002, −0.019, +0.003) | ~0.15–0.3 | ~0.02–0.07 | ~0.55–0.75 |

Dự đoán: R4 và R4b đạt trên HotpotQA và 2Wiki, không chắc trên VIMQA; R6 có thể đạt trên 2Wiki, gần như chắc chắn không
đạt trên HotpotQA (ước lượng khám phá −0.019 nằm sát biên), thiếu power trên VIMQA. R5 và R7 chưa có ước lượng nào:
`pruner_span_ans` và `fuse_sent` chưa từng được đánh giá. Ở mức đoạn, `oracle_span` (giữ đoạn chứa đáp án, biết đáp án
lúc test) vượt `ours_beta` trên HotpotQA và 2Wiki, nên R5 không chắc đạt; tập câu do `ours_beta` và `span_sup` chọn
trùng Jaccard 0.47–0.55 (ablation `units`), là lý do để thử R7, không phải ước lượng của nó.
