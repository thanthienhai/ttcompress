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
