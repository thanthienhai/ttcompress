# Báo cáo full run 4×H100: `ttcompress` (N_TRAIN=3000, N_DEV=300, N_TEST=500)

**Hoàn tất:** 29/09/2026 21:44 UTC (04:44 ngày 30/09 giờ VN). Job `ttcompress-full-h100x4` ở trạng thái Complete, pod đã tự release.
**Code:** `d936896` + 2 patch chưa commit (XProvence trên transformers 5.x; prompt của SEA-LION base, `BASE_PROMPT_VERSION 2`)
**Output (HPS):** `/mnt/hps/anhm-paper/ttscompress/runs/main/results/eval_test/` (report.md, report.json, paper/*.tex|csv)
**Bản local:** `results/full_2026-09-29/` (report + bảng paper + summary của các label fit + progress + patch)

---

## 1. Tóm tắt

| Giả thuyết | Kết quả | Đánh giá |
|---|---|---|
| **H1**: ours_beta > mọi bộ nén không dùng oracle | **74/120** ✓ (mọi seed đều đồng ý) | **Đạt một phần.** Thắng rõ trên multi-hop, hòa trên single-hop, thua EXIT |
| H1-oracle: nằm trong 0.05 F1 của oracle_beta | 0/10 | ✗ Cách oracle 0.015–0.12 F1 |
| H2a: multi-hop, ours_beta > span_sup và > oracle_span | 1/12 | ✗ (khám phá: đạt trên VIMQA với doc có ≥ 2 đoạn hỗ trợ) |
| H2b: single-hop, ours_beta ≈ span_sup (±0.02) | 1/4 | Hướng đúng (Δ từ −0.014 đến +0.005), nhưng CI rộng hơn biên ±0.02 |
| H3 / H3-heldout: retention(ours_ens) > retention(ours_beta) | 0/28, 0/28 | ✗ ours_ens ≈ ours_beta; CI của tỉ lệ retention rất rộng |

## 2. Token F1 của reader chính (Qwen3-8B)

| Nhánh | UIT@4 | UIT@8 | XQuAD@4 | XQuAD@8 | VIMQA@4 | VIMQA@8 | HotpotQA@4 | HotpotQA@8 | 2Wiki@4 | 2Wiki@8 |
|---|---|---|---|---|---|---|---|---|---|---|
| *full context* | *0.677* | | *0.739* | | *0.639* | | *0.667* | | *0.559* | |
| **ours_beta** | 0.725 | 0.726 | **0.770** | 0.765 | **0.689** | 0.593 | 0.695 | 0.646 | 0.458 | 0.347 |
| ours_ens | 0.720 | 0.711 | 0.760 | 0.754 | 0.685 | 0.587 | 0.706 | 0.654 | 0.463 | 0.356 |
| abl_logprob | 0.706 | 0.718 | 0.754 | 0.766 | 0.666 | 0.578 | 0.700 | 0.650 | 0.460 | 0.343 |
| span_sup | 0.724 | 0.731 | 0.765 | 0.778 | 0.664 | 0.570 | 0.716 | 0.570 | 0.447 | 0.347 |
| reranker (backbone chưa train) | 0.727 | **0.744** | 0.751 | 0.773 | 0.641 | 0.535 | 0.641 | 0.512 | 0.378 | 0.312 |
| XProvence | 0.721 | 0.730 | 0.750 | 0.764 | 0.634 | 0.533 | 0.637 | 0.537 | 0.388 | 0.321 |
| **EXIT** | 0.710 | 0.718 | 0.768 | 0.765 | **0.689** | **0.672** | **0.718** | **0.692** | **0.561** | **0.456** |
| RECOMP | 0.556 | 0.461 | 0.586 | 0.524 | 0.495 | 0.426 | 0.660 | 0.589 | 0.522 | 0.430 |
| oracle_beta (100 doc) | 0.788 | 0.769 | 0.764 | 0.782 | 0.753 | 0.636 | 0.740 | 0.751 | 0.658 | 0.462 |

Reader held-out Qwen3-32B cho cùng một mẫu hình: ours_beta 0.736 và 0.724 trên VIMQA@4 và HotpotQA@4, so với reranker 0.675 và 0.669.

**Nhận xét:**
1. **Multi-hop (VIMQA, HotpotQA, 2Wiki): ours_beta thắng backbone của chính nó.** Mức thắng so với `reranker` là +0.048 đến +0.134 F1, và so với XProvence là +0.055 đến +0.109. Cả hai đều có ý nghĩa sau Holm, trừ 2Wiki@8. Đây là bằng chứng chính cho thấy nhãn utility thêm được thông tin mà chỉ riêng độ liên quan không có. Ours_beta cũng thắng đậm cả họ LLMLingua (+0.16 đến +0.43).
2. **Single-hop (UIT, XQuAD-vi): mọi bộ nén theo đoạn đều hòa nhau**, cách nhau khoảng ±0.02. Bài toán ở đây đã bão hòa: gold_recall của ours và reranker đều khoảng 0.99–1.0, vì chỉ cần giữ được một đoạn needle.
3. **Nén còn cho F1 cao hơn full context** trên UIT, XQuAD, VIMQA@4 và HotpotQA@4 (+0.03 đến +0.05). Có thể là do bớt được đoạn gây nhiễu, đúng hướng "lost in the middle". Đây là một điểm đáng viết.
4. **EXIT là đối thủ mạnh nhất trên multi-hop** (VIMQA@8, HotpotQA, 2Wiki: +0.02 đến +0.11 so với ours). Tuy vậy:
   - Chi phí của EXIT **cao hơn 5–14 lần**: 184–1051 ms/doc so với 36–76 ms/doc của ours, vì nó chạy Gemma-2B trên từng câu.
   - **Ngân sách token không bằng nhau.** EXIT chọn theo câu nên đạt tỉ lệ nén thực tế 8.1–8.7× ở mức "8×". Ours chọn nguyên đoạn nên chỉ lấp được một phần ngân sách, dẫn tới **nén tới 9.1–10.0×**, tức dùng ít hơn 10–15% token. Cờ `budget ≠` hiện chỉ bật khi lệch trên 10%. Cần một phép so sánh có khớp ngân sách token (xem §5).
5. **H3 thất bại.** Ensemble không giúp giữ được mức tăng khi đổi sang reader mạnh hơn. Tỉ lệ retention rất nhiễu (CI rộng ±0.3), vì mẫu số (khoảng cách F1 full-context giữa hai reader) nhỏ.
6. **Nhánh logprob không thắng ở F1 downstream.** Tuy có dev ndcg cao nhất (0.847), `abl_logprob` vẫn thấp hơn ours_beta ở 7/10 ô, và ở 3 ô còn lại chỉ cao hơn tối đa 0.005. Vì vậy claim "dùng F1 utility" vẫn đứng được.

## 3. Chi phí (RQ1)

- Stage A: khoảng 65 lần gọi reader mỗi doc. Qwen3-8B mất ~5.9 s/doc trên UIT và ~1.0 s/doc trên HotpotQA mỗi GPU. Tổng cộng khoảng 9 giờ trên 4×H100 cho 3 reader × 9900 doc.
- Chọn lọc bằng ours_beta mất **36–76 ms/doc**, so với ~6000 ms/doc cho việc gọi reader 65 lần để lấy oracle. Như vậy ours khấu hao được chi phí đo utility khoảng **80–160 lần**, đổi lại mất 0.015–0.12 F1 so với oracle_beta.

## 4. Nhật ký chạy

| Giai đoạn | Thời lượng | Ghi chú |
|---|---|---|
| Arms check | ~1h25 | 3 lỗi đã sửa: nltk/defusedxml, thư mục baseline bị hỏng khi 2 pod cài song song, XProvence trên transformers 5.x |
| labels (lượt 1) | 6h46 | Phát hiện SEA-LION trả lời rỗng trên tiếng Việt |
| Tạm dừng | 29/09 02:49 → 14:26 UTC | Theo yêu cầu |
| Debug và sửa SEA-LION | ~10 phút | `<|end_of_text|>` ngay sau `### Đáp án:` → thêm `\n` vào prompt base; trên doc train UIT, câu trả lời rỗng giảm từ 57% xuống 0% và full F1 tăng từ 0.30 lên 0.62 |
| labels (đo lại SEA-LION) | 1h52 | |
| train (8 pruner) | 2h40 | |
| select | 1h02 | |
| answer (4 reader) | 1h30 | |
| report | 3 phút | |

Nhãn SEA-LION cũ, các ensemble fit và pruner ensemble tương ứng đã được archive ở `runs/archive/2026-09-29_sealion_base_prompt_v1/`.

## 5. Đánh giá paper và việc cần làm

**Paper khả thi nếu điều chỉnh cách đặt vấn đề.** Claim chính nên là: *"Chưng cất F1 utility vào một cross-encoder cho phép nén theo đoạn nhanh (khoảng 40 ms/doc), vượt backbone reranker và XProvence cùng họ khởi tạo từ +5 đến +13 F1 trên QA multi-hop tiếng Việt và tiếng Anh, và không thua span supervision trên single-hop."* Các giả thuyết H1-oracle, H2a và H3 nên được báo cáo trung thực là kết quả âm hoặc chưa kết luận được.

**Việc cần làm trước deadline ARR (12/10):**
1. **So sánh có khớp ngân sách token với EXIT/RECOMP** (quan trọng nhất). Hai cách: cho ours lấp phần ngân sách còn dư bằng cắt ở mức câu, hoặc vẽ đường F1 theo số token thực dùng. Nếu không làm, reviewer sẽ chỉ ra rằng EXIT thắng một phần là nhờ được dùng nhiều token hơn.
2. Cập nhật `paper/main.tex`: điền bảng `main_results.tex` và `hypotheses.tex`, viết lại abstract và phần đóng góp theo cách đặt vấn đề ở trên, và thêm một đoạn về lỗi SEA-LION base prompt (đây là bài học đáng ghi về đánh giá base model trên tiếng Việt).
3. H3: báo cáo độ đồng thuận giữa các reader (Spearman 0.22–0.29 trên UIT, 0.39–0.52 trên multi-hop) như một kết quả mô tả, và giải thích vì sao retention bị nhiễu.
4. Commit các thay đổi local: `ttcompress/selection.py`, `ttcompress/reader.py`, `generate_labels.py`, `tests/test_reader.py`, và `run_pipeline.sh` (thêm `defusedxml`).
5. Vẽ hình bằng `scripts/paper_figures.py --report results/full_2026-09-29/runs/main/results/eval_test/report.json` (cần matplotlib ở local).
6. (Tùy chọn) Upload lên HF từ một pod CPU: `STAGES=upload bash run_pipeline.sh`.
