# Đề xuất nghiên cứu (Paper Proposal) về Context/Prompt Compression cho COLING 2027

## TL;DR (Tóm tắt cho quyết định)
- **COLING 2027** tổ chức tại Macau (9–14/5/2027), nộp bài **bắt buộc qua ARR** (ACL Rolling Review) chu kỳ tháng 10/2026, **hạn 12/10/2026**; đây là hội nghị hạng **CORE B / CCF B** (KHÔNG phải A) — nhận định "rank B" của bạn là đúng. Long paper 8 trang, short 4 trang. Đây là venue "vừa sức", phù hợp cho một đóng góp chắc chắn thay vì phải cạnh tranh khốc liệt như ACL/EMNLP hạng A.
- Hướng β/Ridge hiện tại **không còn đủ mới** nếu định vị là "context attribution" (đã bị ContextCite + AT2 chiếm chỗ), nhưng công thức cụ thể — *random-mask → Ridge surrogate của F1 downstream theo chunk → distill hệ số thành pseudo-label để huấn luyện một pruner cross-encoder chạy 1 lượt (one forward pass)* — thì **chưa có ai công bố đúng như vậy**. Đây là khe hở novelty thật.
- **Khuyến nghị:** chọn hướng chính **"Amortized Utility Attribution for Compression"** (distill attribution theo utility F1 thành pruner nhanh), ghép với hai góc gia cố là **tiếng Việt/đa ngôn ngữ** và **reader-agnostic (chuyển nhãn giữa các reader)**. Đây là nơi vừa mới, vừa khả thi với 4×H100, vừa tận dụng được thế mạnh benchmark VCC-Bench mà nhóm tự kiểm soát.

---

## Key Findings (Phát hiện chính)

### 1. Thông tin xác thực về venue COLING 2027
- **Địa điểm & ngày:** Macau, Trung Quốc, 9–14/5/2027.
- **Cách nộp:** Bắt buộc qua **ARR (ACL Rolling Review)** — không nộp trực tiếp. Chu kỳ muộn nhất là **ARR tháng 10/2026**. Theo trang chính thức (2027.coling-iccl.org): "ARR submission deadline · Monday · October 12, 2026; Commitment after meta-reviews · Wednesday · December 23, 2026; Notification of acceptance · Wednesday · February 10, 2027." Lưu ý chu kỳ tháng 10 dùng chung cho cả COLING 2027 và NAACL 2027; chỉ chọn hội nghị sau khi có meta-review. Bài đã có review từ chu kỳ ARR trước cũng có thể commit vào COLING.
- **Giới hạn trang (theo quy định ARR/ACL hiện hành):** long paper **8 trang** thân bài, short paper **4 trang** thân bài; phần **Limitations bắt buộc** (không tính vào giới hạn), references và appendix không giới hạn.
- **Xếp hạng thực tế (đã kiểm chứng):** Cổng CORE (portal.core.edu.au) ghi rõ "Source: CORE2023, Rank: B" và "Source: ICORE2026, Rank: B", kèm chú thích "centiles are fairly strong, but less than other A ranked NLP conferences". Bản ghi myhuiban.com: **CCF B, ICORE B, QUALIS A1**, tỷ lệ chấp nhận trung bình ~30,5% qua 5 kỳ. COLING từng là **CORE A trước 2021** rồi bị hạ xuống B. Google Scholar Metrics (mảng Computational Linguistics) xếp COLING **h5-index 81, h5-median 122** (venue thứ 5 trong ngành, sau ACL/EMNLP/NAACL/TACL); mldeadlines.com ghi h5-index 73. → Kết luận: COLING **rank B**, đúng như bạn nghĩ.
- **Chủ đề phù hợp:** CFP liệt kê "Cross-lingual and Multilingual NLP", "Information Extraction and Retrieval", "Interpretability and Analysis", "Question Answering", "Benchmarking and Evaluation" — tất cả đều bao trùm đề tài compression. Theme track đặc biệt của COLING 2027 là **"NLP for Linguistics"** (tập trung đa dạng ngôn ngữ học — KHÔNG liên quan đề tài này, nên đừng nhắm theme track).

### 2. Bối cảnh cạnh tranh (novelty landscape) cập nhật 2025–2026
- **ContextCite (NeurIPS 2024, arXiv 2409.00729):** random ablation ngữ cảnh + surrogate tuyến tính thưa dự đoán logprob câu trả lời; ứng dụng gồm pruning ngữ cảnh. Đây chính là "bản gốc" của phương pháp β của nhóm — nên β **không còn mới** nếu gọi là attribution. Điểm mấu chốt: ContextCite **KHÔNG amortize** — theo chính bài báo, "ContextCite using just 32 context ablations consistently matches or outperforms the baselines ... Increasing the number of context ablations to {64, 128, 256} can further improve the quality", tức mỗi ví dụ cần **32–128 lần ablation lúc inference**. Chi phí này chính là thứ một pruner amortized loại bỏ được.
- **AT2 – "Learning to Attribute with Attention" (arXiv 2504.13752):** **mối đe dọa novelty gần nhất về mặt ý tưởng "amortize"**. AT2 học hệ số theo attention-head để tái tạo attribution từ ablation, chạy 1 forward pass, và đã thử pruning trên HotpotQA. **Khác biệt then chốt:** AT2 amortize vào chính attention-head của reader (không phải một pruner cross-encoder riêng) và dùng **logprob**, KHÔNG dùng **F1**. → Phải nêu rõ khác biệt này trong bài.
- **LooComp (arXiv 2603.09222, KAIST, 2026):** **gần nhất về kiến trúc** — một pruner cross-encoder encoder-only (ModernBERT) dùng leave-one-out. **Khác biệt:** dùng nhãn nhị phân + ranking/BCE và LOO tất định, KHÔNG dùng random-mask + Ridge surrogate của F1.
- **CORE-RAG / "Less Is More" (arXiv 2508.19282, ICML 2026):** compressor sinh (generative) 1.5B huấn luyện bằng RL (GRPO) với reward = EM downstream. Theo abstract: "At a high compression ratio of 3%, CORE not only avoids performance degradation but also improves the average Exact Match (EM) score by 3.3 points compared to using full documents." Cùng tinh thần "tối ưu trực tiếp theo downstream", nhưng cơ chế là **RL reward**, không phải Ridge surrogate distill thành pruner.
- **ECoRAG (arXiv 2506.05167):** huấn luyện compressor theo "evidentiality" (câu có giúp sinh đúng đáp án không) — dùng tín hiệu đáp án đúng nhưng không có random-mask Ridge surrogate.
- **PoC (arXiv 2603.19733):** dự đoán performance để chọn tỷ lệ nén; không tạo attribution theo từng chunk.
- **Provence (ICLR 2025, arXiv 2501.16214) & XProvence (ECIR 2026, arXiv 2601.18886):** cross-encoder reranker + pruning theo sequence labeling; XProvence huấn luyện đa ngôn ngữ 16 thứ tiếng, phủ 100+ ngôn ngữ (gồm tiếng Việt zero-shot) trên bge-reranker-v2-m3. Đây là **baseline đa ngôn ngữ bắt buộc phải so**, và nó làm suy yếu góc "chỉ vì tiếng Việt" thuần túy.
- **EXIT (ACL Findings 2025, arXiv 2412.12559):** nén trích xuất theo câu, adaptive; baseline mạnh cho multi-hop (huấn luyện trên HotpotQA supporting-fact).
- **Các hướng khác đang nóng (rủi ro đụng hàng cao):** submodular/set-level selection (AdaGReS arXiv 2512.25052, GeoRAG, ScalDPP, "What Survives Into Context" arXiv 2607.00725); soft-prompt (gist tokens, ICAE, xRAG, 500xCompressor arXiv 2408.03094, CompLLM arXiv 2509.19228); KV-cache compression (MiniKV); position-bias-aware compression (SeCo arXiv 2605.09463).
- **"Fixed RAG Compression Collapses Measured Reader Scaling" (arXiv 2606.21807):** phát hiện rất quan trọng — một compressor cố định có thể **che giấu tới 80%** mức nâng cấp reader (từ Qwen 7B lên GPT-4.1-mini trên HotpotQA) và **đảo 31%** xếp hạng model trên LongMemEval-S. Nhóm tác giả phát hành toolkit `ragscale`. Đây là **bằng chứng học thuật ủng hộ mạnh** cho góc reader-agnostic — chính điểm yếu số (8) mà nhóm tự nhận (nhãn gắn với 1 reader Qwen3-8B).

### 3. Tài nguyên tiếng Việt (hỗ trợ góc đa ngôn ngữ)
- **UIT-ViQuAD** (Nguyen et al. 2020): "over 23,000 human-generated question-answer pairs based on 5,109 passages of 174 Vietnamese articles from Wikipedia" — nguồn single-hop hiện dùng.
- **VIMQA (LREC 2022, aclanthology 2022.lrec-1.700):** "over 10,000 Wikipedia-based multi-hop question-answer pairs ... Sentence-level supporting facts are provided" — rất phù hợp cho nhánh multi-hop và tạo haystack. Cần ký User Agreement để lấy bản đầy đủ.
- Bổ sung: ViQA-COVID (multi-span, arXiv 2504.21017), ViHERMES (multi-hop y tế/pháp lý, arXiv 2602.07361), VMLU (đánh giá tổng hợp, ACL 2025). **Chưa có benchmark tiếng Việt chuyên về context compression** → VCC-Bench của nhóm có chỗ đứng rõ ràng.

---

## Details (Phân tích các hướng ứng viên)

### Điểm yếu cốt lõi của pipeline hiện tại (đã xác nhận qua khảo sát)
1. **β ≈ ContextCite** → cần định vị lại là "amortization" (distill thành model nhanh), KHÔNG phải "attribution" (nếu không sẽ bị đánh giá là trùng ContextCite/AT2).
2. **Trên single-hop, β dễ sập về nhãn "chunk chứa gold span"** → BẮT BUỘC phải đánh bại baseline answer-span oracle, và nên dời trọng tâm sang **multi-hop** (VIMQA/HotpotQA/2Wiki) — nơi nhãn span đơn lẻ không đủ vì cần ≥2 chunk.
3. **Giả định cộng tính (additivity) của Ridge** bỏ qua tương tác giữa các chunk (multi-hop cần nhiều chunk cùng lúc) — đây là điểm yếu lý thuyết, cũng là chỗ mở ra ablation thú vị.
4. **Nhãn gắn với 1 reader** → rủi ro theo đúng phát hiện của arXiv 2606.21807; biến điểm yếu này thành một đóng góp (nghiên cứu cross-reader).

### So sánh 4 hướng ứng viên
- **Hướng A — Giữ β/Ridge, định vị lại thành "Amortized Utility Attribution for Compression":** distill hệ số Ridge của F1-downstream theo chunk thành pseudo-label để huấn luyện pruner cross-encoder (Provence-style + Late-Chunking pooling), chạy 1 forward pass lúc inference.
  - *Novelty:* Cao — công thức cụ thể (random-mask + Ridge surrogate của **F1** + distill vào một pruner **riêng biệt**) chưa ai công bố. Khác AT2 (dùng logprob, amortize vào attention-head reader) và khác LooComp (nhãn nhị phân, LOO tất định).
  - *Rủi ro:* Trung bình — reviewer sẽ hỏi "khác gì ContextCite/AT2/CORE-RAG"; phải chuẩn bị bảng so sánh rõ ràng.
  - *Compute:* Vừa — nặng nhất là khâu sinh nhãn (nhiều forward pass của reader).
- **Hướng B — Set-level / non-additive selection:** thay Ridge cộng tính bằng surrogate có tương tác cặp (pairwise) + chọn greedy/submodular, hoặc learned set scorer.
  - *Novelty:* Trung bình — submodular/DPP cho RAG đang bị nhiều nhóm khai thác mạnh (AdaGReS, GeoRAG, ScalDPP, "What Survives"). *Rủi ro:* Cao (dễ đụng hàng), nhưng giải đúng điểm yếu additivity + multi-hop. → Dùng làm **một ablation/phần phụ** của Hướng A, không làm trục chính.
- **Hướng C — Reader-agnostic / cross-reader label transfer:** nghiên cứu tính chuyển được của nhãn compression giữa các reader, đề xuất cách tạo nhãn ổn định xuyên reader (ví dụ ensemble nhiều reader, hoặc chuẩn hóa theo upgrade retention).
  - *Novelty:* Cao và ít người làm; được arXiv 2606.21807 hậu thuẫn trực tiếp. *Rủi ro:* Thấp–trung bình; rất hợp làm **"phần thứ hai"** củng cố cho paper. *Compute:* Vừa (cần nhiều reader để đánh giá nhưng có thể dùng model nhỏ).
- **Hướng D — Soft-prompt/KV compression:** rủi ro cao, đụng nhiều nhóm mạnh (ICAE, xRAG, CompLLM, 500xCompressor); **không khuyến nghị** làm trục chính với ngân sách 4×H100.

---

## Recommendation (Khuyến nghị hành động)

### Hướng chính đề xuất
**Working title:** *"Amortized Utility Attribution for Reader-Agnostic Context Compression in Vietnamese and Multilingual QA"*
(Ghép **Hướng A** làm phương pháp cốt lõi + **Hướng C** làm nghiên cứu phụ trợ; đánh giá trên VCC-Bench tiếng Việt + benchmark tiếng Anh để tạo độ tin cậy.)

**Câu hỏi nghiên cứu / giả thuyết:**
- **RQ1:** Có thể distill attribution theo **utility F1** (không phải logprob như AT2) từ random-mask + Ridge surrogate thành một pruner cross-encoder chạy 1 lượt mà vẫn giữ EM/F1 downstream ở ngân sách token cố định không?
- **RQ2:** Nhãn utility-attribution có **vượt baseline answer-span oracle** trên cả single-hop và multi-hop không? (Đây là test sống–còn của tính mới.)
- **RQ3:** Nhãn sinh từ 1 reader có **chuyển được sang reader khác** không, và cách tạo nhãn nào ổn định xuyên reader nhất (đo bằng upgrade retention)?
- **RQ4:** Trên tiếng Việt (token_F1 cấp âm tiết), phương pháp có vượt **XProvence zero-shot** và **LLMLingua-2** không?

**Method sketch (phác thảo phương pháp):**
1. **Sinh nhãn:** random chunk mask trên haystack (UIT-ViQuAD single-hop + VIMQA multi-hop) → reader Qwen3-8B → token_F1 → Ridge surrogate F1 theo chunk → hệ số β = pseudo-label; có hiệu chỉnh position-bias và chuẩn hóa within-document (within-document-normalized).
2. **Distill (amortize):** huấn luyện pruner cross-encoder đọc [query; full context] + Late-Chunking span pooling per chunk → P(KEEP_i), loss ranking + regression trên β. Bắt đầu bằng KEEP/DROP nhị phân; mở rộng multi-action (KEEP/DROP/TRUNCATE/SUMMARIZE) và budget-aware calibrated selection sau.
3. **Reader-agnostic:** lặp lại sinh nhãn với ≥3 reader (yếu/trung/mạnh), đo **upgrade retention** theo giao thức `ragscale` (arXiv 2606.21807).
4. **Budget-aware:** hiệu chỉnh (calibrate) để chọn ở nhiều tỷ lệ nén (ví dụ giữ 1/4, 1/8), khớp với ngân sách deployment thay vì mask-distribution lúc train.

**Datasets:**
- *Tiếng Việt:* VCC-Bench / vcc_bench_v2, UIT-ViQuAD, VIMQA.
- *Đa ngôn ngữ/tiếng Anh (để credibility):* HotpotQA, 2WikiMultihopQA, Natural Questions, MKQA và/hoặc LongBench.

**Baselines (reviewer chắc chắn sẽ đòi):** LLMLingua-2, LongLLMLingua, Provence/XProvence, RECOMP, EXIT, ContextCite-based pruning, embedding-similarity, **answer-span oracle** (baseline sống–còn), và CORE-RAG nếu code sẵn có.

**Metrics:** EM/F1 downstream ở **ngân sách token cố định**; compression rate; latency/FLOPs; **chi phí sinh nhãn** (số forward pass của reader — điểm nhấn "amortized"); upgrade retention cross-reader.

**Ablations:** F1-surrogate vs logprob-surrogate (để phân biệt với AT2); Ridge cộng tính vs surrogate có tương tác (điểm yếu additivity); có/không position adjustment; nhãn 1 reader vs nhiều reader; single-hop vs multi-hop; hard negatives vs easy distractors (topic-mismatch); ảnh hưởng của chunk boundary/fragmentation.

**Ước lượng compute trên 4×H100 80GB (khả thi):** Reader Qwen3-8B ở BF16 chạy tốt trên 1 GPU với vLLM; 4 GPU chạy song song sinh nhãn. Ví dụ ~20.000 câu × ~64 mask/câu ≈ 1,28M forward pass reader — chia 4 GPU, batch lớn, ước tính vài ngày. Huấn luyện pruner cross-encoder (cỡ vài trăm triệu tham số) rất nhẹ, vài giờ mỗi epoch. Toàn bộ dự án nằm gọn trong ngân sách.

**Timeline theo tuần đến hạn 12/10/2026 (giả định bắt đầu ~T6/2026, ~20 tuần):**
- Tuần 1–3: chốt VCC-Bench task types; lấy VIMQA (ký User Agreement); dựng pipeline sinh nhãn.
- Tuần 4–7: sinh nhãn đa reader; kiểm định chất lượng nhãn (đo noise reader + surrogate).
- Tuần 8–11: huấn luyện pruner + chạy toàn bộ baselines.
- Tuần 12–15: thí nghiệm chính + ablations + cross-reader (upgrade retention).
- Tuần 16–18: thí nghiệm đa ngôn ngữ/tiếng Anh; so trực tiếp XProvence.
- Tuần 19–20: viết bài, nội bộ review, format ARR, nộp.

**Ngưỡng/benchmark làm thay đổi khuyến nghị:**
- Nếu ở RQ2 phương pháp **không vượt** answer-span oracle trên single-hop → dời hẳn trọng tâm sang multi-hop và reader-agnostic (Hướng C lên làm trục chính).
- Nếu tìm thấy (khi search lại sát hạn) một preprint đã dùng đúng "random-mask Ridge-F1 surrogate → distill pruner" → chuyển novelty sang trục **reader-agnostic transfer** (Hướng C), vốn ít bị đụng hơn.
- Nếu XProvence zero-shot đã rất mạnh trên tiếng Việt và khó vượt → nhấn mạnh đóng góp **benchmark VCC-Bench** + phân tích cross-reader thay vì chỉ số SOTA thuần.

---

## Caveats (Lưu ý & rủi ro)
- Nhiều nguồn là **preprint 2026** (PoC 2603.19733, LooComp 2603.09222, BEAVER 2603.19635, CORE-RAG v4, arXiv 2606.21807) — mới, ít trích dẫn, landscape đổi rất nhanh; **cần search lại sát hạn nộp** để chắc chưa bị đụng hàng, đặc biệt cụm từ khóa "F1 surrogate pseudo-label pruner", "amortized attribution compression".
- **Rủi ro số 1 — trùng ý tưởng:** AT2 (amortize ablation attribution, 1 forward pass, đã thử pruning) là mối đe dọa gần nhất; phải nêu rõ khác biệt (F1 vs logprob; pruner cross-encoder riêng vs attention-head của reader). LooComp là mối đe dọa gần nhất về kiến trúc (encoder-only cross-encoder pruner) nhưng dùng nhãn nhị phân + LOO tất định.
- **Rủi ro số 2 — β sập về answer-span** trên single-hop → mitigate bằng trọng tâm multi-hop + so oracle bắt buộc.
- **VCC-Bench task types chưa chốt** → giữ đề xuất linh hoạt cho cả extractive QA và multi-hop (như task yêu cầu).
- Danh sách track/area cụ thể và giới hạn trang chính xác của COLING 2027 nên **xác nhận lại trên site chính thức 2027.coling-iccl.org** trước khi nộp (một vài chi tiết area lấy từ bản CFP trên mailing list, chưa phải trang track đầy đủ).
- Một điểm cần cân nhắc về định vị: vì COLING là hạng B và theme đặc biệt là "NLP for Linguistics", nên nộp vào **area chính** (Information Retrieval / Efficient Methods / QA / Multilinguality) chứ không phải theme track; đóng góp benchmark tiếng Việt + phân tích cross-reader sẽ giúp bài dễ được đánh giá tích cực ở venue này.