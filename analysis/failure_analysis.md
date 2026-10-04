# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Nguyễn Đức Thịnh  
**Khóa:** K4 - Track 3A  

> Nguồn số liệu: `reports/ragas_report.json`, `reports/latency_breakdown.json`, `reports/naive_baseline_report.json` (chạy `python src/pipeline.py` với `HYBRID_TOP_K = 10`, 20 câu test).
> Lần chạy này cho ra bottom-5 khác lần chạy trước (RAGAS dùng LLM chấm nên có dao động).

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.7719 | 0.8595 | +0.0876 |
| Answer Relevancy | 0.6781 | 0.7901 | +0.1120 |
| Context Precision | 0.9250 | 0.9250 | 0.0000 |
| Context Recall | 0.9250 | 0.9500 | +0.0250 |

Production đạt ≥ 0.75 ở cả 4 metric; Faithfulness đạt ≥ 0.85 (sát ngưỡng, khoảng 0.01).

**Latency (query, trung bình, CPU):** search 347 ms · rerank 6,154 ms · generate 1,499 ms · total 8,000 ms. Rerank vẫn cao hơn mục tiêu 150 ms của M3 khoảng 40 lần. Với 20 ứng viên thay vì 10 thì rerank khoảng 10.9 s, nhưng Context Precision đạt 1.000 và Faithfulness 0.873.

---

## Bottom-5 Failures

Thứ tự theo điểm trung bình 4 metric (thấp → cao).

### #1 — avg 0.562
- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** 15 + 3 (9÷3) = 18 ngày phép. Lương Senior (P3–P4): 20–35 triệu VNĐ/tháng.
- **Got:** 18 ngày (đúng), nhưng phần lương nói không có thông tin trong context.
- **Worst metric:** answer_relevancy = 0.0
- **Error Tree:** Output sai một phần → Context đúng? **Không đủ** (không có bảng lương, và có cả bản nghỉ phép 2023 lẫn 2024) → Query OK? Câu hỏi có 2 ý (multi-hop) → Fix ở bước **retrieval**.
- **Root cause:** Câu hỏi cần 2 tài liệu nhưng top-3 chỉ có chính sách nghỉ phép. Context lại chứa bản 2023 (đã thay thế) cùng bản 2024.
- **Suggested fix:** Tách câu hỏi thành 2 sub-query rồi gộp context. Lọc bản cũ bằng metadata `effective_date`.

### #2 — avg 0.766
- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected (GT):** Hạn 15 ngày, quá 5 ngày; 2%/tháng trên 15 triệu = 300.000/tháng (GT ghi pro-rata ≈ 50.000 cho 5 ngày).
- **Got:** Trả lời 2%/tháng và suy ra 300.000 VNĐ, coi như phạt đủ 1 tháng dù mới trễ 5 ngày.
- **Worst metric:** faithfulness = 0.27
- **Error Tree:** Output sai một phần → Context đúng? **Có** (chính sách tạm ứng ghi rõ 15 ngày và 2%/tháng) → Query OK → Fix ở bước **generation**.
- **Root cause:** Context không quy định cách tính theo tháng hay pro-rata. LLM tự suy diễn, nên judge chấm faithfulness thấp. GT cũng dùng cách tính không có trong context, nên câu này mơ hồ ở cả ground truth.
- **Suggested fix:** Thắt prompt "chỉ dùng quy tắc có trong context; nếu thiếu thông tin cách tính thì nói rõ giả định". Sửa lại GT cho khớp tài liệu.

### #3 — avg 0.815
- **Question:** Thâm niên bao nhiêu năm thì được cộng thêm ngày phép?
- **Expected (GT):** Theo chính sách 2024 hiện hành: từ 3 năm, cứ 3 năm cộng 1 ngày.
- **Got:** Trả lời theo bản 2023 ("từ 5 năm, cứ 5 năm cộng 1 ngày"), rồi nhắc thêm bản 2024.
- **Worst metric:** context_precision = 0.5
- **Error Tree:** Output sai → Context đúng? **Có nhưng lẫn bản cũ** (cả bản 2023 và 2024 đều nằm trong context) → Query OK → Fix ở bước **retrieval (versioning)**.
- **Root cause:** Bản 2023 đứng trước bản 2024 trong top context nên LLM dùng bản cũ. Đây là lỗi thật, không phải lỗi judge.
- **Suggested fix:** Lọc hoặc đẩy bản superseded xuống bằng `effective_date`; trong prompt yêu cầu dùng bản có ngày hiệu lực mới nhất.

### #4 — avg 0.825
- **Question:** Nhân viên được tài trợ khóa học 25 triệu, nghỉ việc sau 8 tháng hoàn thành khóa học. Phải hoàn trả bao nhiêu?
- **Expected:** Cam kết 1 năm; nghỉ sau 8 tháng → hoàn 100% = 25 triệu.
- **Got:** 100% = 25 triệu (đúng).
- **Worst metric:** faithfulness = 0.5
- **Error Tree:** Output đúng → Context đúng? **Có** (chính sách hoàn chi đào tạo có điều khoản 100%) → Query OK → Judge chấm thấp dù đáp án đúng.
- **Root cause:** Context thứ hai là "chính sách đào tạo nội bộ", không liên quan đến hoàn chi. Đây là nhiễu. Điểm 0.5 chủ yếu do judge, chưa đủ bằng chứng để kết luận thêm.
- **Suggested fix:** Rerank tốt hơn để loại chunk nhiễu; không cần đổi prompt.

### #5 — avg 0.830
- **Question:** Bao lâu phải đổi mật khẩu một lần?
- **Expected (GT):** Theo chính sách hiện hành (v2.0): 120 ngày. Chính sách cũ yêu cầu 90 ngày.
- **Got:** 120 ngày (đúng).
- **Worst metric:** context_precision = 0.5
- **Error Tree:** Output đúng → Context đúng? **Có nhưng lẫn bản cũ** (chính sách mật khẩu cũ v1.0 và hiện hành v2.0 đều được lấy) → Query OK → Fix ở bước **retrieval (versioning)**.
- **Root cause:** Bản cũ vẫn nằm trong top context. Câu trả lời đúng nhưng context kém chính xác.
- **Suggested fix:** Cùng cách với #3: metadata phiên bản và ưu tiên bản hiện hành.

---

## Cross-cutting insight

- **3/5 câu bottom (#1, #3, #5) có bản cũ lẫn vào context.** Trong đó #3 là lỗi thật: LLM trả lời theo bản 2023. Đây là vấn đề lớn nhất và có bằng chứng rõ nhất.
- **#2 và #4 có context đúng nhưng điểm thấp.** Nguyên nhân là LLM suy diễn thêm (#2) hoặc judge không nối được nguồn (#4).
- Cải tiến có lợi nhất theo bằng chứng hiện tại: (1) metadata `effective_date` và ưu tiên bản hiện hành, (2) tách câu multi-hop, (3) thắt prompt để không suy diễn.

---

## Case Study (cho presentation)

**Question chọn phân tích:** #3 — "Thâm niên bao nhiêu năm được cộng ngày phép?" Đây là lỗi rõ ràng nhất, có thể chỉ ra đúng bước gây lỗi.

**Error Tree walkthrough:**
1. Output đúng? → Không: LLM dùng quy định 2023 (5 năm, cứ 5 năm 1 ngày), trong khi hiện hành là 3 năm, cứ 3 năm 1 ngày.
2. Context đúng? → Có đúng bản 2024, nhưng bản 2023 cũng nằm trong context và được đặt trước.
3. Query rewrite OK? → Có, câu hỏi rõ ràng; vấn đề là không phân biệt phiên bản.
4. Fix ở bước: **retrieval** — gắn `effective_date` cho chunk, ưu tiên hoặc lọc bản hiện hành trước khi đưa cho LLM.

**Nếu có thêm 1 giờ, sẽ optimize:**
- Thêm `effective_date` vào metadata, lọc bản superseded, rồi đo lại context_precision và faithfulness.
- Giảm latency rerank (6.2 s/query trên CPU, mục tiêu M3 là < 150 ms). Còn lại: `FlashrankReranker` (khoảng 90 ms, nhưng model tiếng Anh nên cần kiểm tra chất lượng tiếng Việt) hoặc chạy bge-reranker trên GPU.
