# Reflection — Lab 18: Production RAG

**Họ và tên:** Nguyễn Đức Thịnh  
**Khóa:** K4 - Track 3A  
**Ngày hoàn thành:** 2026-10-04

---

## Phần 1: Mapping bài giảng

| Bài giảng | Module | Hàm | Quan sát |
|-----------|--------|-----|----------|
| Chia nhỏ theo ý nghĩa (semantic) | M1 | `chunk_semantic()` | Ngưỡng 0.85 cắt quá nhiều: ra 208 đoạn, mỗi đoạn trung bình chỉ 99 ký tự. Lần sau mình sẽ thử ngưỡng thấp hơn, khoảng 0.5–0.7. |
| Tìm theo từ khóa + tìm theo nghĩa | M2 | `reciprocal_rank_fusion()` | Gộp theo thứ hạng nên không cần đưa điểm hai bên về cùng thang đo. Nhớ đổi dấu `_` thành khoảng trắng, nếu không "nghỉ phép" sẽ không khớp với "nghỉ_phép". |
| Rerank bằng cross-encoder | M3 | `CrossEncoderReranker.rerank()` | Xếp đúng thứ tự, nhưng chạy trên CPU mất khoảng 6 giây mỗi câu hỏi khi chỉ xét 10 ứng viên (20 ứng viên là 11 giây). Vẫn quá chậm so với mục tiêu 150 ms. |
| Đo chất lượng bằng RAGAS | M4 | `evaluate_ragas()` | Context Recall cao nhất (0.95), Context Precision 0.925. Answer Relevancy thấp nhất (0.79), vì có câu trả lời từ chối một phần khi câu hỏi có hai ý. Faithfulness 0.86, sát ngưỡng 0.85. |
| Làm giàu chunk trước khi index | M5 | `_enrich_single_call()` | Gọi 1 lần cho mỗi chunk, 118 lần, mất khoảng 43 giây. Chưa tách riêng được bao nhiêu điểm là nhờ bước này. |

---

## Phần 2: Khó khăn

**1. Ổ C hết chỗ khi tải model**
- Lỗi: `There is not enough space on the disk. (os error 112)`
- Mình kiểm tra thì ổ C chỉ còn khoảng 1.4 GB, trong khi model bge-m3 cần khoảng 2.3 GB.
- Cách làm: chuyển cache của model và thư mục tạm sang ổ D. Không xóa gì của máy.

**2. Load bge-m3 báo hết bộ nhớ**
- Lỗi: `memory allocation of 67051040 bytes failed`
- RAM còn khoảng 3 GB. Khi đổi thư mục tạm sang ổ D thì load được.
- Bài học: model lớn thì nên để cache và temp ở ổ còn nhiều chỗ.

**3. Docker chưa bật**
- Lỗi: không kết nối được Docker API.
- Cách làm: mở Docker Desktop, đợi nó chạy xong rồi mới `docker compose up -d`.

**4. Đọc sai file `.env`**
- Mình tưởng key vẫn là chữ `sk-...` mẫu, nhưng thật ra đã có key. Lần sau phải mở file kiểm tra lại trước khi kết luận.

**Còn thiếu:**
- Chưa biết cách chấm RAGAS cho tiếng Việt cho chuẩn. Phần hướng dẫn của bộ chấm vẫn là tiếng Anh.
- Chưa biết cách lọc theo phiên bản trong Qdrant để bỏ bản cũ.

---

## Phần 3: Kế hoạch áp dụng vào project

**Project:** Hệ thống hỏi đáp tài liệu nội bộ

**Hiện tại:**
- Chia chunk cha-con, làm giàu chunk, tìm kiếm kết hợp, rerank, rồi cho LLM trả lời.
- Vấn đề chính: rerank chậm, câu hỏi có nhiều ý thì thiếu thông tin, và đôi khi tài liệu cũ lẫn vào kết quả (thấy rõ ở một câu về nghỉ phép).

**Kế hoạch:**
1. **Chia chunk:** giữ kiểu cha-con. Tài liệu có tiêu đề rõ thì thử kiểu theo tiêu đề. Bỏ kiểu theo ý nghĩa với ngưỡng 0.85.
2. **Tìm kiếm:** dùng kết hợp từ khóa và tìm theo nghĩa, nhớ tách từ tiếng Việt.
3. **Rerank:** đã giảm số ứng viên từ 20 xuống 10: rerank từ 11 giây xuống 6 giây, nhưng Context Precision giảm từ 1.0 xuống 0.925. Lần sau thử Flashrank (khoảng 90 ms) nhưng phải kiểm tra chất lượng tiếng Việt trước.
4. **Đánh giá:** chạy RAGAS trên bộ câu hỏi có nhiều loại, theo dõi 5 câu tệ nhất sau mỗi lần sửa.
5. **Làm giàu:** gắn thêm ngày hiệu lực và phiên bản vào metadata, để ưu tiên bản mới.

**Timeline:**
- **Tuần 1:** thêm phiên bản và ngày hiệu lực vào metadata, lọc bản mới, đo lại.
- **Tuần 2:** giảm thời gian rerank, theo dõi thời gian từng bước mỗi tuần.
- **Tuần 3:** tách câu hỏi nhiều ý thành các câu nhỏ, so sánh có và không làm giàu chunk.
