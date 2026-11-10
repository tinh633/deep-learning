# Audit log — Checklist smoke run

Trạng thái 10/11/2026: chưa smoke model/train. Bảng dưới là checklist dự kiến, chưa phải kết quả.
Chiến lược: [Training_Strategy.md](design/Training_Strategy.md).
Thay đổi code/pipeline ghi ở [CHANGELOG.md](CHANGELOG.md); việc tiếp theo ghi ở [progress.md](progress.md).

## Quy tắc ghi nhận

- Chỉ ghi cấu hình, kiểm tra đã chạy, bằng chứng và kết luận smoke; không kể quá trình sửa code.
- Trạng thái: PASS, FAIL hoặc NOT_RUN; không xem chạy không lỗi là mọi kiểm tra đều đạt.
- Mỗi lượt thực có ngày, run ID/thư mục, phiên bản source, model/A–D, seed, split và cấu hình.
- Ghi máy/GPU, precision, batch, số batch/epoch thực chạy và đường dẫn log/checkpoint nguồn.
- Khi kiểm tra thất bại, ghi FAIL và liên kết thay đổi/kiểm chứng lại trong changelog, không xóa kết quả cũ.
- “Ổn định” chỉ trong phạm vi số batch đo: có/không NaN, Inf, OOM, biến động thời gian/VRAM.
- Smoke ngắn không chứng minh hội tụ hoặc khả năng tổng quát hóa; không dùng test để smoke/tuning.
- Audit Markdown là bản tóm tắt; bằng chứng chi tiết đọc từ artifact thật, không bắt buộc thêm audit.py/JSON riêng.

## Checklist trước smoke — Chưa thực thi

A: ResNet freeze; B: ResNet fine-tune; C: ViT freeze; D: ViT fine-tune.
Các cột chỉ được cập nhật khi có run và bằng chứng tương ứng.

| Kiểm tra | A | B | C | D |
|---|---|---|---|---|
| Nguồn weights, class order, split/config đúng | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| Input/output shape và logits 100 lớp | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| Loss hữu hạn, không NaN/Inf | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| Trainable parameters đúng chiến lược | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| Head/backbone gradient đúng | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| Optimizer chứa đúng tham số được mở | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| Tham số/buffers thay đổi đúng policy freeze/BN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| Val eval/no-gradient, không thay state | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| Save/load tái tạo logits trong tolerance đã ghi | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| Resume epoch/optimizer/RNG đúng cấu hình | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| Thời gian, peak VRAM và số batch đo được ghi | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| Không NaN/Inf/OOM trong số batch đã công bố | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |

## Mẫu ghi một lượt thực tế

Khi chạy, thêm mục theo ngày/run và liên kết thư mục bằng chứng thực sự tồn tại:

1. Nhận diện: run ID, model/chiến lược, source revision/hash, weights và split hash.
2. Điều kiện: máy, GPU, batch/precision, seed và phạm vi thực thi.
3. Kết quả: từng check, trạng thái, số đo/tolerance và bằng chứng; ghi rõ check chưa chạy.
4. Ổn định: số batch quan sát, NaN/Inf/OOM, thống kê thời gian/VRAM nếu đã đo.
5. Kết luận: có đủ điều kiện tuning chưa, còn giới hạn nào; liên kết recheck nếu cần.

Chưa có run ID, số đo tài nguyên hoặc kết luận PASS cho model nào ở thời điểm tạo file.
Khi gần 150 dòng, chuyển lượt cũ sang docs/archive/audit_<giai_doan>.md và liên kết, không xóa lịch sử.
