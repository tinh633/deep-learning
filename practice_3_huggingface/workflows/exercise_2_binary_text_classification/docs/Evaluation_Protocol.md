# Chọn mô hình và diễn giải kết quả

## Validation và test

Validation chạy sau mỗi epoch. Trainer lưu mô hình tốt nhất theo validation F1, sau đó đánh giá lại checkpoint đã chọn. `4_evaluate.py --split validation` bổ sung ma trận nhầm lẫn và lỗi mẫu. Khi nhóm chọn xong cấu hình, chạy với `--split test` trên run đó để đo dự đoán trên review giữ riêng. Script lưu kết quả test riêng và kiểm tra file trước khi ghi.

| Chỉ số | Công thức / ý nghĩa |
| --- | --- |
| Accuracy | (TP + TN) / N. Tỷ lệ toàn bộ review được dự đoán đúng. |
| Precision | TP / (TP + FP). Trong các dự đoán tích cực, bao nhiêu review thực sự tích cực. |
| Recall | TP / (TP + FN). Trong các review tích cực, bao nhiêu review được tìm đúng. |
| F1 | Trung bình điều hòa của precision và recall, tính với `POSITIVE` là lớp dương. |
| Loss | Cross-entropy trên logits. Dự đoán sai mà rất tự tin thường chịu phạt lớn. |

Ma trận mặc định có hàng là nhãn thật, cột là nhãn dự đoán. Với thứ tự NEGATIVE, POSITIVE, ma trận là `[[TN, FP], [FN, TP]]`. Ví dụ ô hàng POSITIVE, cột NEGATIVE là review tích cực bị đoán thành tiêu cực.

## Trình bày kết quả

1. Nêu tên run, kích thước split, F1 validation tại checkpoint được chọn và số mẫu test.
2. Trình bày diễn biến loss theo epoch. Nếu train loss giảm trong khi validation loss tăng kéo dài, xem khả năng overfitting. Train loss của Trainer được ghi theo quá trình cập nhật trong epoch, còn validation loss được đo sau epoch trên một phiên bản mô hình xác định.
3. Đọc FP/FN trên ma trận. JSON giữ tối đa 10 lỗi điểm softmax cách xa 0.5 nhất và đoạn đầu bài review. Dựa vào nội dung thật trước khi đề xuất nguyên nhân như phủ định, mỉa mai hoặc truncate.
4. Báo test accuracy và F1 với đúng split. Ví dụ suy luận một câu trong script `6_infer_finetuned.py` minh họa cách dùng mô hình đã lưu sau khi huấn luyện.
