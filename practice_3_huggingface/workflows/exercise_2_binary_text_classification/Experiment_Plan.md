# Kế hoạch thí nghiệm

| Thành phần | Baseline |
| --- | --- |
| Model | `distilbert/distilbert-base-uncased`, head 2 logits |
| Dữ liệu | IMDb, 2.000 train, 400 validation, 500 test |
| Seed | 42 |
| Token | Tối đa 128 token, padding động theo batch |
| Train | 3 epoch, batch 8, LR 2e-5 |
| Tối ưu | AdamW, weight decay 0.01, linear LR, gradient clipping 1.0 |
| Chọn checkpoint | Validation F1 của lớp POSITIVE |
| Đánh giá | Accuracy, precision, recall, F1 và confusion matrix |

Dùng `--learning-rate`, `--batch-size` hoặc `--epochs` để thay một biến nếu nhóm chạy thêm thí nghiệm. Ví dụ so sánh 2e-5 với 1e-5 trên cùng snapshot và seed. Khi nghiên cứu độ dài input, giữ seed và số mẫu, tạo snapshot mới với `--max-length 256` rồi kiểm tra các SHA-256 của mẫu theo split. Thay batch size cũng thay số optimizer step mỗi epoch, nên ghi số step khi so sánh tốc độ và kết quả.

Trong bảng kết quả, điền tên run, biến thay đổi, epoch checkpoint tốt nhất, validation F1 và nhận xét từ loss curve. Giữ test đến lúc chốt mô hình. Bộ mẫu nhỏ giúp thực hành với tài nguyên giới hạn. Khi trình bày, ghi rõ số lượng mẫu và giới hạn độ dài khi diễn giải điểm số.
