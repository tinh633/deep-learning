# AI Usage · Exercise 2

| Bước | Phần AI hỗ trợ | Cách kiểm chứng khi chạy |
| --- | --- | --- |
| Chuẩn bị dữ liệu | Split phân tầng, ghi seed/hash, kiểm tra trùng văn bản giữa các tập | Đọc `split_info.json` và đối chiếu số lượng từng lớp |
| EDA | Thống kê nhãn, median/p90 và tỷ lệ review vượt giới hạn token | Đối chiếu `eda_summary.json` với hai biểu đồ |
| Fine-tuning | Cấu hình Trainer, collator, TensorBoard và checkpoint tốt nhất | Đọc `config.json`, log epoch và validation F1 |
| Đánh giá | Ma trận nhầm lẫn, metrics lớp dương, ví dụ sai | Đối chiếu tổng ma trận với kích thước split |
| Báo cáo | Notebook đọc artifact thật và tài liệu hỏi đáp | Điền nhận xét và ghi rõ run, snapshot và split trình bày |

[Nhật ký chi tiết](docs/AI_Usage_Full.md).
