# Biểu đồ Exercise 2

Từ thư mục workflow, chạy `python src/2_eda.py` để tạo `label_distribution.png` và `token_lengths.png` trong `eda/<snapshot>-tokens-<max_length>/`. Đường dẫn chính xác được lưu trong `eda_summary.json` của snapshot.

Sau khi có run đã huấn luyện, chạy `python src/4_evaluate.py --run lan_1` rồi `python src/5_report.py --run lan_1`. Thay `lan_1` bằng tên run thực tế.

- `lan_n/training_curves.png` ghi loss train/validation và accuracy/F1 validation. Vạch đứt đánh dấu checkpoint được chọn.
- `lan_n/validation_evaluation.png` ghi confusion matrix và metrics validation.
- `lan_n/test_evaluation.png` sinh khi chạy cả evaluate và report với `--split test`. Phần trăm trong từng ô được tính theo tổng hàng, còn số nguyên là số review.
