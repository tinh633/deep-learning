# show_predictions.py

- Chạy: `python practice_1_fashionmnist/src/show_predictions.py [lan_N]` (mặc định lan_10).
- Input: outputs/runs/lan_N/model_report.json, outputs/eda.json và data/FashionMNIST_raw/t10k-*-ubyte. Chạy model_report.py trước nếu thiếu báo cáo.
- Phụ thuộc: NumPy, matplotlib. Không cần PyTorch; không train hay đánh giá lại model.
- Output trong outputs/runs/lan_N/plots/: predictions.png (20 ảnh test đầu tiên) và confident_mistakes.png (20 lỗi có confidence cao nhất).
- model_report.json liệt kê mọi ảnh test bị sai; ảnh không có trong danh sách được xem là dự đoán đúng. Script kiểm tra số ảnh test khớp báo cáo.
- Tiêu đề màu xanh là dự đoán đúng, màu đỏ là sai; mỗi ảnh ghi ID test, nhãn dự đoán và nhãn thật.
- Softmax confidence không phải xác suất đã được kiểm định độ tin cậy.
