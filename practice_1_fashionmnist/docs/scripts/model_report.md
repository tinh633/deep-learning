# model_report.py

- Chạy: `python practice_1_fashionmnist/src/model_report.py`.
- Input: lan_10/best.pt, train.log, model.py và data_pipeline.py cùng cấu hình train.
- Đánh giá cùng checkpoint trên train/validation/test với eval/no_grad, không cập nhật trọng số.
- Output: outputs/runs/lan_10/model_report.json cho HTML đọc qua Live Server.
- Gồm kiến trúc, cấu hình từ log, lịch sử epoch, loss/accuracy, confusion matrix (hàng thật/cột dự đoán) và mọi ảnh sai với ID gốc/nhãn/confidence.
- Train accuracy/loss ở đây là đánh giá checkpoint cố định trong eval mode, không phải trung bình online lúc cập nhật trọng số.
- Softmax confidence không phải xác suất đã được kiểm định độ tin cậy.
- lan_10 ghi nhận 23/30 epoch; best epoch 12. Không train thêm hoặc tuyên bố tối ưu mọi cấu hình.
