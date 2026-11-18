# AI Usage · Exercise 1

| Bước | Phần AI hỗ trợ | Cách kiểm chứng khi chạy |
| --- | --- | --- |
| Chọn model | Xác định checkpoint DistilBERT đã fine-tune trên SST-2 | Đọc model card và đối chiếu `model_id` |
| Tokenize | Viết luồng in token, ID và mask của chính tensor đầu vào | Kiểm tra các mảng có cùng chiều dài và đối chiếu token đặc biệt |
| Inference và visualization | Tính softmax, chọn nhãn, vẽ hai điểm | Kiểm tra tổng gần 1 và nhãn dự đoán là lớp có điểm cao nhất |

[Nhật ký chi tiết](docs/AI_Usage_Full.md).
