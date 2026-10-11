# model.py

- Mục đích: MLP mẫu để người dùng chỉnh kiến trúc; chưa huấn luyện.
- Kiến trúc: Flatten → Linear(784,128) → ReLU → Linear(128,10), dùng nn.Sequential.
- Tham số đầu file: input_size, hidden_size, num_classes. hidden_size là tham số thử nghiệm; 784 và 10 tương ứng kích thước ảnh và số nhãn của bài.
- Đầu vào: float32 [B,1,28,28]; đầu ra: logits [B,10]. Không thêm Softmax trước CrossEntropyLoss.
- Có thể thêm lớp ẩn trong nn.Sequential; kích thước đầu ra của lớp trước phải khớp đầu vào lớp sau.
- Chạy: `python practice_1_fashionmnist/src/model.py`.
- Khi chạy trực tiếp, đọc một batch thật từ train_loader và kiểm tra shape/giá trị hữu hạn; không lưu file, không backward hay cập nhật trọng số.
- Khi import, cung cấp biến model; không tự đọc dataset hoặc chạy kiểm tra.
- Phụ thuộc: PyTorch; kiểm tra trực tiếp cần dữ liệu và các dependency của data_pipeline.py.
