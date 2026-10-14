# model.py

- Mục đích: định nghĩa MLP dùng cho bài và kiểm tra forward; không huấn luyện.
- Kiến trúc: Flatten → Linear(784,256) → ReLU → Linear(256,128) → ReLU → Linear(128,10), dùng nn.Sequential; 235.146 tham số.
- Đầu vào: float32 [B,1,28,28]; đầu ra: logits [B,10]. Không thêm Softmax trước CrossEntropyLoss.
- Đổi kiến trúc bằng cách sửa nn.Sequential; kích thước đầu ra của lớp trước phải khớp đầu vào lớp sau. Khi tải checkpoint cũ, cần kiến trúc tương ứng.
- Chạy: `python practice_1_fashionmnist/src/model.py`.
- Khi chạy trực tiếp, đọc một batch thật từ train_loader và kiểm tra shape/giá trị hữu hạn; không lưu file, không backward hay cập nhật trọng số.
- Khi import, cung cấp biến model; không tự đọc dataset hoặc chạy kiểm tra.
- Phụ thuộc: PyTorch; kiểm tra trực tiếp cần dữ liệu và các dependency của data_pipeline.py.
