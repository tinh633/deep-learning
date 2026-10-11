# data_pipeline.py

- Luồng trực tiếp: transform → đọc train/test → chia validation → tạo ba DataLoader.
- Dùng torchvision FashionMNIST/ToTensor, sklearn train_test_split và PyTorch Subset/DataLoader.
- Lớp FashionMNIST chỉ đổi raw_folder để đọc data/FashionMNIST_raw/; không tự viết đọc dữ liệu.
- Thiết lập ngay trong code: validation 10%, seed 42, batch size 64. Không CLI hoặc hàm wrapper.
- Biến dùng cho bước tiếp: train_loader, val_loader, test_loader; chưa huấn luyện mô hình.
- Kết quả: 54.000 train / 6.000 validation / 10.000 test; ảnh float32 [0,1], batch [64,1,28,28], nhãn int64.
- Chạy: `python practice_1_fashionmnist/src/data_pipeline.py`; chỉ in số mẫu và kích thước batch, không ghi file mới.
- Phụ thuộc: PyTorch, torchvision, scikit-learn. Chạy prepare_data.py nếu chưa có IDX giải nén.
- File data_preparation.json và split_indices.npz nếu có là sản phẩm phiên bản cũ, không được script hiện tại cập nhật.
