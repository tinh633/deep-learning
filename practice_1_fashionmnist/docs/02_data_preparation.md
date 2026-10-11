# Chuẩn bị dữ liệu cho MLP

- Phạm vi: script Python và Dataset/DataLoader; chưa tạo hoặc huấn luyện mô hình.
- Đọc IDX giải nén tại data/FashionMNIST_raw/ bằng FashionMNIST của torchvision; adapter nhỏ chỉ đổi đường dẫn cache, không tải mạng trong bước này.
- Validation chiếm 10% tập train, chia theo nhãn với seed 42: 54.000 train, 6.000 validation. Tham số đặt trực tiếp trong code để dễ đọc.
- Tập test giữ nguyên 10.000 ảnh; không dùng test để chọn cấu hình.
- Chuyển ảnh sang float32 và chia 255 để được miền [0, 1]; nhãn int64. Không dùng mean/std từ test và chưa augmentation.
- Dùng ToTensor để chuyển từng ảnh; dùng train_test_split với stratify=labels và random_state=seed để chia tập, không tự viết thuật toán chia.
- Giữ ảnh dạng [1, 28, 28]; lớp Flatten của MLP sau này sẽ chuyển thành 784 giá trị. Batch có dạng [B, 1, 28, 28].
- Train shuffle với generator có seed; validation/test không shuffle. Mặc định batch size 64, num_workers 0 để đơn giản trên Windows.
- Script hiện tại chỉ tạo dataset/loaders trong RAM và in số mẫu, không xuất JSON hay chỉ số chia. Các file đó nếu còn là kết quả phiên bản cũ.
- Kiểm tra: đủ dữ liệu, train/validation không giao nhau, phủ tập train gốc, phân bố nhãn, seed tái lập và batch có dtype/shape/range đúng.
- Biến train_loader, val_loader, test_loader có thể import ở script huấn luyện sau này. Không hàm wrapper hay CLI.
