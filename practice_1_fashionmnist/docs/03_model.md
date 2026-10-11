# MLP cơ sở

Người dùng cho phép viết mô hình mẫu và sẽ tự chỉnh tham số.

- Input: ảnh grayscale [B,1,28,28], đã chuyển về float32 [0,1].
- Flatten tạo vector 784 giá trị cho mỗi ảnh.
- Lớp ẩn Linear(784,128) và ReLU.
- Lớp cuối Linear(128,10) tạo 10 logits, tương ứng các nhãn FashionMNIST.
- Không dùng Softmax; train.py dùng CrossEntropyLoss và Adam. Cấu hình epochs/lr/seed đặt đầu script train.
- hidden_size đặt đầu model.py để dễ chỉnh; thêm lớp bằng cách sửa nn.Sequential.
- model.py chỉ kiểm tra forward; train.py thực hiện huấn luyện riêng và lưu metrics/checkpoints theo từng run. Tập test chưa được đánh giá.

Chi tiết: [model.py](scripts/model.md).
