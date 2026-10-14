# MLP được dùng

- Input: ảnh grayscale [B,1,28,28], đã chuyển về float32 [0,1] bằng ToTensor.
- Kiến trúc: Flatten → Linear(784,256) → ReLU → Linear(256,128) → ReLU → Linear(128,10), dùng nn.Sequential; 235.146 tham số.
- Đầu ra là 10 logits, tương ứng các nhãn FashionMNIST. Không dùng Softmax vì CrossEntropyLoss đã gồm log-softmax.
- Cấu hình train hiện tại ở đầu train.py: 30 epoch, Adam lr=0,001, không weight decay, StepLR(10; 0,3), seed 42, batch 64.
- model.py chỉ định nghĩa mạng và kiểm tra forward; train.py huấn luyện và lưu metrics/checkpoint theo từng run.
- Các kiến trúc đã thử (128-10, 128-64-10, 512-256-128-10 có Dropout) nằm trong [05_experiments.md](05_experiments.md).
- Mô hình bàn giao: lan_10/best.pt; kết quả validation và test ở [04_result.md](04_result.md).

Chi tiết: [model.py](scripts/model.md).
