# Hỏi–đáp về model và thông số — Practice 2

Tài liệu ghi lại các câu hỏi và giải thích phát sinh trong quá trình làm lab.
Thiết lập thí nghiệm chính thức nằm trong [kế hoạch](Experiment_Plan.md);
trạng thái triển khai và cách chạy nằm trong [README](README.md).
Phân biệt lý do thiết kế, giả thuyết và kết quả đã kiểm chứng; bổ sung câu hỏi mới vào đây.
Giữ tối đa 150 dòng; khi dài hơn, tách theo chủ đề và liên kết từ tài liệu này.

## 1. Hai head H1/H2 khác nhau thế nào, tại sao chọn như vậy?

Hai head nhận feature do backbone trích xuất và trả về 10 logits cho CIFAR-10.
D là kích thước feature: 512 ở VGG16/ResNet18, 1024 ở DenseNet121.

| Tiêu chí | H1 — Linear | H2 — MLP |
|---|---|---|
| Cấu trúc | `Linear(D, 10)` | `Linear(D, 256) → ReLU → Dropout(0.2) → Linear(256, 10)` |
| Xử lý feature | Phân loại trực tiếp | Học biến đổi phi tuyến trước khi phân loại |
| Số tham số | Ít | Nhiều hơn |
| Rủi ro overfit | Thường thấp hơn về capacity | Capacity lớn hơn; có Dropout để regularize |

### Vì sao H1 là baseline?

H1 đơn giản, giúp kiểm tra feature pretrained phù hợp CIFAR-10 đến đâu.
Khi freeze backbone, chỉ học ranh giới phân loại trên feature có sẵn, không thay backbone.
Kết quả vẫn phụ thuộc optimizer/LR và quá trình train, không đo riêng chất lượng backbone.

### H2 bổ sung điều gì?

Lớp ẩn và ReLU cho phép học cách kết hợp feature theo quan hệ phi tuyến.
Dropout 0,2 ngẫu nhiên bỏ một phần activation trong lúc train để regularize;
khi eval, Dropout được tắt. H2 có thể thích nghi tốt hơn nhưng không bảo đảm tốt hơn H1.

### Tại sao dùng 256 neuron và Dropout 0,2?

Đây là cấu hình khởi đầu gọn để có một head MLP đối chiếu với Linear,
chưa phải các giá trị tối ưu đã được chứng minh bằng thực nghiệm.
Dùng cùng cấu trúc cho ba model để giữ quy tắc nhất quán; không cân bằng tuyệt đối số tham số.
Không tự thêm lượt tuning hai thông số này ngoài ngân sách đã thống nhất.

### Hai head có bao nhiêu tham số?

Tính cả weights và bias của các lớp Linear; ReLU và Dropout không có tham số học.

| Backbone | D | H1 | H2 |
|---|---:|---:|---:|
| VGG16 / ResNet18 | 512 | 5.130 | 133.898 |
| DenseNet121 | 1024 | 10.250 | 264.970 |

- H1: `D × 10 + 10`.
- H2: `D × 256 + 256 + 256 × 10 + 10`.
- D khác nhau nên cùng thiết kế head vẫn có số tham số khác nhau.

### Tại sao không thêm Softmax trong model?

Cả hai trả logits trực tiếp cho `CrossEntropyLoss`, vốn xử lý bước log-softmax bên trong.
Khi cần confidence để phân tích dự đoán, tính Softmax từ logits ở bước đánh giá.

### A so với D cho phép kết luận điều gì?

Giữ nguyên backbone freeze, LR, batch, số epoch, split và preprocessing; thay H1 bằng H2.
H2 đồng thời đổi capacity, phép biến đổi phi tuyến và Dropout, nên chỉ so sánh được
toàn bộ thiết kế head trong điều kiện đó; chưa tách riêng tác dụng của từng yếu tố.
Không kết luận H2 luôn tốt hơn, hoặc riêng ReLU/Dropout tạo cải thiện khi chưa có bằng chứng.

### Hiện đã có kết quả nào?

Đã kiểm tra cơ chế forward/gradient/update và freeze/BatchNorm bằng weights ngẫu nhiên.
Chưa có kết quả train CIFAR-10 để kết luận head nào chính xác hơn.
