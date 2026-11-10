# ResNet-50 — CNN baseline

Thiết kế cho coursework, chưa có implementation/model checkpoint CIFAR-100.
Tham chiếu: [Define](../define/Requirements_Analysis.md), [Design](Implementation_Plan.md), [ViT-B/16](ViT_B16.md).

## 1. Vai trò và lý do chọn

ResNet-50 cung cấp mốc đối chiếu CNN pretrained với model chính ViT-B/16.
Residual learning, global average pooling và head Linear giúp giải thích rõ pipeline.
Baseline không có nghĩa model yếu hoặc phải thua ViT; kết luận theo kết quả thực.
Cặp này không cùng số tham số nên chỉ so sánh hai model/weights cụ thể trong transfer learning.

## 2. Phiên bản sử dụng

- Constructor: torchvision.models.resnet50.
- Weights: ResNet50_Weights.IMAGENET1K_V1, pretrained ImageNet-1K.
- Không dùng DEFAULT vì ResNet50 DEFAULT hiện trỏ IMAGENET1K_V2.
- Triển khai torchvision đặt stride downsample ở convolution 3 × 3 của bottleneck.
- Biến thể triển khai này thường gọi ResNet v1.5; không phải ResNet V2 pre-activation.
- Đổi head fc từ 1.000 lớp sang Linear(2048, 100), không train từ đầu backbone.

Nguồn: [torchvision ResNet-50](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html).

## 3. Kiến trúc và kích thước tensor

| Bước | Tensor khi input 224 × 224 | Vai trò |
|---|---|---|
| Input | B × 3 × 224 × 224 | RGB đã resize/normalize |
| Conv 7×7, stride 2; BN/ReLU | B × 64 × 112 × 112 | Đặc trưng cục bộ ban đầu |
| MaxPool | B × 64 × 56 × 56 | Giảm độ phân giải |
| layer1: 3 bottleneck | B × 256 × 56 × 56 | Nhóm residual đầu |
| layer2: 4 bottleneck | B × 512 × 28 × 28 | Tăng channel, giảm kích thước |
| layer3: 6 bottleneck | B × 1024 × 14 × 14 | Đặc trưng sâu hơn |
| layer4: 3 bottleneck | B × 2048 × 7 × 7 | Nhóm đặc trưng cuối |
| AdaptiveAvgPool2d(1), flatten | B × 2048 | Tổng hợp không gian từng channel |
| fc mới | B × 100 | Logits CIFAR-100 |

Một bottleneck có nhánh 1×1 → 3×3 → 1×1, kết hợp đầu vào bằng phép cộng residual.
Shortcut là identity hoặc projection nếu channel/kích thước khác nhau.
BatchNorm chuẩn hóa theo channel; running statistics là buffers, không bị freeze chỉ bởi requires_grad=False.
GAP/flatten không có tham số học; lớp fc mới có tham số và luôn được train trong cả A/B.

## 4. Hai chế độ huấn luyện

| Nội dung | A — Freeze | B — Fine-tune toàn bộ |
|---|---|---|
| Backbone parameters | requires_grad=False | requires_grad=True |
| fc mới | requires_grad=True | requires_grad=True |
| Forward train | Backbone eval, head train | Train; policy BN cần chốt trước run |
| Optimizer | Chỉ fc parameters | Toàn bộ parameters được mở |
| Khởi tạo | Pretrained + head mới | Pretrained + head mới, không nối tiếp A |

Freeze cần giữ BatchNorm ở eval để mean/variance không đổi trong train pass.
Fine-tune phải công bố có cập nhật running statistics hay giữ BN eval do batch nhỏ.
Giữ BN eval không đồng nghĩa đóng băng affine weight/bias nếu chúng vẫn requires_grad=True.
Val/test luôn eval và không gradient; không thay đổi BN buffers khi đánh giá.

## 5. Số tham số và chi phí

Head 1.000 lớp gốc: tổng 25.557.032 tham số theo metadata torchvision.
Head mới: 2048 × 100 + 100 = 204.900 tham số.
Tổng sau thay head dự kiến: 23.712.932; trainable A = 204.900, B = toàn bộ tổng này.
Đây là phép tính từ kiến trúc, cần kiểm tra lại trên model thực khi triển khai.
VRAM còn phụ thuộc activation, gradient, optimizer, batch và precision; không suy từ tham số đơn thuần.

## 6. Preprocessing, kiểm thử và phân tích

- Input chung 224 × 224; normalization theo weights và được ghi trong config.
- Train augmentation phải cùng protocol với ViT; val/test cố định, không lấy thống kê test.
- Kiểm tra output B × 100, head có gradient, backbone A không có gradient và BN buffers không đổi.
- Kiểm tra B mở toàn bộ parameters, policy BN đúng, optimizer chứa đủ trainable parameters.
- Nạp lại state_dict phải tái tạo logits eval tương đương trên cùng input và precision.
- So A/B để đo lợi ích fine-tune; so A/C và B/D với ViT để đối chiếu chất lượng/tài nguyên.
- Không khẳng định residual connection gây ra chênh lệch accuracy khi còn khác recipe và quy mô.

Chưa có số đo CIFAR-100 hoặc GPU của model này trong coursework.
