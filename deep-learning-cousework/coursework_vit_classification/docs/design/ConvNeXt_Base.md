# ConvNeXt-Base — CNN baseline

Cặp hiện tại: ConvNeXt-Base và [ViT-B/16](ViT_B16.md), pretrained ImageNet-1K.
Thay ResNet-50 từ 18/11/2026; kết quả ResNet cũ chỉ là lịch sử kỹ thuật.

## 1. Vai trò và lựa chọn

ConvNeXt là CNN hiện đại, dùng depthwise convolution và residual, không có self-attention.
Chọn Base để gần quy mô ViT-B/16 thay vì dùng CNN nhỏ hơn nhiều lần.
Sau head CIFAR-100: 87.668.964 so với 85.875.556 tham số, chênh khoảng 2,09% so với ViT.
Đây là gần tương đương, chưa bằng tuyệt đối; không tuyên bố đã đạt điều kiện bằng chính xác.
Recipe pretraining vẫn khác nhau; kết quả là so sánh hai model pretrained cụ thể.

## 2. Phiên bản và weights

- Constructor: torchvision.models.convnext_base.
- Weights: ConvNeXt_Base_Weights.IMAGENET1K_V1, không dùng ImageNet-21K.
- Cache local có checksum; không tải weights trong kiểm thử tổng hợp.
- Thay riêng classifier[2] thành Linear(1024, 100).
- Giữ classifier[0] LayerNorm2d pretrained và classifier[1] Flatten.

Nguồn: [Torchvision model](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.convnext_base.html),
[implementation](https://docs.pytorch.org/vision/main/_modules/torchvision/models/convnext.html).

## 3. Luồng xử lý

| Bước | Tensor với ảnh 224×224 | Vai trò |
|---|---|---|
| Stem convolution 4×4, stride 4 + LayerNorm | B×128×56×56 | Giảm kích thước ảnh |
| Stage 1, 3 block | B×128×56×56 | Đặc trưng cục bộ |
| Downsample + stage 2, 3 block | B×256×28×28 | Tăng channels |
| Downsample + stage 3, 27 block | B×512×14×14 | Nhóm xử lý sâu nhất |
| Downsample + stage 4, 3 block | B×1024×7×7 | Đặc trưng cuối |
| AdaptiveAvgPool2d(1) | B×1024×1×1 | Tổng hợp không gian |
| LayerNorm2d + Flatten | B×1024 | Chuẩn bị vector |
| Linear(1024, 100) mới | B×100 | Logits CIFAR-100 |

Mỗi block: depthwise conv 7×7 → LayerNorm → Linear mở rộng 4 lần → GELU → Linear thu hẹp,
sau đó layer scale, stochastic depth và cộng residual. Không softmax trước CrossEntropyLoss.

## 4. Freeze và fine-tune

| Thành phần | A — Freeze | B — Fine-tune |
|---|---|---|
| Features và LayerNorm cuối pretrained | Freeze | Train |
| Linear classifier[2] mới | Train | Train |
| Trainable parameters | 102.500 | 87.668.964 |
| Mode khi train | Model eval, riêng Linear train | Toàn model train |
| Stochastic depth | Tắt | Bật theo cấu hình torchvision |

ConvNeXt dùng LayerNorm, không có running mean/variance như BatchNorm của ResNet cũ.
Val/test luôn eval, không gradient. B khởi tạo mới từ pretrained, không nối tiếp A.
Freeze giữ cả LayerNorm cuối bất biến; smoke kiểm tra mọi tensor ngoài classifier[2].

## 5. Tài nguyên và nghiệm thu

Head ImageNet: 1024×1000+1000; thay bằng 1024×100+100 để tính tổng mới.
Không gắn thêm tham số vô dụng để tạo số lượng bằng ViT.
Fine-tune cần gradient, activation và optimizer state; gần số tham số không có nghĩa cùng VRAM/tốc độ.
Máy GPU 4 GB phải smoke lại A/B; số đo ResNet cũ không dùng ước tính ConvNeXt.
Kiểm thử cấu trúc xác nhận head, tham số, freeze/mode và logits B×100 trên meta device.
Regression tổng hợp xác nhận orchestration/resume; không thay thế smoke GPU pretrained thật.
A đã hoàn tất 2 pilot và final 20 epoch, batch16, LR chọn 3e-4; xem [A_Results](../A_Results.md).
B fine-tune vẫn cần smoke/budget riêng; không dùng số đo A để khẳng định B đủ tài nguyên.
