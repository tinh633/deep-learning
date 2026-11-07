# Các model trong benchmark

Tài liệu giới thiệu sáu kiến trúc, pretrained weights và cách sử dụng trong bài.
Cấu hình thực tế nằm ở [config.py](../src/config.py); không có model.py riêng vì model lấy trực tiếp từ torchvision.
Luồng đo chi tiết: [1_benchmark.md](src/1_benchmark.md).

## 1. Model đã train sẵn, không train lại

- Cả sáu dùng pretrained weights ImageNet-1K, enum IMAGENET1K_V1 tương ứng từng model.
- Không fine-tune, không thay head và không cập nhật trọng số trong benchmark.
- Chạy model.eval() và torch.inference_mode(), không tính gradient.
- Mỗi model dùng transforms đi kèm weights; classifier vẫn dự đoán 1.000 lớp.
- Dù ảnh đánh giá thuộc 10 lớp, dự đoán sang 990 lớp khác vẫn bị tính sai.
- Cùng nguồn ImageNet-1K không đồng nghĩa cùng recipe hoặc điều kiện huấn luyện.

Luồng chung: ảnh → RGB → preprocessing theo weights → CNN → 1.000 logits → softmax → Top-1/Top-5.
Softmax tạo điểm xác suất tương đối trên 1.000 lớp, không bảo đảm confidence đã được hiệu chỉnh.

## 2. Tổng quan kiến trúc

| Model | Ý tưởng chính | Tham số thực đo, triệu | Vai trò đối chiếu |
|---|---|---:|---|
| VGG16 | Xếp chồng convolution 3 × 3 và head fully connected lớn | 138,358 | CNN tuần tự với nhiều tham số |
| GoogLeNet | Inception: nhiều nhánh xử lý rồi concatenate | 6,625 | Đặc trưng nhiều quy mô trong một block |
| ResNet50 | Residual connection cộng đầu vào với nhánh học đặc trưng | 25,557 | Backbone sâu với đường truyền tắt |
| DenseNet121 | Nối đặc trưng các layer trước bằng concatenate | 7,979 | Tái sử dụng đặc trưng |
| EfficientNet-B0 | MBConv, squeeze-and-excitation; bản cơ sở họ EfficientNet | 5,289 | Cân bằng chất lượng và chi phí |
| MobileNetV2 | Depthwise convolution, inverted residual và linear bottleneck | 3,505 | CNN hướng tới tài nguyên hạn chế |

Số tham số lấy từ [results.csv của lan_1](../outputs/lan_1/results.csv), gồm head 1.000 lớp đang dùng.
Ít tham số không tự động đồng nghĩa chạy nhanh hơn hoặc dùng ít peak VRAM hơn.

## 3. VGG16

- Gồm 13 convolution và 3 lớp Linear có tham số; tên 16 không tính pooling/ReLU.
- Luồng chính: các nhóm Conv/ReLU/MaxPool → adaptive pooling → flatten → classifier.
- Các convolution 3 × 3 được xếp sâu để xây dựng đặc trưng từ đơn giản đến phức tạp.
- Head Linear lớn góp phần đáng kể vào tổng số tham số.
- Bài dùng vgg16 thường, không phải vgg16_bn có BatchNorm.
- Khi đọc kết quả, xem đồng thời accuracy, thời gian và VRAM thay vì chỉ số layer.

Nguồn: [torchvision VGG16](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.vgg16.html).

## 4. GoogLeNet

- Là kiến trúc Inception v1, không phải Inception v3 hoặc Inception-ResNet.
- Một Inception block có các nhánh convolution/pooling, ghép đầu ra theo chiều channel.
- Convolution 1 × 1 được dùng để biến đổi/giảm số channel trước một số phép tính lớn hơn.
- Sau các block, global average pooling tổng hợp đặc trưng rồi Linear dự đoán lớp.
- Kiến trúc có cơ chế auxiliary classifier phục vụ huấn luyện; benchmark chỉ chấm đầu ra chính ở eval.
- Tính nhiều nhánh có thể ảnh hưởng tốc độ thực; không suy ra chỉ từ số tham số.

Nguồn: [torchvision GoogLeNet](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.googlenet.html).

## 5. ResNet50

- Dùng bottleneck block với chuỗi convolution 1 × 1 → 3 × 3 → 1 × 1.
- Residual block kết hợp đặc trưng theo phép cộng: y = F(x) + shortcut(x).
- Shortcut có thể là identity hoặc projection khi cần đổi kích thước/channel.
- Đường tắt hỗ trợ tối ưu mạng sâu; không có nghĩa model bỏ qua việc học ở các block.
- Sau backbone là global average pooling và Linear 1.000 lớp.
- Đây là ResNet50 của torchvision, không phải tên ResNet50V2 từ thư viện khác.

Nguồn: [torchvision ResNet50](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html).

## 6. DenseNet121

- Trong dense block, layer nhận các đặc trưng trước đó thông qua concatenate.
- Khác ResNet: DenseNet ghép channel, còn residual connection chủ yếu cộng tensor.
- Transition layer dùng convolution/pooling để giảm quy mô giữa các dense block.
- Cuối mạng: normalization/ReLU → global average pooling → classifier.
- Tái sử dụng đặc trưng giúp giảm nhu cầu học lại, nhưng nhiều phép nối/truy cập bộ nhớ có chi phí.
- Vì vậy ít tham số hơn ResNet50 không bảo đảm latency thấp hơn.

Nguồn: [torchvision DenseNet121](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.densenet121.html).

## 7. EfficientNet-B0

- B0 là bản cơ sở của họ EfficientNet, không phải EfficientNetV2.
- Dùng MBConv với depthwise convolution và squeeze-and-excitation để điều chỉnh trọng số channel.
- Họ EfficientNet nghiên cứu mở rộng đồng thời depth, width và resolution; benchmark chỉ chạy B0 cố định.
- Cuối backbone: global average pooling, dropout và Linear; dropout được tắt khi eval.
- Hiệu quả triển khai phải được kiểm tra trên phần cứng thực tế, không chỉ dựa vào thiết kế tiết kiệm phép tính.

Nguồn: [torchvision EfficientNet-B0](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b0.html).

## 8. MobileNetV2

- Inverted residual mở rộng channel ở giữa block rồi chiếu về biểu diễn hẹp.
- Depthwise convolution xử lý từng channel riêng; pointwise convolution kết hợp channel.
- Linear bottleneck tránh activation phi tuyến tại đầu ra projection hẹp của block.
- Skip connection dùng khi hình dạng đầu vào/đầu ra phù hợp.
- Model này là MobileNetV2, không phải MobileNetV1 hoặc V3.
- Là lựa chọn để kiểm tra model nhỏ có đạt cân bằng accuracy/tốc độ/bộ nhớ phù hợp hay không.

Nguồn: [torchvision MobileNetV2](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.mobilenet_v2.html).

## 9. Đọc kết quả trong phạm vi bài

Trên lan_1, cùng 500 ảnh validation/10 lớp và RTX3050 Laptop:

- EfficientNet-B0 có Top-1 cao nhất mẫu: 76,2% (381/500).
- MobileNetV2 có latency batch 1 thấp nhất và throughput batch 16 cao nhất trong lượt đo.
- GoogLeNet có peak allocated VRAM batch 16 thấp nhất; VGG16 cao nhất.

Đây là kết quả của các model/weights cụ thể, không chứng minh kiến trúc nào luôn tốt nhất.
Các chênh lệch nhỏ còn phụ thuộc mẫu ảnh, preprocessing, recipe và điều kiện máy.
Không dùng kết quả smoke_1/ImageNetV2 để thay kết quả lan_1/ImageNet-1K.
