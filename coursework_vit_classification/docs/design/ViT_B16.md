# ViT-B/16 — Model chính

Thiết kế cho coursework, chưa có implementation/model checkpoint CIFAR-100.
Tham chiếu: [Define](../define/Requirements_Analysis.md), [Design](Implementation_Plan.md), [CNN baseline](ResNet50.md).

## 1. Vai trò và tên model

ViT là Vision Transformer, xử lý chuỗi patch ảnh bằng Transformer encoder.
B là Base: 12 encoder block, hidden size 768, 12 attention head và MLP hidden size 3072.
16 là kích thước patch 16 × 16, không phải số block hoặc số lớp phân loại.
Model này đáp ứng hướng ViT image classification; không dùng kiến trúc DeiT hoặc Swin thay thế.

## 2. Phiên bản và nguồn weights

- Constructor: torchvision.models.vit_b_16.
- Weights: ViT_B_16_Weights.IMAGENET1K_V1.
- Weights được torchvision train trên ImageNet-1K với recipe điều chỉnh từ DeiT.
- Recipe huấn luyện tham khảo DeiT không biến kiến trúc thành DeiT distilled.
- Không có distillation token/teacher trong model và pipeline coursework này.
- Không nhầm với google/vit-base-patch16-224 vốn có lịch sử pretraining ImageNet-21K.
- Chỉ thay heads.head thành Linear(768, 100); giữ patch embedding và encoder pretrained.

Nguồn: [ViT-B/16 weights](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.vit_b_16.html),
[mã nguồn torchvision](https://docs.pytorch.org/vision/main/_modules/torchvision/models/vision_transformer.html).

## 3. Luồng xử lý và kích thước tensor

| Bước | Kích thước | Ý nghĩa |
|---|---|---|
| Input RGB | B × 3 × 224 × 224 | Ảnh CIFAR-100 sau preprocessing |
| Patch projection Conv2d kernel/stride 16 | B × 768 × 14 × 14 | Chiếu mỗi patch sang vector 768 chiều |
| Reshape và hoán vị | B × 196 × 768 | Chuỗi 14 × 14 = 196 patch |
| Thêm CLS token | B × 197 × 768 | Token học được phục vụ phân loại |
| Cộng positional embedding | B × 197 × 768 | Thông tin vị trí của các token |
| 12 encoder block, final LayerNorm | B × 197 × 768 | Trao đổi thông tin toàn chuỗi |
| Lấy CLS token đầu chuỗi | B × 768 | Biểu diễn phục vụ phân loại |
| heads.head mới | B × 100 | Logits CIFAR-100 |

Mỗi encoder block có LayerNorm → multi-head self-attention → residual,
sau đó LayerNorm → MLP với GELU → residual.
Attention cho phép patch tổng hợp thông tin từ các patch khác, không chỉ vùng lân cận.
Positional embedding và CLS token là tham số được học, không phải nhãn hoặc tọa độ EDA.
ViT này dùng CLS để phân loại, không lấy trung bình toàn bộ patch token thay thế.
Patch projection dùng Conv2d là cách triển khai linear projection từng patch, không phải CNN backbone lai.

## 4. Hai chế độ huấn luyện

| Nội dung | C — Freeze | D — Fine-tune toàn bộ |
|---|---|---|
| Patch projection | Freeze | Train |
| CLS/positional embedding | Freeze | Train |
| Encoder và final LayerNorm | Freeze | Train |
| Head Linear mới | Train | Train |
| Optimizer | Chỉ head | Tất cả parameters được mở |
| Khởi tạo | Pretrained + head mới | Pretrained + head mới, không tiếp tục từ C |

Trong C, backbone giữ eval; head train. Trong D, model ở train khi học.
LayerNorm khác BatchNorm: không có running mean/variance kiểu BatchNorm.
Val/test luôn eval và không gradient; nếu cấu hình dropout thì eval tắt dropout.

## 5. Tham số và tài nguyên

Tổng với head ImageNet 1.000 lớp: 86.567.656 theo metadata torchvision.
Head mới: 768 × 100 + 100 = 76.900 tham số.
Tổng sau thay head dự kiến: 85.875.556; trainable C = 76.900, D = toàn bộ tổng này.
Số liệu này tính từ cấu hình, chưa phải log của một run đã thực hiện.
Fine-tune cần lưu gradient, optimizer state và activation ngoài model weights.
GPU 4 GB có rủi ro thiếu VRAM; batch nhỏ/AMP cần smoke kiểm chứng, không hứa chạy được trước khi đo.
Resize CIFAR-100 lên 224 phù hợp cấu hình pretrained nhưng không làm ảnh gốc giàu chi tiết hơn.

## 6. Kiểm thử trước train chính thức

- Xác nhận input 224, 196 patch + CLS và output B × 100.
- Kiểm tra chỉ head có gradient ở C; D mở cả CLS, vị trí, projection và encoder.
- Đảm bảo optimizer nhận các parameters được mở; không softmax trước CrossEntropyLoss.
- Kiểm tra save/load và resume tái tạo đúng cấu hình, class order và split.
- Đo thời gian/VRAM của C/D; khóa batch/precision sau smoke run.
- Không dùng checkpoint head ImageNet 1.000 lớp để chấm nhãn CIFAR-100 trực tiếp.

## 7. Attention và cách diễn giải

Visualize attention là phần mở rộng, không điều kiện để bắt đầu bốn thí nghiệm cốt lõi.
Forward torchvision mặc định không trả attention weights; muốn xem cần thiết kế lấy attention đúng.
Phải ghi rõ layer/head hay attention rollout, thay vì coi mọi heatmap là cùng một đại lượng.
Heatmap chỉ hỗ trợ quan sát, không chứng minh vùng ảnh là nguyên nhân duy nhất của dự đoán.
PCA/t-SNE RGB hiện có trong EDA không phải attention hoặc embedding của ViT.

## 8. Kết quả cần báo cáo

So C/D để đo lợi ích fine-tune; đối chiếu A/C và B/D theo cùng protocol.
Trình bày accuracy/macro-F1, lỗi theo lớp và thời gian/VRAM bên cạnh chênh lệch tham số.
ViT không bắt buộc thắng CNN; chưa có kết quả CIFAR-100 thực đo trong coursework.
