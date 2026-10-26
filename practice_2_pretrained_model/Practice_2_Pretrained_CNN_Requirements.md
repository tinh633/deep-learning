# Practice 2 – Thực hành với các kiến trúc mạng Neural Network đã được huấn luyện trước

## 1. Yêu cầu bài tập (Exercises)

1. **Thử nghiệm với các mô hình Pre-trained khác nhau:** VGG, DenseNet, v.v.
2. **Điều chỉnh các siêu tham số (Hyperparameters):** Learning rate, batch size, v.v.
3. **Sử dụng TensorBoard:** Theo dõi quá trình huấn luyện mô hình.

## 2. Mục tiêu (Objectives)

- Làm quen với việc tải và sử dụng các mô hình đã được huấn luyện trước từ `torchvision.models`.
- Hiểu cách điều chỉnh và fine-tune các mô hình Pre-trained cho một bài toán cụ thể.

## 3. Các bước thực hiện (Steps)

### Bước 1: Thiết lập môi trường

- Đảm bảo đã cài đặt PyTorch và torchvision.
- Nếu chưa, cài đặt bằng lệnh:

```bash
pip install torch torchvision
```

### Bước 2: Tải mô hình Pre-trained

- `torchvision.models` cung cấp nhiều mô hình như ResNet, VGG, DenseNet, v.v.
- Tải các trọng số đã được huấn luyện trên bộ dữ liệu ImageNet.

Ví dụ:

```python
from torchvision import models

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
```

### Bước 3: Khám phá kiến trúc mô hình

- In mô hình để xem cấu trúc tổng thể.
- Xác định các layer và tham số của mô hình.

### Bước 4: Điều chỉnh mô hình cho bài toán cụ thể

- Thay thế lớp cuối cùng (Classification Layer) để phù hợp với số lượng class trong dataset.
- Có thể **freeze** các layer trước đó để giữ nguyên trọng số khi huấn luyện lại (Transfer Learning).
- Hoặc **unfreeze** một số layer cuối để fine-tune mô hình.

### Bước 5: Chuẩn bị dữ liệu

- Tải và chuẩn bị dataset.
- Sử dụng `torchvision.transforms` để tiền xử lý dữ liệu.
- Sử dụng `torch.utils.data.DataLoader` để tạo các batch dữ liệu.

### Bước 6: Huấn luyện mô hình

- Xác định hàm loss và optimizer.
- Viết training loop để huấn luyện mô hình trên dataset.

### Bước 7: Đánh giá mô hình

- Đánh giá hiệu suất mô hình trên tập test.

## 4. Checklist đề xuất để hoàn thành bài thực hành

| STT | Công việc | Yêu cầu |
|---|---|---|
| 1 | Chọn Dataset | Dataset cho Image Classification |
| 2 | Chọn Models | Nên thử ít nhất VGG và DenseNet |
| 3 | Load Pre-trained | Sử dụng trọng số ImageNet |
| 4 | Transfer Learning | Thay Classification Head, freeze backbone |
| 5 | Fine-tuning | Thử unfreeze một số layer cuối |
| 6 | Hyperparameter Tuning | Thử các learning rate và batch size |
| 7 | Training | Huấn luyện các mô hình |
| 8 | TensorBoard | Theo dõi loss và accuracy trong quá trình train |
| 9 | Evaluation | Đánh giá trên tập test |
| 10 | Comparison | So sánh kết quả giữa các mô hình và cấu hình |

> **Lưu ý:** Slide không quy định cụ thể số lượng mô hình, dataset hoặc mức accuracy phải đạt. Checklist ở mục 4 là đề xuất triển khai, không phải toàn bộ yêu cầu bắt buộc từ giảng viên.

**Trọng tâm:** Sử dụng Pre-trained CNN, thay Classification Head, thực hiện Transfer Learning/Fine-tuning và đánh giá ảnh hưởng của hyperparameters đến hiệu suất mô hình.
