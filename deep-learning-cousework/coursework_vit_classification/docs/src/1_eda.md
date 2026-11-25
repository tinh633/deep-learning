# 1_eda.py — Chuẩn bị và khảo sát

Source: [1_eda.py](../../src/1_eda.py). Thiết kế dữ liệu: [Data_EDA](../design/Data_EDA.md).

| 5W1H | Mô tả |
|---|---|
| What | Tải/integrity, split, thống kê, duplicate và PCA/t-SNE |
| Why | Hiểu dữ liệu trước train và cung cấp artifact cho HTML |
| Which | CIFAR-100, NumPy, SHA-256 exact duplicate, PCA và t-SNE trên đặc trưng RGB pooled |
| When | Trước model/train hoặc khi cần tái sinh local artifact |
| Where | Dataset/RGB gzip trong data/, JSON thống kê/tọa độ trong outputs/ |
| How | torchvision, NumPy, hashing, PCA fit train và t-SNE mẫu train |

```powershell
.venv/Scripts/python.exe -u coursework_vit_classification/src/1_eda.py
```

1. Tải/kiểm tra official CIFAR-100 và dùng split seed 42 đã lưu.
2. Xuất sáu RGB gzip local, không commit pixel và không tạo atlas.
3. Tính phân bố, train RGB, brightness/contrast và exact duplicate trong/giữa split.
4. Pool ảnh xuống RGB 8×8, PCA fit train; chiếu đủ 60.000 ảnh.
5. t-SNE 5.000 ảnh train, 50 ảnh/lớp; nhãn chỉ lấy mẫu/tô màu, không tạo cụm giả.
6. Ghi eda.json, cifar100_split.json và projection.json cho HTML.

Không tự lọc duplicate hoặc thay normalization pretrained.
RGB projection chưa phải embedding model, không chứng minh CNN/ViT phân tách lớp tốt.
Mở HTML qua HTTP/Live Server, không mở file:// để đọc gzip/JSON.
