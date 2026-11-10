# Coursework — Phân loại CIFAR-100 bằng CNN và ViT

Trạng thái: đã triển khai tải CIFAR-100, split, EDA và HTML explorer.
Model/train/evaluate/demo còn rỗng. Notebook chưa có cell hoặc kết quả.
Tạm dừng triển khai mới, chờ giảng viên xác nhận quan hệ giữa ViT classification và Anti Face Deepfake.
Kế hoạch CIFAR-100 được lưu làm phương án hiện tại, không xem là phạm vi đã được giảng viên xác nhận.

- [Define — phân tích yêu cầu](docs/define/Requirements_Analysis.md).
- [Design — kế hoạch, chiến lược train và cấu trúc thư mục](docs/design/Implementation_Plan.md).
- [Chiến lược train/tuning — ma trận lượt chạy và cách chọn LR](docs/design/Training_Strategy.md).
- [Model CNN baseline — ResNet-50](docs/design/ResNet50.md).
- [Model chính — ViT-B/16](docs/design/ViT_B16.md).
- [Tiến độ](docs/progress.md).
- [Ghi nhận hỗ trợ AI](docs/AI_usage.md).
- [Audit log smoke run](docs/Audit_Log.md) và [changelog code/pipeline](docs/CHANGELOG.md).

## Chạy dữ liệu và EDA

Từ thư mục gốc workspace, dùng môi trường Python có các thư viện trong requirements.txt:

```powershell
.venv/Scripts/python.exe -m pip install -r coursework_vit_classification/requirements.txt
.venv/Scripts/python.exe -u coursework_vit_classification/src/1_eda.py
.venv/Scripts/python.exe -m http.server 8000 --bind 127.0.0.1
```

Mở [HTML explorer](http://127.0.0.1:8000/coursework_vit_classification/visualization/explorer.html)
trong Chrome/Edge hiện đại; có thể dùng Live Server từ workspace. Không mở bằng file://.
Lần đầu cần mạng để tải archive chính thức; lần sau torchvision kiểm tra và dùng dữ liệu local.
Split được giữ nguyên; script từ chối seed/kích thước/phân bố không hợp lệ thay vì tự ghi đè.
EDA xuất outputs/eda.json và RGB gzip trong data/CIFAR100_web/ để duyệt đủ 60.000 ảnh.
outputs/projection.json chứa PCA toàn bộ dataset và t-SNE mẫu train cân bằng 5.000 ảnh.
Trong HTML, mục Không gian ảnh hỗ trợ chọn nhiều lớp, zoom/kéo và hover xem ảnh gốc.

Xem [workflow dữ liệu](docs/design/Data_EDA.md) để biết thống kê và giới hạn.
Pipeline tiếp theo: train → evaluate → demo; mã dùng chung ở src/.
Dữ liệu đặt ở data/ và không commit. Kết quả chỉ sinh trong outputs/ khi chạy thực tế.
Chưa có checkpoint, số liệu thực nghiệm hoặc hướng dẫn chạy hoàn chỉnh.
