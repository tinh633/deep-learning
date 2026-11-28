# Coursework — Phân loại CIFAR-100 bằng CNN và ViT

Trạng thái: đã triển khai tải CIFAR-100, split, EDA và HTML explorer.
Đã có pipeline, monitor và HTML trình bày; Coursework_Report.ipynb còn rỗng, notebook demo chưa hoàn thiện.
Giảng viên đã xác nhận hướng ViT classification trên CIFAR-100; không kết hợp Anti Face Deepfake.
Cập nhật 28/11/2026: A/B/C đủ 2 pilot × 5 epoch + final 20 epoch; best val 81,46% / 89,70% / 82,08%.
B/C do các thành viên khác huấn luyện; local chỉ tích hợp/kiểm tra, giữ nguyên các gói B_ban_giao và C_ban_giao.
Cặp hiện tại: ConvNeXt-Base/ViT-B16, khoảng 87,67/85,88 triệu tham số (chênh 2,09%).
ConvNeXt A/B đã có smoke batch16 đạt; D/GPU vẫn cần kiểm chứng, số đo ResNet cũ không đại diện cặp mới.

- [Define — phân tích yêu cầu](docs/define/Requirements_Analysis.md).
- [Design — kế hoạch, chiến lược train và cấu trúc thư mục](docs/design/Implementation_Plan.md).
- [Chiến lược train/tuning — ma trận lượt chạy và cách chọn LR](docs/design/Training_Strategy.md).
- [Phân chia công việc — quy trình và gói bàn giao](docs/design/Work_Allocation.md).
- [Model CNN baseline — ConvNeXt-Base](docs/design/ConvNeXt_Base.md).
- [Model chính — ViT-B/16](docs/design/ViT_B16.md).
- [Tiến độ](docs/progress.md).
- [Chỉ mục runs — phân biệt smoke, pilot và final](outputs/runs/README.md).
- [HTML trình bày ViT theo đề bài PDF](visualization/presentation.html): đọc, trình chiếu, in PDF; demo còn bổ sung.
- [Hướng dẫn trang trình bày và phần còn thiếu](docs/design/Presentation.md).
- [Kết quả Exp A — pilot, final và kiểm chứng checkpoint](docs/A_Results.md).
- [Kết quả A/B/C, provenance và giới hạn kiểm chứng](docs/Experiment_Results.md).
- [Ghi nhận hỗ trợ AI](docs/AI_usage.md).
- [Audit log smoke run](docs/Audit_Log.md) và [changelog code/pipeline](docs/CHANGELOG.md).
- [Source workflow — 5W1H từng file](docs/src/README.md).
- [Checkpoint Git LFS — tải, phạm vi và cách sử dụng](docs/Checkpoints.md).
- [Training Monitor — biểu đồ log huấn luyện](Training_Monitor.ipynb): Run All với kernel .venv; lọc run, đọc history mới sau mỗi epoch, tự cập nhật tùy chọn.

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
Đã có A final lan_14, C nhập ở lan_15–19 và B nhập ở lan_20–23; D chưa có gói tích hợp.
B final best val 89,70%; [mapping B](outputs/import_B_manifest.json), [mapping C](outputs/import_C_manifest.json).
Gói C có test nguồn; nhập gói không chạy train/test mới và không dùng test để tuning.
Checkpoint trong các run smoke chỉ phục vụ kiểm chứng kỹ thuật, không phải model đã train đầy đủ.

## Kiểm tra smoke

```powershell
.venv/Scripts/python.exe -u coursework_vit_classification/src/0_smoke.py
```

Smoke cần đặt PHASE=smoke; mặc định mini batch 2, workers 0, 20 train/4 val, không mở test.
Mỗi chiến lược tạo một lan_N mới; kết luận có bằng chứng nằm trong docs/Audit_Log.md.
Logger dừng khi RAM/Windows commit thấp; không khởi chạy nhiều smoke process đồng thời.
Guard đã bổ sung sau checkpoint/replay; mới kiểm tra hồi quy CPU tổng hợp, chưa recheck D trên GPU.

## Train, evaluate và demo

Xem [train](docs/src/2_train.md), [evaluate](docs/src/3_evaluate.md), [demo](docs/src/4_demo.md).
Config giữ A_final/lan_14; TRAINING_APPROVED=False để chặn chạy lặp. Evaluate mặc định validation, không test.
Test cần RUN_TEST=True, checkpoint final; demo cần chỉ định ảnh và run, không tải/train ngầm.

Kiểm tra pipeline CPU không tải weights/dataset:

```powershell
.venv/Scripts/python.exe coursework_vit_classification/tests/test_pipeline.py
```

Model/dữ liệu ở bài kiểm tra là tổng hợp, không dùng làm số liệu CIFAR-100.
Mô tả tình huống, cách đọc kết quả và giới hạn: [test_pipeline.md](docs/src/test_pipeline.md).
