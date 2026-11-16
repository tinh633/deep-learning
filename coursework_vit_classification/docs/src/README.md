# Source workflow

Cập nhật: 16/11/2026. Tài liệu source nằm cùng cấp nhóm với docs/define và docs/design.
Mỗi file dưới đây mô tả 5W1H, input/output, luồng xử lý, cách chạy và giới hạn theo code hiện có.
Các mục dùng What, Why, Which, When, Where, How; Which nêu lựa chọn model/dataset/công cụ/cấu hình của file.

| File Python | Tài liệu | Trách nhiệm |
|---|---|---|
| config.py | [config.md](config.md) | Giá trị chung và các lựa chọn chưa được chốt |
| data.py | [data.md](data.md) | Dataset, split, transforms và loaders |
| model.py | [model.md](model.md) | ResNet/ViT, freeze/fine-tune, nạp checkpoint |
| logger.py | [logger.md](logger.md) | Log, RAM guard, checkpoint và provenance |
| 0_smoke.py | [0_smoke.md](0_smoke.md) | Kiểm tra kỹ thuật trước tuning |
| 1_eda.py | [1_eda.md](1_eda.md) | Tải/khảo sát dữ liệu và HTML artifacts |
| 2_train.py | [2_train.md](2_train.md) | Một lượt tuning hoặc final, best/last và resume |
| 3_evaluate.py | [3_evaluate.md](3_evaluate.md) | Metric, ROC/AUC/AP và predictions theo split |
| 4_demo.py | [4_demo.md](4_demo.md) | Dự đoán một ảnh bằng checkpoint được chọn |

Kiểm tra hỗ trợ ngoài src/: [tests/test_pipeline.py](test_pipeline.md) — regression CPU tổng hợp.

Thứ tự: EDA → smoke → khóa protocol → pilot/final → evaluate → demo/report.
Không import script đánh số để dùng lại hàm; các module chung không tự chạy pipeline.
Train/evaluate có main guard để an toàn với worker trên Windows.

Đã có code khác với đã train: chưa chạy tuning/final hoặc official test trên model thật.
PHASE mặc định smoke, TRAINING_APPROVED=False; 2_train.py yêu cầu cấu hình rõ ràng.
Các bài kiểm tra CPU dùng dữ liệu/model tổng hợp không được tính là kết quả CIFAR-100.

Thiết kế: [Implementation_Plan](../design/Implementation_Plan.md).
Kiểm chứng thực: [Audit_Log](../Audit_Log.md); thay đổi: [CHANGELOG](../CHANGELOG.md).
