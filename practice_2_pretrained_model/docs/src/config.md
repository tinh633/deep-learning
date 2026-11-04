# config.py — Cấu hình dùng chung

Source: [config.py](../../src/config.py).
Được data, EDA, train và evaluate import; không tự train hoặc tải dataset.

## Các nhóm cấu hình

| Nhóm | Giá trị hiện tại và tác dụng |
|---|---|
| Đường dẫn | PROJECT_FOLDER suy từ vị trí file; data/, outputs/, outputs/cifar10_split.json |
| Model | MODEL_NAME=vgg16, PRETRAINED=True |
| Thí nghiệm | EXPERIMENT=A; tra bảng EXPERIMENTS để lấy fine_tune/lr/batch |
| Train | SEED=42, EPOCHS=20, WEIGHT_DECAY=0 |
| Dữ liệu | IMAGE_SIZE=224, VAL_SIZE=5000, NUM_WORKERS=0, PIN_MEMORY=False |
| Resume | RESUME_RUN=None tạo lượt mới; tên lan_N để tiếp tục lượt cũ |
| Evaluate | EVAL_RUN=lan_1, RUN_TEST=False |

A: freeze backbone, LR0.001; B: mở nhóm cuối, LR0.001;
C: mở nhóm cuối, LR0.0001. Cả ba dùng batch64 và head Linear10.
H1 chỉ là tên metadata, không có lựa chọn head thứ hai trong model hiện tại.
MODEL_NAMES liệt kê vgg16/densenet121; không phải danh sách tự chạy nối tiếp.

## Cách hoạt động

FINE_TUNE, LEARNING_RATE, BATCH_SIZE được tính khi import từ hàng A/B/C đã chọn.
DATA gom các tham số loader, được lưu vào checkpoint để tái sử dụng khi đánh giá.
IMAGENET_MEAN/STD là hằng chuẩn hóa cho weights ImageNet-1K;
thống kê RGB tính trong EDA không tự thay thế hai bộ hằng này.

get_settings() đọc bytes của split và tính SHA256, trả dictionary gồm model,
thí nghiệm, train/data config, nguồn CIFAR10, input_shape và dấu nhận diện split.
Hàm không tạo file split; train gọi sau build_loaders để split đã tồn tại.
File thiếu làm hàm dừng; chỉ thay cách xuống dòng cũng có thể đổi SHA.

## Khi chỉnh tham số

Chọn MODEL_NAME và EXPERIMENT trước khi chạy train; script chỉ chạy một tổ hợp.
Đánh giá lấy model/settings từ checkpoint, không từ MODEL_NAME đang chọn.
EVAL_RUN và RUN_TEST chỉ điều khiển run/split cần đánh giá.
Resume kiểm tra settings khớp; đổi epochs, batch, workers hoặc LR có thể bị từ chối.
Không tăng EPOCHS để kéo dài run bằng resume mà chưa thay thiết kế kiểm tra.

Liên quan: [data](data.md), [train](2_train.md), [logger](logger.md).
