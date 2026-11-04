# data.py — Dataset, split và DataLoader

Source: [data.py](../../src/data.py).
Module dùng chung cho EDA/train/evaluate; import không tự tải hoặc tạo loader.

## Đầu vào và đầu ra

Đọc CIFAR-10 trong data/ và cấu hình DATA. Trả ba loader train/validation/test.
Ảnh gốc RGB32×32, nhãn0–9; sau preprocessing tensor float32 `[B,3,224,224]`.
Split mặc định45.000/5.000 từ50.000 ảnh train chính thức; test10.000 giữ riêng.

## load_datasets(download=False)

Tạo hai torchvision.datasets.CIFAR10 cho train gốc và test, chưa gán transform.
EDA đọc .data/.targets để xem ảnh thô. download=False cần dataset có sẵn;
download=True cho torchvision tải/kiểm tra dữ liệu khi cần.
Không có script tải riêng; có thể chuẩn bị từ thư mục src bằng:

```powershell
python -c "from data import load_datasets; load_datasets(download=True)"
```

## load_split(targets, settings=None, split_file=SPLIT_FILE)

1. Nếu JSON tồn tại: đọc và so seed, val_size, toàn bộ nhãn với dữ liệu hiện tại.
2. Nếu chưa có: train_test_split dùng stratify và seed cố định để chia ID.
3. Kiểm tra ID phủ mỗi mẫu đúng một lần; số validation khớp cấu hình.
4. Kiểm tra tỷ lệ từng lớp ở cả hai nhóm, sai lệch không quá một mẫu.
5. Chỉ ghi JSON khi file chưa tồn tại; trả train_indices và val_indices.

Mỗi lớp mặc định có4.500 train,500 validation. ID của hai nhóm đều tham chiếu
mảng train gốc50.000; test dùng ID riêng0–9.999.
Split lỗi làm dừng, không tự ghi đè. File lưu nhãn/ID, không lưu pixel.

## build_loaders(settings=None, download=False, training=True)

Tạo dataset train và dataset validation riêng, dù cùng đọc bộ train gốc,
để transform ngẫu nhiên không lan sang validation.
Resize224×224 bằng bilinear/antialias; ToTensor đổi RGB về[0,1];
Normalize theo từng kênh bằng `(x-mean)/std` ImageNet.

- training=True: train thêm RandomHorizontalFlip mặc định p=0.5, shuffle=True.
- training=False: train cũng dùng transform đánh giá và shuffle=False.
- Validation/test luôn không augmentation, không shuffle.
- Subset giữ đúng thứ tự ID đã lưu; generator train dùng seed cố định.
- seed_worker lấy torch.initial_seed để đặt seed random/NumPy trong worker.
- batch/workers/pin_memory lấy settings; mặc định64/0/False.

build_loaders còn gọi torch.manual_seed, nên có tác động lên RNG toàn cục.
Nếu tăng workers trên Windows, chạy qua train/evaluate có main guard.
Không có persistent_workers/prefetch tùy chỉnh hay tự đổi batch theo VRAM.

Liên quan: [config](config.md), [EDA](1_eda.md), [evaluate](3_evaluate.md).
