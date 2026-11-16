# data.py — Dữ liệu và loaders

Source: [data.py](../../src/data.py). Workflow EDA: [Data_EDA](../design/Data_EDA.md).

| 5W1H | Mô tả |
|---|---|
| What | Nạp dataset, split, transforms và DataLoader |
| Why | Thí nghiệm dùng cùng IDs, nhãn và preprocessing |
| Which | CIFAR100 của torchvision, fine labels 100 lớp, split seed 42 và ImageNet normalization |
| When | Chỉ khi gọi hàm; import không tải dữ liệu |
| Where | CIFAR-100 trong data/, split trong outputs/ |
| How | torchvision CIFAR100, stratified split, Subset và DataLoader |

## Hàm và dữ liệu

| Hàm | Đầu vào → đầu ra |
|---|---|
| load_datasets(download) | Official train/test cho EDA |
| load_split(targets) | Train IDs/validation IDs, có kiểm tra seed/kích thước/trùng/cân bằng |
| build_loaders(batch_size, num_workers, train_images, val_images) | Train/val loaders và class order |
| evaluation_transform() | Resize/tensor/normalize xác định, dùng chung evaluate/demo |
| build_evaluation_loader(split, batch_size, num_workers, maximum) | Loader, classes và IDs theo split |
| preprocessing_settings() | Metadata preprocessing để đối chiếu checkpoint |

Train: resize 224, horizontal flip, ToTensor, ImageNet normalize.
Val/test: không flip ngẫu nhiên. Thống kê RGB CIFAR không tự thay ImageNet normalization.
Train/val chia sẻ pixel array qua bản sao nhẹ dataset; không nhân đôi toàn dữ liệu trong RAM.
Loader train có generator seed 42, shuffle; val/evaluate không shuffle.
Smoke giới hạn số ảnh bằng chọn không hoàn lại từ IDs train/val; chưa phải mẫu đánh giá accuracy.
build_loaders không nạp official test; evaluation loader chỉ nạp test khi được script evaluate cho phép.
Train/validation evaluation vẫn dùng transform xác định, không dùng augmentation như online training.
maximum dùng vài IDs đầu để kiểm tra; report phải ghi số lượng giới hạn, không giả thành toàn split.

## Giới hạn

Split hiện kiểm tra cứng 450 train/50 val mỗi lớp; lọc duplicate cần version split và code phù hợp.
Script giữ nguyên IDs/ảnh đã có, không tự sửa dữ liệu khi kiểm tra thất bại.
