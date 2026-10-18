# 2. Data preparation

Script: src/1_preparedata.py; tham số DATA trong config.py.
Chạy: `python practice_1_fashionmnist/src/1_preparedata.py`.

- Tải, xác minh MD5 và giải nén IDX vào data/FashionMNIST_raw.
- data.py chứa build_loaders dùng torchvision FashionMNIST, chỉ đọc dataset khi được gọi.
- 1_preparedata.py chạy trực tiếp từng bước tải/kiểm tra, không hàm main hoặc entry guard.
- Stratified split seed 42: 54.000 train / 6.000 validation; giữ test 10.000.
- Mỗi nhãn có 5.400 train / 600 validation / 1.000 test.
- ToTensor: float32 [0,1], ảnh [B,1,28,28]; batch mặc định 64.
- Train shuffle có seed; validation/test tuần tự; chưa augmentation/normalization bổ sung.
- Không dùng thống kê validation/test để fit preprocessing.

## Nguồn và link tải

[Repository chính thức Zalando Research](https://github.com/zalandoresearch/fashion-mnist).

| File | Link tải | MD5 |
|---|---|---|
| Train images | [IDX.gz](https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/train-images-idx3-ubyte.gz) | 8d4fb7e6c68d591d4c3dfef9ec88bf0d |
| Train labels | [IDX.gz](https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/train-labels-idx1-ubyte.gz) | 25c81989df183df01b3e8a0aad5dffbe |
| Test images | [IDX.gz](https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/t10k-images-idx3-ubyte.gz) | bef4ecab320f06d8554ea6380940ec79 |
| Test labels | [IDX.gz](https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/t10k-labels-idx1-ubyte.gz) | bb300cfdad3c16e7a12a480ee83cd310 |

Tiếp theo: [EDA](03_eda.md).
