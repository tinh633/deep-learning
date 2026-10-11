# Nguồn dữ liệu FashionMNIST

- [Repository chính thức — Zalando Research](https://github.com/zalandoresearch/fashion-mnist)
- [Thư mục dữ liệu gốc](https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion)

Link tải trực tiếp được `src/prepare_data.py` sử dụng:

| Nội dung | Link tải | MD5 |
|---|---|---|
| Ảnh train | [train-images-idx3-ubyte.gz](https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/train-images-idx3-ubyte.gz) | `8d4fb7e6c68d591d4c3dfef9ec88bf0d` |
| Nhãn train | [train-labels-idx1-ubyte.gz](https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/train-labels-idx1-ubyte.gz) | `25c81989df183df01b3e8a0aad5dffbe` |
| Ảnh test | [t10k-images-idx3-ubyte.gz](https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/t10k-images-idx3-ubyte.gz) | `bef4ecab320f06d8554ea6380940ec79` |
| Nhãn test | [t10k-labels-idx1-ubyte.gz](https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/t10k-labels-idx1-ubyte.gz) | `bb300cfdad3c16e7a12a480ee83cd310` |

Dữ liệu lưu tại `data/FashionMNIST_raw/`. Script kiểm tra MD5 trước khi sử dụng và giữ cả bản nén lẫn file IDX giải nén.
