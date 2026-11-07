# config.py — Cấu hình benchmark

Source: [config.py](../../src/config.py).
Workflow tiếp theo: [Chuẩn bị dữ liệu](0_preparedata.md) → [Benchmark](1_benchmark.md).

## Mục đích

Tập trung các đường dẫn, danh sách model, bộ ảnh và protocol đo.
Import file này chỉ khai báo cấu hình; không tải dataset/weights hoặc chạy inference.
Benchmark không có optimizer, loss huấn luyện, backward hoặc cập nhật trọng số.

## Đường dẫn

| Biến | Vai trò |
|---|---|
| ROOT | Thư mục test_model_present, suy ra từ vị trí file |
| MANIFEST | data/imagenet1k/manifest.csv, danh sách ảnh và nhãn |
| OUTPUT | outputs/, chứa các lượt lan_N và smoke_N |

Cache weights nằm dưới data/torch_cache/, do script benchmark thiết lập.
Dataset và cache giữ local, không commit Git.

## Model và pretrained weights

Giới thiệu kiến trúc và vai trò từng model: [Các model trong benchmark](../Models.md).

MODELS khai báo VGG16, GoogLeNet, ResNet50, DenseNet121, EfficientNet-B0, MobileNetV2.
Mỗi model chọn rõ enum IMAGENET1K_V1, không chọn DEFAULT có thể đổi theo phiên bản.
Các weights đã huấn luyện sẵn trên ImageNet-1K; bài này chỉ đánh giá inference.
Giữ nguyên classifier 1.000 lớp, không thay head thành 10 lớp.
Cùng V1 không đồng nghĩa cùng recipe hoặc ngân sách huấn luyện.

## Dữ liệu đánh giá

| Biến | Giá trị | Ý nghĩa |
|---|---:|---|
| DATASET_NAME | ImageNet-1K | Tên dữ liệu ghi trong report |
| HF_DATASET | ILSVRC/imagenet-1k | Nguồn Hugging Face có điều kiện truy cập |
| SPLIT | validation | Không dùng ảnh train |
| SELECTED_CLASSES | 10 lớp | Mapping index ImageNet gốc sang tên lớp |
| IMAGES_PER_CLASS | 50 | Lấy đủ ảnh validation mỗi lớp được chọn |
| MAX_IMAGES | 500 | Giới hạn số ảnh cho benchmark chính |
| SEED | 42 | Xáo manifest và thứ tự chạy model có thể tái lập |

10 lớp: golden retriever, tabby, zebra, airliner, school bus, mountain bike,
coffee mug, laptop, strawberry và volcano. Labels vẫn là index 0–999.
Ảnh thuộc một trong 10 lớp nhưng dự đoán sang 990 lớp còn lại vẫn tính sai.

## Protocol timing

| Biến | Giá trị | Ý nghĩa |
|---|---:|---|
| BATCH_SIZE | 16 | Batch accuracy và phép đo throughput |
| WARMUP | 20 | Forward khởi động trước khi đo mỗi batch |
| REPEATS | 3 | Số nhóm đo |
| TIMING_REQUESTS | 100 | Số request mỗi nhóm; tổng 300 request/phép đo |

Latency dùng batch 1 riêng, không lấy thời gian batch 16 chia cho 16.
--smoke giữ batch 16 nhưng đổi warm-up/repeats/requests thành 2/1/3 và chỉ lấy 16 ảnh.
FP32/TF32 off, device và num_workers=0 được thiết lập trong 1_benchmark.py.

## Lưu ý khi tinh chỉnh

0_preparedata.py hiện kiểm tra cứng tổng 500 ảnh và 50 ảnh/lớp.
Không chỉ đổi SELECTED_CLASSES hoặc IMAGES_PER_CLASS rồi cho rằng mọi bước tự thích nghi.
Giảm MAX_IMAGES có thể tạo mẫu không còn cân bằng, cần ghi rõ giới hạn trước khi so sánh.
Không thay protocol giữa các model trong cùng một lượt benchmark.
