# CNN pretrained — ImageNet-1K validation

## Tài liệu source

- [Các model — kiến trúc, pretrained weights và vai trò so sánh](docs/Models.md).
- [config.py — Cấu hình dữ liệu, model và timing](docs/src/config.md).
- [0_preparedata.py — Chuẩn bị ảnh và manifest](docs/src/0_preparedata.md).
- [1_benchmark.py — Inference, chỉ số và tài nguyên](docs/src/1_benchmark.md).

So sánh6 model có weights ImageNet1K hoàn chỉnh trên **10 lớp ×50 ảnh =500 ảnh**.
Không train/fine-tune; giữ nguyên classifier1.000 lớp. Dự đoán ra990 lớp khác vẫn tính sai.

## Vì sao thực hiện benchmark?

Phần trình bày kiến trúc CNN giải thích ý tưởng như tăng chiều sâu, Inception,
residual connection, tái sử dụng đặc trưng và thiết kế model nhẹ.
Benchmark bổ sung bằng chứng thực nghiệm: khi dùng các model đã train sẵn
để xử lý cùng ảnh trên cùng máy, những khác biệt đó thể hiện ra sao?

Mục tiêu là kiểm tra khả năng sử dụng trực tiếp pretrained model và đánh đổi
**độ chính xác – tốc độ – bộ nhớ**, phục vụ lựa chọn model cho nhu cầu triển khai.
Không cần huấn luyện lại vì câu hỏi đang xét là chất lượng và chi phí suy luận
của các weights ImageNet-1K có sẵn.

## Benchmark giúp trả lời điều gì?

- Model đoán đúng bao nhiêu trong500 ảnh? Xem số đúng/sai, Top-1 và Top-5.
- Model yếu ở lớp nào, nhầm sang lớp nào? Xem recall, confusion và cùng ID ảnh.
- Xử lý một ảnh có nhanh không? Xem latency batch1 và P95.
- Xử lý nhiều ảnh liên tiếp có hiệu quả không? Xem throughput batch16.
- Model cần bao nhiêu bộ nhớ GPU? Xem peak allocated VRAM ở cùng batch.
- Model ít tham số có đủ chính xác và nhanh hơn không? Đối chiếu MobileNetV2
  với các model còn lại thay vì suy luận chỉ từ kích thước kiến trúc.

Khi ưu tiên accuracy, tốc độ phản hồi hoặc bộ nhớ, lựa chọn có thể khác nhau;
không có một cột duy nhất xác định model tốt nhất cho mọi ứng dụng.
Các số đo phản ánh bộ ảnh, weights và phần cứng hiện tại, không chứng minh
một kiến trúc luôn vượt trội. Cùng bộ đánh giá giúp so sánh nhất quán,
nhưng khác training recipe vẫn là giới hạn khi giải thích nguyên nhân kết quả.

## Model và dữ liệu

VGG16, GoogLeNet, ResNet50, DenseNet121, EfficientNet-B0, MobileNetV2.
Tất cả chọn IMAGENET1K_V1 và transforms đi kèm từng weights.
Cùng V1 không có nghĩa cùng training recipe; so sánh model/weights hoàn chỉnh.

Chọn trước10 lớp: golden retriever, tabby, zebra, airliner, school bus,
mountain bike, coffee mug, laptop, strawberry, volcano.
Lấy đủ50 ảnh validation/lớp, không lấy ảnh train để bù, không chọn ảnh theo dự đoán.
Validation nằm ngoài train nhưng có thể được dùng để phát triển/lựa chọn model;
không bảo đảm tuyệt đối model chưa từng được kiểm tra trên dữ liệu này.

## Cấp quyền tải

1. Đăng nhập tài khoản tại [ILSVRC/imagenet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k).
2. Đọc/chấp nhận điều kiện truy cập của dataset; đăng nhập đơn thuần chưa đủ.
3. Đăng nhập công cụ local bằng `hf auth login`; dùng xác nhận qua trình duyệt
   nếu CLI cung cấp. Đăng nhập web không tự đăng nhập script Python.
   Nếu dùng token Read, nhập trong terminal, không gửi vào chat hoặc lưu source.

Từ gốc workspace, dùng .venv:

```powershell
.\.venv\Scripts\python.exe -m pip install datasets huggingface_hub
.\.venv\Scripts\hf.exe auth login
.\.venv\Scripts\python.exe test_model_present/src/0_preparedata.py
.\.venv\Scripts\python.exe test_model_present/src/1_benchmark.py --smoke
.\.venv\Scripts\python.exe test_model_present/src/1_benchmark.py
```

Script chuẩn bị chỉ đọc split validation, streaming và chỉ lưu ảnh10 lớp đã chọn.
Để tìm đủ50 ảnh/lớp vẫn có thể phải đọc phần lớn50.000 bản ghi và truyền nhiều dữ liệu;
streaming không đảm bảo chỉ tải đúng dung lượng500 ảnh. Không tải split train.
Ghi revision nguồn, thứ tự nhãn, SHA ảnh và manifest để đối chiếu.
Thiếu quyền/token hoặc mapping không khớp thì dừng; không thay bằng dataset khác.

## Cấu trúc gọn

| File/thư mục | Vai trò |
|---|---|
| src/config.py | Sáu model,10 lớp, số ảnh, batch và protocol timing |
| src/0_preparedata.py | Lấy500 ảnh validation và manifest |
| src/1_benchmark.py | Accuracy, latency, throughput và VRAM |
| Model_Benchmark.ipynb | Đọc bảng/biểu đồ và phân tích từng lớp |
| data/imagenet1k/ | Dataset mới; manifest.csv và dataset_metadata.json |
| data/torch_cache/ | Weights pretrained đã tải, dùng lại |
| outputs/lan_N/ | Report JSON/CSV, predictions, status và log mỗi model |
| outputs/smoke_N/ | Smoke thử nhanh; không dùng xếp hạng model |

Đã bỏ audit riêng, outline thuyết trình và kế hoạch trùng lặp.
README hướng dẫn, progress ghi trạng thái và AI_usage ghi ngắn việc hỗ trợ.
Smoke_1 cũ thuộc ImageNetV2, không phải số đo cho500 ảnh ImageNet1K mới.
Giữ dữ liệu/cache cũ local; manifest mới ở đường dẫn riêng để tránh trộn hai bộ.

## Protocol và đọc số đo

- Mỗi model chạy một subprocess; dùng cùng manifest, seed42, batch16, FP32/TF32 off.
- Top-1/Top-5 được tính trên đủ1.000 đầu ra, không cắt logits còn10 lớp.
- Latency batch1: forward trên tensor đã preprocess và nằm trên GPU,
  đồng bộ CUDA trước/sau;20 warm-up rồi3×100 request, có mean/median/P95.
- Throughput batch16 và peak allocated/reserved MiB đo riêng; không cộng hai loại VRAM.
- Timing không tính download/decode/transform/transfer; evaluation_seconds có các bước
  loader/transform/transfer/inference nhưng không phải benchmark end-to-end I/O lạnh.
- Smoke16 ảnh dùng2 warm-up và1×3 request, tách benchmark đầy đủ.
- Report giữ URL/SHA weights, transform, môi trường, profile và raw timings;
  không cần audit riêng để kiểm tra nguồn số đo.

Run All notebook chỉ đọc report, không train hoặc benchmark lại.
Chỉ10 lớp nên kết quả không đại diện toàn ImageNet1K; ghi accuracy cùng recall từng lớp.
Ảnh/lớp hữu hạn, một máy, recipes khác nhau; không kết luận chênh lệch nhỏ ổn định.
Dataset/ảnh/weights cache không commit Git.

Trạng thái: [progress](progress.md). API: [Torchvision](https://docs.pytorch.org/vision/main/models).

## Kết quả thực đo — lan_1, 2026-11-02

RTX3050 Laptop, FP32/TF32 off,500 ảnh/10 lớp; latency batch1, throughput/VRAM batch16.

| Model | Top-1 % | Top-5 % | ms/ảnh | ảnh/s | Peak allocated MiB |
|---|---:|---:|---:|---:|---:|
| EfficientNet-B0 | 76.2 | 95.2 | 8.53 | 651.8 | 199.0 |
| ResNet50 | 73.6 | 93.2 | 6.35 | 265.9 | 278.4 |
| DenseNet121 | 72.4 | 91.0 | 14.94 | 283.6 | 202.3 |
| MobileNetV2 | 71.6 | 91.2 | 5.96 | 842.3 | 192.1 |
| GoogLeNet | 71.2 | 91.0 | 6.83 | 738.3 | 152.9 |
| VGG16 | 69.4 | 91.4 | 10.73 | 135.8 | 939.7 |

EfficientNet-B0 có accuracy cao nhất trong mẫu này; MobileNetV2 nhanh nhất ở cả hai phép đo.
GoogLeNet có peak allocated batch16 thấp nhất; ít tham số không luôn đồng nghĩa ít activation memory.
Không khái quát xếp hạng này cho toàn ImageNet hoặc phần cứng khác.
Chi tiết: [CSV](outputs/lan_1/results.csv), [notebook](Model_Benchmark.ipynb).

## Dữ liệu train và số ảnh đúng/sai

Sáu weights pretrained được công bố cho ImageNet-1K; recipe và điều kiện train khác nhau,
không phải thử nghiệm sáu model được train lại bằng một pipeline kiểm soát chung.
Benchmark hiện dùng cùng500 ảnh và nhãn cho mọi model, preprocessing theo từng weights.
Bảng chính notebook ưu tiên Model/Tổng ảnh/Đúng/Sai/Accuracy/timing/VRAM.
Đúng/Sai là Top-1; top5_correct cho biết số ảnh có nhãn thật trong năm dự đoán đầu.
Số đếm tính trực tiếp từ predictions, không suy ngược phần trăm đã làm tròn.
Report lan_1 được bổ sung số đếm từ predictions; timing, accuracy và source_sha256
vẫn thuộc lần inference gốc, không phải một lần chạy benchmark mới sau sửa code.
