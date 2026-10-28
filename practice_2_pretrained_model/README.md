# Practice 2 — Pretrained CNN trên CIFAR-10

## Phạm vi đã thống nhất

- Đọc [yêu cầu bài](Practice_2_Pretrained_CNN_Requirements.md), rồi [kế hoạch thí nghiệm](Experiment_Plan.md).
- VGG16, ResNet18, DenseNet121; 6 cấu hình chung/model, tổng 18 lượt × 20 epoch.
- Dùng `.py` làm luồng chính; chưa cần notebook hoặc hai bản triển khai song song.
- Đã triển khai đủ 7 file Python và chạy EDA thật; train/evaluate mới được smoke test.
- [progress.md](progress.md) ghi tiến độ/bàn giao; [Training.md](Training.md) ghi cách train/evaluate và giới hạn kiểm tra.

## Tổ chức source dự kiến — 7 file

Cấu trúc thư mục đã tạo:

```text
practice_2_pretrained_model/
├── src/             # Các file Python khi triển khai từng bước
├── data/            # CIFAR-10 tải về; được Git ignore
├── outputs/         # Split, thống kê, log, TensorBoard và checkpoint
├── visualization/   # Hình EDA và tài sản trực quan
├── README.md
├── Experiment_Plan.md
├── Practice_2_Pretrained_CNN_Requirements.md
└── AI_Usage_Step.md  # Theo giai đoạn, tổng hợp vào docs/AI_Usage_Full.md
```

Đã tạo và triển khai đủ 7 file Python; kết quả kiểm tra được ghi trong tài liệu workflow.
Không tạo `.gitkeep`; thư mục theo run sẽ tạo khi chạy.

| File trong `src/` | Trách nhiệm |
|---|---|
| `config.py` | Split, seed, preprocessing, model đang chọn và cấu hình A–F |
| `data.py` | Tải CIFAR-10, tạo/lưu split, transforms và DataLoader dùng chung |
| `model.py` | Ba model khai báo rõ ràng, thay head và freeze/unfreeze |
| `logger.py` | Thu thập cấu hình máy, định dạng log, TensorBoard và đóng/flush log |
| `1_eda.py` | Chuẩn bị/kiểm tra dữ liệu, thống kê và hình khảo sát trước train |
| `2_train.py` | Train/validation trực tiếp, metrics mỗi epoch, best checkpoint |
| `3_evaluate.py` | Đọc checkpoint, đánh giá, phân tích lỗi và so sánh các lượt |

- Train/evaluate gọi các module dùng chung; không import script đánh số lẫn nhau.
- Module dùng chung không tự tải dữ liệu hoặc chạy tác vụ khi import.
- Một lần chạy train chọn một model và một cấu hình; không viết 18 script riêng.
- Dùng lời gọi PyTorch trực tiếp và vòng lặp epoch/batch hiện ngay trong file train.
- Tách các khối setup, forward/loss, backward/step, validation và lưu kết quả bằng dòng trống.
- Evaluate viết tuần tự: load checkpoint → `eval()` → `no_grad()` → tổng hợp metrics.
- Logging không điều khiển train; tránh Trainer, callback, registry hoặc chuỗi hàm nhiều tầng.
- Chỉ tách helper khi được dùng lại hoặc làm rõ logic phức tạp; tránh wrapper một lời gọi.
- Ảnh code giảng viên chỉ định hướng phong cách, không khóa optimizer, số epoch hay transforms.
- Dùng cấu hình và API phù hợp bài thực tế; TensorBoard phải ghi metrics đo được.

## EDA — phần bắt buộc trước train

- Kiểm tra số ảnh, shape RGB 32×32, dtype, nhãn, khoảng pixel và phân bố lớp.
- Lưu split phân tầng cố định; kiểm tra ID không giao nhau và thống kê từng lớp/split.
- Kiểm tra ảnh trùng chính xác, kể cả qua các split, tách biệt với kiểm tra ID trùng.
- Báo cáo trùng ảnh nếu có; không tự xóa/repartition dữ liệu hoặc thay test chính thức.
- Hiển thị ảnh mẫu từng lớp, đa dạng nội lớp và các cặp lớp có thể khó phân biệt.
- Tính thống kê RGB từ train; không dùng validation/test để fit preprocessing.
- Minh họa ảnh gốc → resize 224×224 → augmentation → normalization.
- Khi hiển thị ảnh normalized, đảo normalization hoặc ghi rõ thang giá trị.
- Ghi rõ resize không tạo thêm chi tiết ảnh; không suy diễn nhãn sai chỉ từ quan sát.
- EDA phục vụ quyết định dùng train/validation; không dùng quan sát test để tuning.
- Lưu thống kê/split trong `outputs/`; hình EDA trong `visualization/`.
- Chỉ tạo một HTML xem EDA nếu cần; không tạo hai trang template và generated trùng chức năng.

## Logging đầy đủ, code train/evaluate vẫn gọn

**Đầu mỗi lượt:** ghi cấu hình thực tế một lần, không chỉ tên cấu hình dự kiến.

- Máy: OS, CPU, RAM, GPU/VRAM, device thực dùng; Python, PyTorch, torchvision,
  CUDA/cuDNN; thông tin không lấy được ghi rõ unavailable, không suy đoán.
- Dữ liệu: nguồn, split/định danh chỉ số, seed, transforms, batch, shuffle, worker, pin memory.
- Model: pretrained weights identifier, kiến trúc đầy đủ sau thay head, feature shape,
  tổng/trainable/frozen parameters, tên các nhóm mở và chính sách BatchNorm.
- Train: loss/options, optimizer/parameter groups, LR, weight decay, epoch, scheduler,
  chế độ số học, thiết lập seed/determinism và đường dẫn output.

**Mỗi epoch:** một dòng console và file: epoch, train/val loss, accuracy, LR,
thời gian epoch, best epoch. TensorBoard ghi cùng metrics theo epoch; không log từng batch.

**Kết thúc:** trạng thái hoàn tất/lỗi/gián đoạn, số epoch thực tế, tổng thời gian,
best checkpoint/metrics, số optimizer step và GPU memory peak nếu đo được.
Flush log trong khi chạy; lỗi/gián đoạn phải giữ artifact đã ghi và không báo Finished giả.

**Evaluate:** ghi checkpoint, model/config đã lưu, split được đánh giá, device và metrics;
không tự lấy config hiện tại thay cấu hình checkpoint, không tự chạy test trước khi khóa model.

## Trình tự triển khai và review

1. Chuẩn bị dữ liệu + EDA; ghi nguồn dataset, URL tải và kiểm tra toàn vẹn thực tế.
2. Sau review EDA: kiểm tra ba model/head, forward shape, gradient và freeze/BatchNorm.
3. Hoàn thiện logging, train/validation và checkpoint; smoke test ngắn trước train chính thức.
4. Chạy các hàng A–F theo kế hoạch, đủ ba model mỗi hàng; không tự mở rộng ngân sách.
5. So sánh validation, review/khóa checkpoint rồi mới test và hoàn thiện phân tích lỗi.

Mỗi phần cập nhật tài liệu/progress và bằng chứng kiểm tra; chờ phản hồi trước phần tiếp theo.
## Phần dữ liệu đã triển khai

- `config.py`: seed 42, validation 5.000, batch 64, worker 0 sau lỗi shared memory Windows; `.venv` dùng CUDA.
- `load_datasets(download=True)`: tải train/test bằng torchvision, giữ ảnh gốc cho EDA.
- `load_split(targets)`: lưu `outputs/cifar10_split.json`, kiểm tra coverage, class ratio
  và cấu hình khi dùng lại; không tự ghi đè split cũ không tương thích.
- `build_loaders(download=False)`: train/val/test; train có flip, eval không ngẫu nhiên,
  resize bilinear antialias 224×224 và chuẩn hóa ImageNet. Test loader chưa đánh giá model.
- Import module không tải dữ liệu; gọi download rõ ràng từ bước chuẩn bị dữ liệu/EDA.
- Nguồn: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html).
- URL torchvision dùng: `https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz`;
  MD5 archive: `c58f30108f718f92721af3b95e74349a`. torchvision kiểm tra checksum khi tải.
- `.venv`: torch 2.11.0+cu126, torchvision 0.26.0+cu126; GPU + worker 2 đã qua smoke test pretrained A.
- Dataset đã tải, kiểm tra toàn vẹn và giải nén; EDA thật và cả ba loader đã kiểm tra.
  Model đã kiểm tra cấu trúc/gradient bằng weights ngẫu nhiên, chưa xác minh pretrained weights.

## Xem EDA trên HTML

Giao diện theo mẫu Practice 1: header xanh đậm, teal, menu ngang, thẻ thống kê và khung ảnh cuộn.

1. Đợi CIFAR-10 tải và kiểm tra checksum xong.
2. Từ gốc workspace chạy `python practice_2_pretrained_model/src/1_eda.py`.
3. Mở `visualization/eda.html` qua Live Server để đọc `outputs/eda.json` tự động.
   Hoặc mở HTML trực tiếp, chọn `outputs/eda.json` và 6 file `.rgb.gz` trong `data/CIFAR10_web/`.

- Một trang [eda.html](visualization/eda.html) chứa bộ lọc split/lớp, toàn bộ ảnh có ID,
  bảng phân bố lớp, histogram RGB, preprocessing và nhóm ảnh trùng SHA-256.
- Gallery đủ 60.000 ảnh: lọc từng/tất cả split, lớp, tìm ID hoặc khoảng ID (100-200),
  sắp xếp, 50/100/200 ảnh mỗi trang, nhảy trang và bấm ảnh để xem lớn pixel/mượt.
- `1_eda.py` xuất 6 file `data/CIFAR10_web/*.rgb.gz`; HTML giải nén RGB và vẽ canvas như Practice 1.
- Toàn bộ data (kể cả RGB cho viewer) không đưa lên Git; tải CIFAR-10 rồi chạy 1_eda.py để tạo lại trên máy.
- JSON lưu ID/nhãn và SHA-256 từng chunk; train/validation dùng chung nguồn train, không nhân đôi pixel.
- Stats RGB chỉ tính train; test chỉ kiểm tra cấu trúc/ảnh trùng, không dùng tuning.
- Đã kiểm tra dữ liệu giả và dữ liệu thật; Edge headless qua HTTP đọc JSON thật và render
  bảng, gallery, histogram thành công. Chưa tự động hóa thao tác Live Server IDE.

## Kết quả EDA thật

- Train/validation/test: 45.000/5.000/10.000; mỗi lớp tương ứng 4.500/500/1.000 ảnh.
- Split phủ đủ train chính thức, không giao ID; 60.000 ảnh RGB/nhãn khớp dữ liệu gốc.
- Không có nhóm trùng pixel RGB chính xác trong 60.000 ảnh; chưa khảo sát near-duplicate.
- RGB train mean: 0,491137 / 0,482111 / 0,446449; std: 0,246869 / 0,243430 / 0,261603.
- Đây là thống kê mô tả; preprocessing vẫn dùng ImageNet theo kế hoạch pretrained.
- Kiểm tra độc lập mean/std từ histogram khớp; batch cả ba split là 64×3×224×224, hữu hạn.
- ResNet18 A lan_1 đủ 20 epoch, best epoch 18 val 88,54%; VGG16 A đang train GPU tại lan_2, worker 0.
- Đã đối chiếu 6 chunk RGB đủ 60.000 ảnh với dataset gốc, kiểm tra canvas/filter,
  ID/khoảng ID, trang cuối, preview trong Edge. Đã xem screenshot desktop/mobile.

## Model đã triển khai

- `PretrainedCNN(name, head='H1', fine_tune=False, pretrained=True)` trong `model.py`.
- [Model_QA.md](Model_QA.md) lưu các câu hỏi/giải thích về model và thông số phát sinh khi làm lab.
- Chọn `MODEL_NAME` và `EXPERIMENT` A–F trong config; chưa tự chạy train khi import.
- 12 tổ hợp head/freeze qua forward/backward/update/BatchNorm với input giả 1×3×224×224.
