# Kế hoạch thực hiện — CNN và ViT trên CIFAR-100

Ngày cập nhật: 05/11/2026. Thời lượng dự kiến: 4 tuần, nhóm 6 thành viên.
Yêu cầu và nghiệm thu: [Requirements_Analysis.md](Requirements_Analysis.md).

## 1. Quyết định và trạng thái

- Đã thống nhất: CIFAR-100; bốn chiến lược CNN/ViT × freeze/fine-tune.
- Hướng model hiện tại: ResNet-50 baseline và ViT-B/16 nguyên bản từ torchvision.
- Bỏ phương án DeiT để tên kiến trúc bám sát yêu cầu ViT.
- Chưa smoke run; batch size, learning rate, số epoch và thời gian chạy chưa chốt.
- Không có kết quả train/test của đồ án này ở thời điểm lập kế hoạch.

## 2. Model và classifier

| Thành phần | CNN baseline | Model chính |
|---|---|---|
| Model | ResNet-50 | ViT-B/16 |
| Trọng số | ResNet50_Weights.IMAGENET1K_V1 | ViT_B_16_Weights.IMAGENET1K_V1 |
| Pretraining | ImageNet-1K | ImageNet-1K |
| Đặc trưng phân loại | Vector sau global average pooling | CLS token cuối encoder |
| Head CIFAR-100 | Linear(2048, 100) | Linear(768, 100) |
| Tham số với head 1.000 lớp gốc | Khoảng 25,6 triệu | Khoảng 86,6 triệu |

ViT-B/16: Base, 12 encoder block, hidden size 768, 12 attention head, patch 16 × 16.
Đầu vào 224 × 224 tạo 196 patch; thêm CLS thành 197 token.
Đếm và lưu lại total/trainable parameters sau thay head; không dùng số gốc làm số đo cuối.
Head chỉ gồm Linear để tránh thêm biến số về kiến trúc classifier.

Cặp hiện tại không tương đương số tham số, chưa đáp ứng phần mở rộng same parameter count.
Không suy diễn chênh lệch accuracy chỉ do CNN hay attention; phải xét quy mô và pretraining.

Nguồn: [ResNet-50](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html),
[ViT-B/16](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.vit_b_16.html).

## 3. Bốn chiến lược train

| Mã | Model | Backbone | Head |
|---|---|---|---|
| A | ResNet-50 | Freeze toàn bộ | Train head mới |
| B | ResNet-50 | Fine-tune toàn bộ | Train head mới |
| C | ViT-B/16 | Freeze toàn bộ | Train head mới |
| D | ViT-B/16 | Fine-tune toàn bộ | Train head mới |

Mỗi lần train khởi tạo độc lập từ đúng pretrained checkpoint, không chạy B tiếp từ A hoặc D tiếp từ C.
Trong freeze, backbone ở eval để giữ BatchNorm statistics và tắt dropout của backbone.
Trong fine-tune, backbone ở train; công bố chính sách BatchNorm và kiểm tra batch nhỏ trước khi chốt.
Validation/test luôn model.eval() và không tính gradient.

Các phép đối chiếu:
- A với C: khả năng chuyển giao của đặc trưng backbone khi chỉ học head.
- B với D: hiệu quả hai model sau fine-tune.
- A với B, C với D: lợi ích và chi phí khi cập nhật backbone.

## 4. Protocol và thông số cần chốt sau smoke run

- Split đề xuất 45k/5k/10k phân tầng, seed 42, lưu indices dùng chung.
- Input 224 × 224; dùng normalization tương ứng weights, xác minh hai bên trước khi khóa config.
- Train augmentation đơn giản, dùng chung; validation/test không augmentation ngẫu nhiên.
- Loss CrossEntropyLoss; model trả logits, không softmax trước loss.
- Đề xuất AdamW; learning rate freeze có thể cao hơn fine-tune, cần chạy thử để chọn.
- Weight decay và scheduler: ghi tường minh; chưa quyết định loại hoặc giá trị.
- Cùng ngân sách epoch chính thức; mỗi cấu hình có cơ hội tinh chỉnh tương đương.
- Không ép cùng learning rate nếu điều đó gây bất lợi cho một kiến trúc.
- Smoke bắt đầu batch nhỏ; thử mixed precision, đo khả năng fine-tune ViT trên GPU 4 GB.
- Nếu batch khác nhau, cân nhắc accumulation và ghi effective batch; accumulation không thay BatchNorm.
- Best theo validation accuracy, hòa thì validation loss thấp hơn; last là epoch hoàn tất gần nhất.
- Resume cần model, optimizer, scheduler/scaler nếu có, epoch và RNG; không chỉ model weights.

Mục tiêu ban đầu là 4 lần train chính thức, mỗi cấu hình một seed.
Smoke run không tính là kết quả chính thức. Chỉ thêm lượt khi có lỗi hoặc ngân sách cho phép và được thống nhất.
Nếu GPU không đáp ứng, ghi số đo và thống nhất tài nguyên/phương án trước khi thay model hoặc phạm vi.

## 5. Logging và đánh giá

- Đầu run: model/weights, kiến trúc, total/trainable parameters, config, split, seed và phiên bản thư viện.
- Phần cứng: GPU/VRAM, CPU/RAM, precision, batch size, optimizer và preprocessing.
- Mỗi epoch: train/val loss, accuracy, learning rate và thời gian; không ghi log từng batch.
- Lưu best.pt, last.pt, history và dự đoán có ID/nhãn/xác suất khi evaluate.
- Chỉ số chính: test top-1 accuracy và macro-F1; bổ sung top-5 và precision/recall theo lớp.
- Confusion matrix 100 lớp kèm các cặp nhầm nổi bật; ví dụ đúng/sai và độ tin cậy.
- ROC-AUC/AP đa lớp là tùy chọn sau phần cốt lõi, ghi rõ one-vs-rest và cách lấy trung bình.
- Ghi thời gian train, inference latency/throughput và peak VRAM cùng điều kiện đo.
- Đo inference cùng máy, precision, batch và warm-up; phân biệt forward-only với end-to-end.
- Đánh giá train/val/test bằng best checkpoint và preprocessing xác định khi so sánh generalization gap.

## 6. Lịch 4 tuần và phân công dự kiến

| Tuần | Công việc | Đầu ra kiểm tra được |
|---|---|---|
| 1 | Đề cương, data/EDA, model, smoke bốn cấu hình | Split cố định, model đúng, số đo tài nguyên |
| 2 | Khóa cấu hình, train bốn chiến lược | Log, history, best/last và validation |
| 3 | Test, so sánh, phân tích lỗi, demo | Bảng kết quả, hình phân tích, demo chạy được |
| 4 | Notebook, hoàn thiện báo cáo, kiểm thử và tập trình bày | Bộ nộp tái lập được, giới hạn rõ ràng |

Vai trò dự kiến, chưa gán tên: 1 tích hợp/model/train; 2 data/EDA; 3 kiểm thử/log/checkpoint;
4 đánh giá/phân tích; 5 demo/visualization; 6 báo cáo/tái lập và kiểm tra tài liệu.
Các thành viên phải giải thích được phần phụ trách; người tích hợp rà soát giao diện và kết quả chung.

## 7. Cấu trúc thư mục dự kiến

```text
coursework_vit_classification/
├── README.md                      # Giới thiệu, cài đặt và thứ tự chạy
├── Requirements_Analysis.md        # Yêu cầu, phạm vi và nghiệm thu
├── Implementation_Plan.md          # Thiết kế thí nghiệm và lịch thực hiện
├── progress.md                    # Trạng thái thực tế, kiểm tra và bước tiếp theo
├── AI_usage.md                    # Ghi nhận hỗ trợ AI và cách kiểm chứng
├── requirements.txt               # Thư viện cần cài
├── Coursework_Report.ipynb         # Báo cáo đọc kết quả, so sánh và phân tích
├── src/
│   ├── config.py                  # Dataset, model, chiến lược và tham số train
│   ├── data.py                    # Dataset, split, transforms và DataLoader dùng chung
│   ├── model.py                   # Nạp pretrained, thay head, freeze/fine-tune
│   ├── logger.py                  # Ghi config, phần cứng, log epoch và checkpoint
│   ├── 1_eda.py                   # Kiểm tra dữ liệu, tạo thống kê EDA
│   ├── 2_train.py                 # Một luồng train cho cả bốn cấu hình
│   ├── 3_evaluate.py              # Nạp checkpoint, tính metric và lưu dự đoán
│   └── 4_demo.py                  # Dự đoán ảnh bằng checkpoint được chọn
├── docs/                          # Mô tả workflow/code khi triển khai
├── visualization/
│   └── explorer.html             # Trang xem dữ liệu và ảnh dự đoán khi cần
├── data/                          # CIFAR-100 và cache pretrained; không commit
└── outputs/
    ├── eda.json                   # Thống kê EDA, không nhúng pixel ảnh
    ├── cifar100_split.json         # Chỉ số split dùng chung
    └── runs/
        └── lan_1/                 # Mỗi lần train tạo thư mục mới, không ghi đè
            ├── config.json        # Model, chiến lược, seed, split và môi trường
            ├── train.log          # Cấu hình đầu run và một dòng mỗi epoch
            ├── history.json       # Loss, accuracy, LR, thời gian từng epoch
            ├── best.pt            # Checkpoint tốt nhất theo validation
            ├── last.pt            # Trạng thái để resume từ epoch hoàn tất
            └── evaluation/        # Metric, dự đoán và số đo inference
```

Đã tạo bộ khung file nguồn; checkpoint, JSON metric và log chỉ sinh sau khi chạy thực tế.
Mã A–D nằm trong config của run; tên lan_1, lan_2... chỉ là thứ tự thực hiện, không cố định chiến lược.
Notebook đọc artifact; không tự train, tải dữ liệu hay chạy test khi Run All mặc định.
Các module dùng chung không có tác dụng phụ; không import script đánh số để tái sử dụng code.
Demo dùng chung model/preprocessing đã kiểm chứng; HTML không chứa pipeline train riêng.

## 8. Tiến độ và ghi nhận hỗ trợ AI

- 05/11/2026: tổng hợp yêu cầu được cung cấp và các quyết định thí nghiệm thành hai tài liệu.
- 05/11/2026: bổ sung cấu trúc thư mục theo các lab, vai trò script và vị trí artifact; chưa tạo file rỗng.
- Hỗ trợ AI: tổ chức yêu cầu, bảng chiến lược, protocol, rủi ro và checklist; chưa sinh mã huấn luyện.
- Kiểm tra tài liệu: thực hiện kiểm tra UTF-8, số dòng và liên kết nội bộ sau khi tạo.
- Bước tiếp theo: triển khai config/data/EDA, model rồi kiểm tra tài nguyên bằng smoke run.
- Đã tạo bộ khung; trạng thái thực tế và kiểm tra tiếp theo ghi tại [progress.md](progress.md).
