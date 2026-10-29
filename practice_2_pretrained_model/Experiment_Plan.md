# Practice 2 — Kế hoạch 6 lượt train

## 1. Phạm vi

Phân loại CIFAR-10 bằng VGG16 và DenseNet121 pretrained ImageNet-1K.

Đáp ứng checklist VGG/DenseNet, transfer learning, fine-tuning và thay đổi hyperparameter với thời gian vừa phải.

Cùng split, preprocessing, seed, optimizer và ngân sách 20 epoch; đổi từng yếu tố.

Thay kế hoạch cũ 3 model × 6 cấu hình bằng **2 model × 3 cấu hình = 6 lượt**.
Không đặt accuracy bắt buộc. Không thêm ensemble/TTA/HPO/scheduler vào phạm vi này.
model.py chỉ triển khai VGG16/DenseNet121 với head Linear 10 lớp.

## 2. Lý do chọn model và kiến trúc

| Model | Đặc điểm | Backbone/head dùng trong lab | Fine-tune mở phần nào? |
|---|---|---|---|
| VGG16 | Các block convolution 3×3 tuần tự, dễ giải thích | features → GAP 1×1 → flatten → Linear(512,10) | Ba conv cuối: features.24/26/28 |
| DenseNet121 | Nối feature các lớp trước, tái sử dụng đặc trưng | features → ReLU → GAP 1×1 → flatten → Linear(1024,10) | denseblock4 và norm5 |

- Pretrained: torchvision ImageNet-1K (1.000 lớp), IMAGENET1K_V1 cho cả hai model.
- Thay head 1.000 lớp bằng H1: Linear(D,10). Không thử H2 trong kế hoạch mới.
- Freeze: backbone không học, chỉ head nhận gradient.
- Fine-tune: mở nhóm cuối cùng head; phần trước vẫn freeze.
- Giữ running statistics BatchNorm; affine chỉ học nếu thuộc nhóm mở.
- Không đổi tên state_dict hoặc kiến trúc VGG16 A đã train; giữ khả năng resume.

## 3. Dữ liệu và EDA

- CIFAR-10: ảnh RGB 32×32, 10 lớp; 45.000 train / 5.000 validation / 10.000 test.
- Một split phân tầng seed 42 cho cả sáu lượt; kiểm tra coverage, nhãn và ID không giao nhau.
- EDA: ảnh mẫu từng lớp, phân bố lớp, shape/dtype/range, RGB mean/std/histogram, ảnh trùng và preprocessing.
- Thống kê RGB tính từ train; không fit preprocessing hoặc chọn cấu hình bằng test.
- Train: resize bilinear antialias 224×224 → random horizontal flip → tensor → normalize ImageNet.
- Validation/test: resize → tensor → normalize; không augmentation ngẫu nhiên.
- ImageNet mean (0.485,0.456,0.406), std (0.229,0.224,0.225); resize không tạo thêm chi tiết.
- HTML giữ bộ duyệt đầy đủ dữ liệu; notebook dùng EDA report và ảnh mẫu để trình bày.

## 4. Ma trận thí nghiệm mới

Mỗi hàng chạy đủ **VGG16 và DenseNet121**; tổng sáu lượt, tối đa 120 model-epoch.

| Mã | Backbone | Head | LR | Batch | Câu hỏi đối chiếu |
|---|---|---|---:|---:|---|
| A | Freeze | H1 | 0.001 | 64 | Baseline transfer learning |
| B | Mở nhóm cuối | H1 | 0.001 | 64 | A/B: fine-tune có cải thiện không? |
| C | Mở nhóm cuối | H1 | 0.0001 | 64 | B/C: LR thấp hơn ảnh hưởng thế nào? |

- CrossEntropyLoss, Adam, weight decay 0, 20 epoch/lượt; không scheduler/early stopping.
- Mỗi model bắt đầu từ pretrained và seed 42 cho từng cấu hình, không nạp A để tạo B/C.
- Resume chỉ tiếp tục đúng run bị ngắt, cùng model/cấu hình/split và ngân sách ban đầu.
- Cùng LR phục vụ đối chứng; không khẳng định đó là LR tối ưu cho mọi model.
- Nhóm cuối có số tham số khác nhau; không quy toàn bộ chênh lệch cho riêng kiểu kết nối.
- Thứ tự: hoàn tất A của hai model, rồi B, rồi C; không chạy đồng thời trên GPU 4 GB.
- Mã B/C mới khác B/C cũ. Dùng model_config thực trong checkpoint để phân loại kết quả lịch sử.

## 5. Log, checkpoint và trình bày kết quả

- Ghi một lần môi trường, kiến trúc, pretrained identifier, split hash, transforms và optimizer thực tế.
- Mỗi epoch ghi train/val loss, accuracy, LR, thời gian; TensorBoard cùng metrics, không theo dõi batch.
- best.pt: val accuracy cao nhất; hòa chọn val loss thấp hơn; vẫn hòa giữ epoch sớm hơn.
- last.pt: epoch hoàn tất gần nhất, model/Adam/random/DataLoader state để resume.
- Dừng giữa epoch thì phần epoch chưa hoàn tất không được tính; chạy lại epoch đó khi resume.
- Notebook đọc artifact thật, gắn model/cấu hình/split/epoch/nguồn; thiếu thì ghi chưa có, không tạo số minh họa.
- Lượt gián đoạn chưa được tính là thí nghiệm hoàn tất và không so như lượt đủ 20 epoch.

## 6. Đánh giá và so sánh

- Dùng validation chọn một cấu hình/checkpoint cho mỗi model; khóa hai checkpoint trước test.
- Chỉ sau xác nhận mới bật RUN_TEST; test không dùng chọn LR, epoch hoặc fine-tune policy.
- Metrics: loss, accuracy, macro-F1, precision/recall/F1 theo lớp và confusion matrix.
- Sau test: TP/TN/FP/FN, TPR/recall, FPR, specificity; ROC OvR từng lớp và macro/micro AUC. TPR/FPR argmax tách biệt với ROC quét ngưỡng.
- Lưu xác suất đủ 10 lớp và thứ tự cột trong test report để notebook phân tích; so cùng split/ID/nhãn, không chọn ngưỡng hoặc cấu hình bằng test.
- Phân tích ảnh sai theo ID thật, confidence và cặp lớp nhầm; so cùng ID/split.
- Bổ sung PR/AP OvR theo lớp, macro/micro AP và đối chiếu bốn nhóm đúng/sai giữa hai checkpoint; không tự chọn cặp bằng điểm test.
- Báo cáo thời gian, trainable parameters, GPU nếu đo được; một seed không chứng minh ưu thế thống kê.
- Ghi kết quả kèm model/cấu hình, split/device, epoch và artifact nguồn.

## 7. Đối chiếu checklist

| Yêu cầu | Bằng chứng |
|---|---|
| Dataset/EDA | CIFAR-10, split cố định, EDA report và notebook |
| VGG/DenseNet/pretrained | Hai model, ImageNet-1K V1, kiến trúc/tham số thực |
| Transfer learning | A: thay head H1, freeze backbone |
| Fine-tuning | B/C: mở nhóm cuối |
| Hyperparameters | B/C chỉ thay learning rate; batch giữ cố định |
| Train/TensorBoard | Sáu lượt đủ epoch, event logs, history và best/last |
| Evaluation | Hai checkpoint khóa bằng validation rồi mới test |
| Comparison | Bảng sáu lượt, đường cong, per-class/confusion/ảnh sai và giới hạn |

## 8. Flow thực hiện

1. Mở [Practice_2_Report.ipynb](Practice_2_Report.ipynb), chọn .venv, Run All để đọc EDA/artifact.
2. Kiểm tra notebook và cấu hình; train hiện đang tạm dừng.
3. Train mới VGG16 A từ pretrained, RESUME_RUN=None; tạo checkpoint từ lượt mới.
4. Chạy các lượt còn lại theo ma trận mới, cập nhật notebook từ artifact.
5. Review/khóa checkpoint, evaluate test rồi hoàn thiện nhận xét.

Chi tiết vận hành: [Training.md](Training.md). Trạng thái hiện tại: [progress.md](progress.md).
Nhật ký AI: [Step](AI_Usage_Step.md), [Full](docs/AI_Usage_Full.md).
