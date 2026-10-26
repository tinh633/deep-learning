# Practice 2 — Kế hoạch thí nghiệm Pretrained CNN trên CIFAR-10

## 1. Mục tiêu và phạm vi

- Bám [yêu cầu và checklist Practice 2](Practice_2_Pretrained_CNN_Requirements.md).
- Dùng duy nhất CIFAR-10, bài toán phân loại ảnh RGB thành 10 lớp.
- So sánh VGG16, ResNet18 và DenseNet121 với trọng số pretrained ImageNet.
- Khảo sát thay classification head, freeze/unfreeze, learning rate và batch size.
- Theo dõi bằng TensorBoard; đánh giá và phân tích kết quả theo cùng quy trình.
- Kế hoạch ban đầu: **18 lượt train chính thức = 3 model × 6 cấu hình**.
- Kế hoạch đã được chấp nhận; data/EDA/model đã triển khai, chưa có kết quả train.
- [README](README.md) ghi tổ chức 7 file Python, EDA, logging và trình tự triển khai.

### Lý do chọn ba model

| Model | Điểm nổi bật | Vai trò trong lab |
|---|---|---|
| VGG16 | Các lớp convolution 3×3 xếp tuần tự theo block | Đối chứng kiến trúc tuần tự, dễ quan sát backbone/head |
| ResNet18 | Residual connection cộng đầu vào với đầu ra nhánh biến đổi | Khảo sát đặc trưng pretrained của kiến trúc residual |
| DenseNet121 | Nối các feature từ lớp trước trong cùng dense block | Khảo sát tái sử dụng feature qua dense connections |

Ba model đại diện ba cách kết nối khác nhau, đủ để so sánh có trọng tâm trong 18 lượt;
VGG và DenseNet cũng bám các ví dụ trong đề. Không mặc định model nào chính xác hơn.
Kết quả còn phụ thuộc pretraining, độ sâu, số tham số và head; không quy mọi chênh lệch
cho riêng kiểu kết nối. VGG16 đã thay head nên không so nguyên bản ba model ImageNet.

## 2. Dữ liệu và điều kiện chung

- Chia 50.000 ảnh train chính thức thành 45.000 train và 5.000 validation.
- Chia phân tầng theo lớp, seed 42; lưu chỉ số và dùng đúng một split cho mọi lượt.
- Giữ riêng 10.000 ảnh test chính thức; không dùng để chọn cấu hình hoặc epoch.
- Kiểm tra số lượng, phân bố lớp, nhãn và tính không giao nhau giữa các split.
- Ảnh đầu vào thống nhất 224×224; dùng cùng nội suy và chuẩn hóa ImageNet.
- Train: resize, random horizontal flip, chuyển tensor và chuẩn hóa.
- Validation/test: resize, chuyển tensor và chuẩn hóa; không augmentation ngẫu nhiên.
- Không dùng augmentation riêng cho từng model; ghi rõ ảnh gốc chỉ có 32×32.
- Mọi lượt dùng cùng seed 42 và cùng quy tắc seed cho DataLoader/augmentation.
- Cố định phiên bản thư viện, thiết bị và chế độ số học; lưu cấu hình thực tế.

## 3. Khám phá kiến trúc và thay head

Trước khi train, in kiến trúc gốc, xác định feature extractor và classification head;
ghi tổng tham số, tham số được train và kích thước feature đầu vào head.

| Model | Feature extractor giữ lại | Head gốc được thay | Nhóm cuối mở khi fine-tune |
|---|---|---|---|
| VGG16 | `features` | Toàn bộ `classifier` | Các conv của block 5 |
| ResNet18 | Backbone trước `fc` | `fc` | `layer4` |
| DenseNet121 | `features` | `classifier` | `denseblock4` và `norm5` |

- Chuẩn hóa đầu ra backbone bằng adaptive average pooling 1×1 rồi flatten.
- Head H1: `Linear(D, 10)`, với D là số kênh feature của từng backbone.
- Head H2: `Linear(D, 256) → ReLU → Dropout(0.2) → Linear(256, 10)`.
- Giữ nguyên cấu trúc và trọng số pretrained bên trong feature extractor.

Giải thích lựa chọn và giới hạn so sánh H1/H2 tại [Model_QA.md](Model_QA.md).

## 4. Freeze và unfreeze

- **Freeze:** mọi tham số backbone không nhận gradient; chỉ train head mới.
- Đặt backbone ở `eval()` khi train để giữ hành vi Dropout/BatchNorm ổn định.
- **Fine-tune:** chỉ mở nhóm cuối trong bảng; các nhóm trước vẫn đóng băng.
- Khi fine-tune, BatchNorm giữ running statistics ở chế độ eval; affine parameters
  chỉ được train nếu thuộc nhóm đã mở. Head luôn ở chế độ train trong bước train.
- Các nhóm cuối có quy mô khác nhau: đây là so sánh theo cùng chính sách mở block,
  không phải so sánh với số tham số được train bằng nhau.

## 5. Ma trận 18 lượt train

Mỗi hàng bên dưới phải chạy đủ **VGG16, ResNet18 và DenseNet121**.

| Mã | Backbone | Head | LR | Batch | Mục đích đối chiếu |
|---|---|---|---|---|---|
| A | Freeze | H1 | 0.001 | 64 | Baseline chung |
| B | Freeze | H1 | 0.0001 | 64 | B so với A: chỉ đổi LR |
| C | Freeze | H1 | 0.001 | 32 | C so với A: chỉ đổi batch |
| D | Freeze | H2 | 0.001 | 64 | D so với A: chỉ đổi head |
| E | Mở nhóm cuối | H1 | 0.001 | 64 | E so với A: chỉ đổi freeze policy |
| F | Mở nhóm cuối | H1 | 0.0001 | 64 | F so với B: đổi freeze; F/E: đổi LR |

- Cả 18 lượt: CrossEntropyLoss, Adam, weight decay 0, không scheduler.
- Cố định 20 epoch/lượt, không early stopping để cùng ngân sách epoch.
- LR trong bảng áp dụng cho toàn bộ tham số đang được train, gồm cả head.
- Mỗi lượt khởi tạo lại từ cùng pretrained weights và seed; không tiếp tục từ A.
- Không đổi cấu hình riêng cho một model giữa chừng hoặc ghi đè lượt thất bại.
- Nếu OOM/lỗi số học, ghi trạng thái thất bại; review điều chỉnh chung trước khi chạy lại.
- Batch 32 tạo nhiều optimizer step hơn batch 64; ghi số step và thời gian thực tế.
- Ma trận có kiểm soát từng thay đổi, chưa khảo sát đầy đủ mọi tương tác tham số.

## 6. TensorBoard, lưu kết quả và đánh giá

- Mỗi lượt có thư mục riêng `outputs/runs/lan_N/`; không ghi đè lượt cũ.
- Lưu model/weights identifier, split, seed, preprocessing, head, freeze policy,
  optimizer, LR, batch, epoch, tham số train, phiên bản thư viện và thiết bị.
- TensorBoard ghi theo epoch: train/val loss, accuracy, LR và thời gian epoch.
- Console/file log ghi một dòng mỗi epoch, không log từng batch hoặc mở rộng ngân sách chưa duyệt.
- Lưu best checkpoint theo validation accuracy; hòa thì chọn validation loss thấp hơn,
  nếu vẫn hòa giữ epoch sớm hơn. Lưu epoch và metrics tương ứng.
- Báo cáo validation accuracy/loss, best epoch, thời gian train, tham số train và
  bộ nhớ GPU đỉnh nếu đo được trên cùng thiết bị.
- Chọn cấu hình tốt nhất của mỗi model bằng validation và khóa cả ba trước khi test.
- Đánh giá test một lần cho mỗi checkpoint đã khóa: loss, accuracy, macro-F1,
  precision/recall/F1 từng lớp, confusion matrix và ví dụ dự đoán sai.
- Không dùng kết quả test để quay lại tuning; chưa công bố model cuối khi chưa review.
- Một seed chỉ cho kết quả trong thiết lập này, chưa chứng minh ưu thế thống kê.
- Cùng LR là đối chứng có kiểm soát, không đảm bảo LR tối ưu cho mọi kiến trúc.

## 7. Đối chiếu checklist và tiêu chí hoàn thành

| Checklist gốc | Kế hoạch đáp ứng / bằng chứng cần có |
|---|---|
| 1. Dataset | CIFAR-10; split cố định, thống kê và kiểm tra dữ liệu |
| 2. Models | VGG16, ResNet18, DenseNet121; bảng khám phá kiến trúc |
| 3. Load pretrained | Trọng số ImageNet, lưu định danh weights thực tế |
| 4. Transfer learning | A–D: backbone freeze, head mới H1/H2 |
| 5. Fine-tuning | E–F: mở nhóm cuối, kiểm tra gradient/tham số cập nhật |
| 6. Hyperparameters | A/B và E/F đổi LR; A/C đổi batch, đồng bộ cả ba model |
| 7. Training | Đủ 18 lượt hoàn thành; lỗi/gián đoạn phải được ghi rõ |
| 8. TensorBoard | Có event logs và đường cong train/validation của từng lượt |
| 9. Evaluation | Ba checkpoint khóa bằng validation rồi mới đánh giá test |
| 10. Comparison | Bảng 18 lượt, đối chiếu từng yếu tố và phân tích lỗi |

Không đặt accuracy bắt buộc vì đề bài không quy định. Hoàn thành khi có đủ bằng chứng
trong bảng, báo cáo đúng giới hạn thí nghiệm và không trộn validation với test.

## 8. Tiến độ và bước tiếp theo

- Data/EDA thật đã kiểm tra; `model.py` có PretrainedCNN, `config.py` đủ A–F, mặc định ResNet18/A.
- 12 tổ hợp weights ngẫu nhiên qua forward/gradient/update; frozen weights/BN không đổi, eval đúng.
- H1 tổng tham số VGG16/ResNet18/DenseNet121: 14.719.818 / 11.181.642 / 6.964.106;
  fine-tune trainable tương ứng: 7.084.554 / 8.398.858 / 2.170.378.
- Train/evaluate đã smoke test riêng; các kiểm tra tạm không thuộc 18 lượt chính thức.
- Logger/train đã triển khai, smoke test tạm A/E trên CIFAR-10 nhỏ đã qua; xem [Training.md](Training.md).
- ResNet18 pretrained A đã qua GPU smoke trong `.venv` với worker 2; chưa có lượt GPU chính thức hoàn tất.
- Làm từng phần có review; ghi nhận đóng góp/kiểm tra tại [AI_usage.md](AI_usage.md).
