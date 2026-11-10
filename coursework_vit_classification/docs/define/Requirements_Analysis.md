# Define — Định nghĩa đồ án

Cập nhật: 10/11/2026. Đã có data/EDA; model/train/evaluate/demo chưa triển khai.
Phạm vi ViT/CIFAR-100 đang chờ giảng viên xác nhận; chưa mở rộng sang Anti Face Deepfake.
Thiết kế: [Implementation_Plan.md](../design/Implementation_Plan.md). Trạng thái: [progress](../progress.md).

## 1. Project name — Tên đề tài

**Phân loại ảnh CIFAR-100 bằng Vision Transformer và đối chiếu CNN trong học chuyển giao.**
Model chính: ViT-B/16. Baseline: ResNet-50. Cả hai sử dụng pretrained ImageNet-1K.
Đây là đồ án vận dụng, thực hiện trong 4 tuần bởi nhóm 6 thành viên.

## 2. Problem — Bài toán và câu hỏi thực nghiệm

Xây dựng pipeline phân loại ảnh RGB vào một trong 100 lớp CIFAR-100 bằng ViT.
Ảnh nhỏ, nhiều lớp và chỉ 500 ảnh train gốc/lớp làm việc học đặc trưng và tổng quát hóa khó.
Tận dụng pretrained giúp khảo sát chuyển giao đặc trưng mà không phải train backbone từ đầu.

Ba câu hỏi cần trả lời bằng kết quả thực:

1. Khi chỉ train head, đặc trưng pretrained CNN hay ViT phù hợp dữ liệu hơn?
2. Fine-tune toàn bộ giúp mỗi model cải thiện bao nhiêu so với freeze?
3. Chất lượng dự đoán đánh đổi thế nào với thời gian, bộ nhớ và số tham số?

Không đặt giả thuyết bắt buộc ViT thắng CNN và không có ngưỡng accuracy bắt buộc trong đề đã cung cấp.
Không suy diễn đây là so sánh kiến trúc thuần túy vì số tham số và recipe pretraining khác nhau.

## 3. Target user — Người sử dụng

| Nhóm | Nhu cầu | Cách sử dụng |
|---|---|---|
| Thành viên nhóm | Chạy và giải thích thí nghiệm nhất quán | Cấu hình, script, log và checkpoint |
| Giảng viên/người đánh giá | Kiểm chứng phương pháp, baseline và kết quả | Notebook, tài liệu, demo và artifact |
| Người xem demo | Hiểu đầu vào và dự đoán trên phạm vi 100 lớp | Ảnh mẫu, Top-5 và thông tin model |

Chưa hướng tới người dùng thương mại hoặc dịch vụ nhận diện mọi loại ảnh.
Ảnh ngoài 100 lớp vẫn bị ép vào một lớp; demo phải thể hiện giới hạn này.

## 4. Input/output — Dữ liệu vào và sản phẩm ra

| Giai đoạn | Đầu vào | Đầu ra |
|---|---|---|
| Data/EDA | CIFAR-100 gốc, seed | Split cố định, thống kê, gallery, PCA/t-SNE |
| Train | Split train/val, pretrained, cấu hình A–D | Best/last checkpoint, history, log và metadata |
| Evaluate | Checkpoint đã chọn bằng val, split đánh giá | Metric, confusion, ID/nhãn/xác suất dự đoán |
| Report | Artifact của bốn thí nghiệm | Bảng so sánh, biểu đồ, lỗi điển hình và giới hạn |
| Demo | Ảnh RGB, checkpoint tương thích | Top-1/Top-5 trong 100 lớp, confidence |

Model nhận tensor B × 3 × 224 × 224 sau preprocessing và trả logits B × 100.
Softmax dùng để trình bày dự đoán, không đặt trước CrossEntropyLoss khi train.
Confidence không mặc định là xác suất đã hiệu chỉnh hoặc độ chắc chắn tuyệt đối.

## 5. Core features — Chức năng cốt lõi

- Chuẩn bị dữ liệu có nguồn rõ ràng; kiểm tra integrity, nhãn và split.
- EDA: phân bố lớp, RGB, ảnh cực trị, ảnh trùng, PCA/t-SNE và gallery.
- Bốn chiến lược: CNN freeze, CNN fine-tune, ViT freeze, ViT fine-tune.
- Tuning nhỏ: hai LR/chiến lược bằng pilot cùng ngân sách; chọn bằng validation, không dùng test.
- Freeze toàn backbone, chỉ học head mới; fine-tune học backbone và head.
- Khởi tạo bốn run độc lập từ pretrained; không huấn luyện B nối tiếp A hoặc D nối tiếp C.
- Ghi cấu hình/máy, một dòng log/epoch, best/last và resume theo epoch hoàn tất.
- Đánh giá checkpoint nhất quán, phân tích theo lớp và đối chiếu lỗi trên cùng ID ảnh.
- Notebook trình bày và demo dùng cùng pipeline, không nhân bản logic train.

Phần mở rộng trong hướng dẫn: CNN/ViT cùng số tham số và trực quan hóa attention.
Cặp hiện tại chưa đáp ứng cùng số tham số; attention chỉ xem xét sau phần cốt lõi.
Không mở rộng sang deepfake, detection, segmentation, train từ đầu hoặc tìm kiếm hàng loạt model.

## 6. Technology constraints — Ràng buộc kỹ thuật

- Python/PyTorch/torchvision; code trực tiếp, module nhỏ, không thêm Trainer framework.
- Dùng ViT nguyên bản vit_b_16; không thay bằng DeiT/Swin mà không thống nhất lại.
- Enum weights chỉ định rõ ImageNet-1K; ghi phiên bản và provenance từng checkpoint.
- CIFAR-100: 60.000 ảnh RGB 32 × 32, 100 fine labels; coarse labels không là đích phân loại.
- Split hiện có: 45.000 train, 5.000 val, 10.000 official test; phân tầng seed 42.
- Phát hiện 12 nhóm trùng pixel xuyên split; chính sách xử lý cần chốt trước train chính thức.
- Không tự sửa split hoặc xóa ảnh; nếu đổi protocol phải tạo phiên bản và ghi đầy đủ lý do.
- Fit thống kê phục vụ lựa chọn preprocessing chỉ trên train; augmentation chỉ ở train.
- Không dùng test để tinh chỉnh LR, số epoch, checkpoint hoặc chọn cấu hình cho lượt tiếp theo.
- GPU local từng dùng có 4 GB; khả năng fine-tune ViT phải kiểm chứng bằng smoke run.
- LR, batch, weight decay, scheduler và epoch chưa chốt; không ghi mục tiêu thời gian thiếu số đo.
- Không phát sinh dịch vụ trả phí hoặc đổi model/phạm vi khi chưa thống nhất.
- Dataset, pixel dẫn xuất và cache không commit; không tạo artifact train giả.
- Resize 32 lên 224 không phục hồi chi tiết ảnh đã mất.

## 7. Evaluation criteria — Tiêu chí đánh giá và nghiệm thu

| Nhóm tiêu chí | Bằng chứng cần có |
|---|---|
| Đúng yêu cầu | ViT phân loại CIFAR-100; có CNN baseline, giải thích đúng kiến trúc |
| Đúng quy trình | Split/provenance xác định, không tuning trên test, freeze/fine-tune đúng |
| Chất lượng | Top-1 accuracy, macro-F1; bổ sung Top-5 và precision/recall/F1 theo lớp |
| Phân tích lỗi | Confusion 100 lớp, cặp nhầm nổi bật và ảnh sai có ID |
| Chi phí | Thời gian train, inference latency/throughput, peak VRAM và tham số |
| Tái lập | Config, seed, môi trường, history, checkpoint và hướng dẫn chạy |
| Sản phẩm | Source, dữ liệu mẫu local, notebook, demo, tài liệu và giới hạn |

So sánh A↔C, B↔D và A↔B/C↔D; không bắt buộc một model thắng tất cả tiêu chí.
Best checkpoint theo validation accuracy, hòa thì loss thấp hơn; test sau khóa protocol.
Một seed/cấu hình là phạm vi ban đầu, không đủ chứng minh độ ổn định thống kê.
ROC-AUC/AP đa lớp là bổ sung nếu thời gian cho phép, phải ghi rõ one-vs-rest và averaging.
Mục tiêu accuracy kỳ vọng chỉ ghi sau pilot có căn cứ, không lấy làm lời hứa đạt điểm tối đa.

## 8. Đối chiếu đầu bài và việc còn cần thống nhất

Đáp ứng phần đề tài ViT classification, dataset được cho phép, baseline, demo và phân tích giới hạn.
Không coi phần mở rộng cùng số tham số hoặc attention là đã hoàn thành.
Cần xác nhận hai dòng ViT/Anti Face Deepfake là lựa chọn hay yêu cầu kết hợp trước triển khai tiếp.
Sau đó chốt ảnh trùng, GPU và thông số sau smoke trước bốn run chính thức.
Nguồn dữ liệu: [CIFAR chính thức](https://www.cs.toronto.edu/~kriz/cifar.html).
