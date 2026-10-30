# Practice 2 — Pretrained CNN trên CIFAR-10

Đã hoàn tất **VGG16 và DenseNet121 × ba cấu hình A/B/C**, mỗi lượt 20 epoch.
Python thực hiện train/evaluate; notebook trình bày phương pháp, kết quả và phân tích.

## Xem bài

1. Mở [Practice_2_Report.ipynb](Practice_2_Report.ipynb), chọn kernel `.venv`, Run All.
2. Mặc định chỉ đọc kết quả; không train, tải weights hoặc chạy inference/test lại.
3. Mở [visualization/eda.html](visualization/eda.html) bằng Live Server để xem dữ liệu/ảnh tương tác.
4. HTML đi theo EDA → không gian đặc trưng → dataset → tiền xử lý/kiểm tra → TensorBoard → lỗi test.

## Sáu lượt đã có

| Run | Model | Cấu hình | Best epoch | Test accuracy |
|---|---|---|---:|---:|
| lan_1 | DenseNet121 | A | 10 | 90,01% |
| lan_2 | DenseNet121 | B | 7 | 93,06% |
| lan_3 | DenseNet121 | C | 16 | 93,48% |
| lan_4 | VGG16 | A | 20 | 84,48% |
| lan_5 | VGG16 | B | 20 | 91,91% |
| lan_6 | VGG16 | C | 19 | 93,10% |

Các số liệu trên lấy từ báo cáo hiện tại; kết quả inference giữa máy từng lệch một ảnh.
A freeze backbone; B mở nhóm cuối, LR .001; C như B, LR .0001. Head Linear 10 lớp, batch64.
Pretrained ImageNet-1K V1, Adam, weight decay0, không scheduler/early stopping.
Mỗi run có best.pt/last.pt đúng tên và báo cáo train/validation/test khớp SHA256 best checkpoint.

## Dữ liệu và EDA

- CIFAR-10: 45.000 train / 5.000 validation / 10.000 test, phân tầng seed42.
- Resize224, chuẩn hóa ImageNet; chỉ train có random horizontal flip.
- EDA: phân bố lớp, RGB, duplicates, độ sáng/tương phản/bão hòa/biến thiên pixel, phân vị/theo lớp/tương quan.
- Một bản đồ PCA chung cho 60.000 ảnh, dùng RGB 8×8, tô màu theo lớp; xem ảnh khi rê chuột/nhấn điểm.
- Gallery lỗi ghép VGG/DenseNet cùng cấu hình và ID, có bộ lọc lớp/confidence và phân trang.
- Dataset/RGB local không đưa lên Git; các bảng và tọa độ không chứa pixel ảnh.

Nguồn: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html).

## Source và cách chạy

Chạy từ gốc workspace bằng `.venv`.

| File trong src/ | Vai trò |
|---|---|
| config.py | Hằng số, model/cấu hình A/B/C và run đang chọn |
| data.py | CIFAR-10, split cố định và DataLoader |
| model.py | Hai backbone, GAP → flatten → Linear10, freeze/fine-tune |
| logger.py | Log máy/cấu hình, epoch, TensorBoard, best/last và resume |
| 1_eda.py | EDA, PCA toàn dataset, outputs/eda.json và RGB local cho HTML |
| 2_train.py | Vòng lặp train/validation; không cần chạy lại để xem bài |
| 3_evaluate.py | Metrics/confusion/xác suất test từ best checkpoint |

```powershell
.\.venv\Scripts\python.exe practice_2_pretrained_model/src/1_eda.py
```

Chỉ chạy lại lệnh trên khi cần tái tạo artifact. Giữ nguyên split; không train lại để mở notebook/HTML.
Đánh giá lại qua cell 7.0: chọn EVALUATION_RUNS, bật RUN_EVALUATION; EVALUATE_TEST bật riêng.
Chi tiết train/resume: [Training.md](Training.md).

## TensorBoard

```powershell
.\.venv\Scripts\python.exe -m tensorboard.main --logdir practice_2_pretrained_model/outputs/runs --host 127.0.0.1 --port 6006 --reload_interval 2
```

Mở 127.0.0.1:6006 hoặc iframe HTML. Events local hiện có ở lan_2/lan_3; history đủ sáu lượt.
Live Server không khởi động TensorBoard. Không dựng event giả cho các run thiếu file gốc.

## Phạm vi và giới hạn

Notebook có bảng sáu lượt, train/val curves riêng, test bars, confusion ba split cùng hàng,
TPR/FPR, ROC-AUC, PR/AP và phân tích lỗi cùng cấu hình A/B/C.
Một seed, nhiều máy/phiên bản thư viện; không xếp hạng tốc độ kiến trúc bằng thời gian khác phần cứng.
Test đã được quan sát ở cả sáu cấu hình; không dùng để tiếp tục tune. Chưa thử nhiều batch/scheduler/weight decay.
Nhánh bàn giao `practice2-pretrained-model` lưu đủ 12 best/last bằng Git LFS, kèm events gốc hiện có.
Sau khi clone/checkout nhánh, chạy `git lfs install` rồi `git lfs pull` nếu checkpoint chưa được tải đầy đủ.
Dataset và ảnh pixel phát sinh vẫn giữ local; tái tạo bằng `1_eda.py` sau khi chuẩn bị dữ liệu.

Thiết kế: [Experiment_Plan.md](Experiment_Plan.md). Giải thích: [Model_QA.md](Model_QA.md).
Trạng thái/kiểm tra: [progress.md](progress.md). Nhật ký AI: [Step](AI_Usage_Step.md), [Full](docs/AI_Usage_Full.md).
