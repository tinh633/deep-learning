# Tiến độ và bàn giao — Practice 2

Cập nhật: 2026-10-27. **Đã chuẩn bị pipeline, EDA và GPU; chưa có lượt train chính thức hoàn tất.**
Phiên sau bắt đầu từ mục 3, đối chiếu file/Git thực tế trước khi thực hiện.

## 1. Tóm tắt việc đã làm

| Phần | Đã hoàn thành |
|---|---|
| Thiết kế | CIFAR-10, VGG16/ResNet18/DenseNet121; 6 cấu hình chung/model, tổng 18 lượt × 20 epoch |
| Dữ liệu | Tải/kiểm tra CIFAR-10; split seed 42: 45.000/5.000/10.000, không giao ID train/validation |
| EDA | Phân bố lớp, RGB/histogram, preprocessing, kiểm tra trùng pixel; 0 nhóm trùng chính xác |
| Viewer | Một eda.html, canvas đọc 6 RGB gzip; duyệt đủ 60.000 ảnh, lọc/tìm ID/phân trang/preview |
| Model | Ba backbone, H1/H2, freeze hoặc mở nhóm cuối; giữ BatchNorm running statistics |
| Train/log | Vòng lặp trực tiếp, validation mỗi epoch, log máy/config/kiến trúc, TensorBoard, best checkpoint |
| Evaluate | Đọc config checkpoint, xác minh split hash; metrics/confusion/per-class/ID/confidence/ảnh sai |
| GPU | `.venv` CUDA hoạt động với RTX 3050 Laptop 4 GB; worker 2 có guard Windows và giới hạn BLAS threads |
| Tài liệu | README, kế hoạch, Training, Model_QA và AI_usage; tiến độ riêng tại file này |

**Bằng chứng đã có:**

- 12 tổ hợp model/head/freeze qua forward/gradient/update/eval với weights ngẫu nhiên.
- Smoke CPU train/evaluate khớp checkpoint/scikit-learn; kiểm tra split sai và interrupt đã qua.
- Smoke GPU pretrained ResNet18 A: batch 64, worker 2, 128 train/64 validation, 1 epoch qua;
  checkpoint/TensorBoard hoạt động, peak CUDA allocated khoảng 488,8 MiB.
- Đối chiếu đủ 60.000 pixel/nhãn/hashes; Edge kiểm tra canvas, lọc/phân trang/preview,
  đọc file cục bộ không mạng và bố cục mobile đã qua.
- Smoke artifact đã xóa; các kiểm tra trên không được tính là kết quả của 18 lượt chính thức.

## 2. Trạng thái cần nhớ

- Config hiện tại: **ResNet18 / A / pretrained=True / H1 / freeze backbone**.
- Adam LR 0,001, weight decay 0, batch 64, 20 epoch; ảnh 224×224, seed 42.
- `num_workers=2`, `pin_memory=False`; giữ `RUN_TEST=False`.
- `.venv`: torch 2.11.0+cu126, torchvision 0.26.0+cu126, TensorBoard 2.21.0; pip check qua.
- `.venv-gpu` thừa đã xóa. Lượt CPU lan_1 trước đây bị dừng, không phải kết quả hoàn tất.
- Lần kiểm tra hiện tại: outputs/runs không có thư mục run; Git đang báo xóa artifact lan_1 cũ.
  Không tự khôi phục/xóa thêm; xác nhận trạng thái khi bắt đầu phiên tiếp theo.
- Đã push pipeline/GPU ở `8d0e610`, chuyển atlas sang RGB/canvas ở `5c7832e`, nhánh `Thái`.
- Người dùng đã yêu cầu commit/push toàn bộ tài liệu/cleanup sau review; đối chiếu commit mới nhất bằng Git.
- 236 atlas PNG đã xóa; giữ 6 file RGB phục vụ HTML. Archive/Python batches và env không nằm trên Git.

## 3. Flow chưa làm — tiếp tục theo thứ tự

### Bước 1 — Kiểm tra đầu phiên, chuẩn bị lượt đầu

- [ ] Đọc AGENTS.md/quy tắc chung, file này và Training.md; kiểm tra Git, config và artifact thực tế.
- [ ] Xác nhận dùng đúng `.venv`, CUDA=True, còn đủ dữ liệu/split; nếu chuyển máy phải tải lại CIFAR-10 gốc.
- [ ] Kiểm tra tiến trình đang chạy trước khi train; không khởi tạo hai lượt vô tình cùng lúc.

### Bước 2 — Chạy và review baseline A

- [ ] Người dùng bắt đầu ResNet18 A; kiểm tra log `device=cuda`, đúng pretrained/freeze/head/split.
- [ ] Chạy đủ 20 epoch; xác minh log Finished, history đủ epoch và best checkpoint load được.
- [ ] Dùng 3_evaluate.py để đánh giá train/validation, không bật test; xem loss/gap/lớp yếu/ảnh sai.
- [ ] Review lượt đầu với người dùng; không tự chuyển sang đợt train dài.
- [ ] Chạy VGG16 A và DenseNet121 A với cùng điều kiện; weights pretrained hai model này chưa kiểm tra thực tế.

### Bước 3 — Hoàn thành thí nghiệm có kiểm soát

- [ ] Tiếp tục B → C → D → E → F theo kế hoạch, mỗi cấu hình chạy đủ ba model.
- [ ] B đổi LR, C đổi batch, D đổi head, E/F mở nhóm cuối với hai LR; không thêm tổ hợp ngoài ngân sách.
- [ ] Lưu run/config/checkpoint/epoch/thời gian; đánh dấu thất bại/gián đoạn, không ghi đè run cũ.
- [ ] So sánh ba model trong cùng cấu hình và các cặp cấu hình trong cùng model bằng validation.

### Bước 4 — Tổng hợp và phân tích kết quả

- [ ] Lập bảng 18 lượt: cấu hình, trạng thái, best epoch, val loss/accuracy, tham số và chi phí thực tế.
- [ ] Phân tích per-class, confusion, ảnh sai chung/được sửa/sai thêm trên cùng ID và cùng split.
- [ ] Bổ sung kết quả model/learning curves/so sánh lỗi vào HTML hiện có nếu cần; hiện viewer chỉ có EDA.
- [ ] Viết nhận xét dựa trên dữ liệu; không kết luận chỉ từ accuracy hoặc gán tác dụng riêng cho từng thành phần H2.

### Bước 5 — Khóa cấu hình rồi đánh giá test

- [ ] Review và khóa một checkpoint cho mỗi model bằng validation trước khi bật test.
- [ ] Đánh giá ba checkpoint đã khóa trên test; không dùng test để chọn lại hyperparameters.
- [ ] Hoàn thiện báo cáo so sánh, giới hạn một seed và tổng kết checklist; không áp accuracy mục tiêu tự đặt.
- [ ] Cập nhật tiến độ/AI_usage và bàn giao; chỉ commit/push khi người dùng yêu cầu.

## 4. Tài liệu để tiếp tục

- [Experiment_Plan.md](Experiment_Plan.md): cấu hình A–F, kiểm soát thí nghiệm và checklist.
- [Training.md](Training.md): lệnh train/evaluate/TensorBoard, artifact và xử lý Windows/GPU.
- [Model_QA.md](Model_QA.md): giải thích model/head/thông số, bổ sung câu hỏi mới tại đây.
- [AI_usage.md](AI_usage.md): lịch sử phối hợp; [README.md](README.md): cấu trúc và EDA.

Chưa hoàn thành bước nào thì giữ trạng thái chưa làm; chỉ cập nhật sau khi có artifact/bằng chứng.
