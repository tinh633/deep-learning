# Tiến độ và bàn giao — Practice 2

Cập nhật: 2026-10-28. **ResNet18 A hoàn tất tại lan_1; VGG16 A đang chạy tại lan_2 trên GPU.**
Theo yêu cầu người dùng đã xóa lượt lỗi, đổi tên lượt ResNet18 thành công lan_2 → lan_1.
Weights VGG16 tải xong và SHA-256 khớp; preflight GPU batch64 qua, peak khoảng 2.446 MiB.
Pipeline cũ 66286 từng dừng do sandbox không đọc được cache; đã chạy lại với quyền đọc phù hợp.
Session hiện tại 95990: VGG16 A bắt đầu 16:00:25, worker 0, batch 64, 20 epoch; chưa xác nhận epoch hoàn tất.
Không khởi chạy thêm lượt trùng. Config/log lan_2 xác nhận CUDA và IMAGENET1K_V1.

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
| Tài liệu | README, kế hoạch, Training, Model_QA, AI_Usage_Step/Full; tiến độ riêng tại file này |

**Bằng chứng đã có:**

- 12 tổ hợp model/head/freeze qua forward/gradient/update/eval với weights ngẫu nhiên.
- Smoke CPU train/evaluate khớp checkpoint/scikit-learn; kiểm tra split sai và interrupt đã qua.
- Smoke GPU pretrained ResNet18 A: batch 64, worker 2, 128 train/64 validation, 1 epoch qua;
  checkpoint/TensorBoard hoạt động, peak CUDA allocated khoảng 488,8 MiB.
- Đối chiếu đủ 60.000 pixel/nhãn/hashes; Edge kiểm tra canvas, lọc/phân trang/preview,
  đọc file cục bộ không mạng và bố cục mobile đã qua.
- Smoke artifact đã xóa; các kiểm tra trên không được tính là kết quả của 18 lượt chính thức.

## 2. Trạng thái cần nhớ

- Config hiện tại: **VGG16 / A / pretrained=True / H1 / freeze backbone**.
- Adam LR 0,001, weight decay 0, batch 64, 20 epoch; ảnh 224×224, seed 42.
- `num_workers=0`, `pin_memory=False`; giữ `RUN_TEST=False`. GPU vẫn được dùng khi worker bằng 0.
- `.venv`: torch 2.11.0+cu126, torchvision 0.26.0+cu126, TensorBoard 2.21.0; pip check qua.
- `.venv-gpu` thừa đã xóa. Run lỗi worker 2 đã xóa theo yêu cầu; từng lỗi shared mapping Windows 1455.
- `lan_1` hiện là ResNet18 A thành công (trước đổi tên là lan_2), 20/20 epoch và 14.080 optimizer steps.
- Best epoch 18: val 88,54%, loss 0,337606; đã đối chiếu history và load được checkpoint.
- Hoàn tất lúc 14:20:32; tổng thời gian khoảng 3.437 giây. Đây chưa phải kết quả test.
- Log cũ giữ nguyên đường dẫn lịch sử lan_2; đã thêm dòng ghi nhận đổi tên sang lan_1.
- Đã evaluate lan_1: train 89,6533%, loss 0,295765; val 88,54%, loss 0,337606, macro-F1 0,885577.
- Báo cáo metrics/ID/confusion và PNG lỗi đã lưu; tổng số mẫu/lỗi khớp, test chưa chạy.
- Commit bàn giao gần nhất: `e7c2848` trên `Thái`; lượt train mới và cập nhật tiến độ chưa commit/push.
- Người dùng đã yêu cầu commit/push toàn bộ tài liệu/cleanup sau review; đối chiếu commit mới nhất bằng Git.
- 236 atlas PNG đã xóa; giữ 6 file RGB phục vụ HTML. Archive/Python batches và env không nằm trên Git.
- Yêu cầu mới: không commit bất kỳ data nào, kể cả RGB viewer. Bỏ theo dõi 6 file RGB đã push,
  giữ file trên máy, bỏ ngoại lệ Git ignore; clone mới cần tải CIFAR-10 và chạy lại EDA.
- Đã push 4bdb86b lên Thái: không còn file data trong cây Git hiện tại; không rewrite lịch sử cũ.

## 3. Flow chưa làm — tiếp tục theo thứ tự

### Bước 1 — Kiểm tra đầu phiên, chuẩn bị lượt đầu

- [ ] Đọc AGENTS.md/quy tắc chung, file này và Training.md; kiểm tra Git, config và artifact thực tế.
- [ ] Xác nhận dùng đúng `.venv`, CUDA=True, còn đủ dữ liệu/split; nếu chuyển máy phải tải lại CIFAR-10 gốc.
- [ ] Kiểm tra tiến trình đang chạy trước khi train; không khởi tạo hai lượt vô tình cùng lúc.

### Bước 2 — Chạy và review baseline A

- [x] Theo yêu cầu người dùng đã bắt đầu ResNet18 A; log xác nhận CUDA, pretrained/freeze/head/split đúng.
- [x] ResNet18 A đủ 20 epoch; xác minh Finished, 20 history rows và best checkpoint đọc được.
- [x] Evaluate ResNet18 A train/validation, test tắt; đã xuất metrics/confusion/per-class/ảnh sai.
- [x] Người dùng yêu cầu tiếp tục lượt tiếp theo; thực hiện VGG16 A theo thứ tự ba model ở cấu hình A.
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
- [ ] Cập nhật progress và AI_Usage_Step; cuối giai đoạn tổng hợp vào AI_Usage_Full; chỉ commit/push khi được yêu cầu.

## 4. Tài liệu để tiếp tục

- [Experiment_Plan.md](Experiment_Plan.md): cấu hình A–F, kiểm soát thí nghiệm và checklist.
- [Training.md](Training.md): lệnh train/evaluate/TensorBoard, artifact và xử lý Windows/GPU.
- [Model_QA.md](Model_QA.md): giải thích model/head/thông số, bổ sung câu hỏi mới tại đây.
- [AI_Usage_Step.md](AI_Usage_Step.md): ghi từng giai đoạn; [AI_Usage_Full.md](docs/AI_Usage_Full.md): lịch sử tổng hợp.
- [README.md](README.md): cấu trúc và EDA. Đã đổi tên/hợp nhất log Full/Step theo yêu cầu; không đổi source.
- Kế hoạch đã bổ sung đúng ImageNet-1K/IMAGENET1K_V1 cho cả ba model, khớp torchvision local; không đổi weights trong code.

Chưa hoàn thành bước nào thì giữ trạng thái chưa làm; chỉ cập nhật sau khi có artifact/bằng chứng.

## Handoff - 2026-10-29

- ResNet18 A (lan_1): completed 20 epochs; best epoch 18, validation accuracy 88.54%. No last checkpoint exists for this older run.
- VGG16 A (lan_2): restarted from scratch at user request, then stopped after epoch 1. best.pt/last.pt verified with Adam state; next epoch is 2.
- RESUME_RUN='lan_2'; no training restart authorized in this Git session. Local checkpoints are required to resume.
- New checkpoints save optimizer and Python/NumPy/Torch/CUDA/DataLoader RNG; resume validates experiment, data and split settings.
- CPU smoke checks passed for best/last saving, same-folder resume, state restoration and legacy fallback.
- Text logs and TensorBoard now record epoch metrics only; batch tracking, live.jsonl and progress.json removed.
- EDA HTML embeds TensorBoard directly; no training.js or custom train dashboard. Server/API previously verified; browser embedding not yet verified.
- Live Server ignores outputs/data/events to prevent artifact changes from reloading the page; cause was inferred, not reproduced in browser.
- Earlier temporary legacy resume/reset-Adam trial was deleted when the user requested a fresh lan_2. See AI usage records for history.
- Git excludes datasets, weights, TensorBoard events and dataset-derived error images; local files remain available.
- User authorized commit and push to Thai branch on 2026-10-29. Check Git history for final commit identity.
