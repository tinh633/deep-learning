# Huấn luyện, logging và đánh giá — Practice 2

## Trạng thái

- `logger.py`, `2_train.py` và `3_evaluate.py` đã triển khai và smoke test.
- Các kiểm tra của agent là smoke test; lượt CPU người dùng đã dừng để chuyển GPU, chưa hoàn tất.
- `.venv` hiện dùng Python 3.14, torch 2.11.0+cu126, torchvision 0.26.0+cu126,
  TensorBoard 2.21.0; ResNet18 pretrained đã qua smoke test CUDA và worker 2.
- Đã cài TensorBoard vào `.venv`; các dependency nằm trong `requirements.txt`.

## Trước khi train chính thức

1. Dùng môi trường Python có các dependency trong `requirements.txt`.
2. Kiểm tra dữ liệu/split và review EDA, model, logging/training.
3. Trong `src/config.py`, chọn `MODEL_NAME` và `EXPERIMENT` A–F.
4. Giữ `PRETRAINED=True` cho các lượt chính thức. Lần đầu model có thể tải ImageNet weights.
5. `TRAIN` cố định 20 epoch, weight decay 0; optimizer Adam, không scheduler.
6. Dùng đúng `.venv`; kiểm tra log có `device=cuda`. RTX 3050 Laptop 4 GB đã được xác minh.

Từ gốc workspace, sau review có thể chạy:

```powershell
.\.venv\Scripts\python.exe practice_2_pretrained_model/src/2_train.py
```

Lệnh này chạy một lượt thật theo config hiện tại, không phải smoke test.
Chưa chạy lệnh nguyên cấu hình chính thức trong phiên triển khai này.
Chạy lại tạo lượt mới, không resume hoặc ghi đè run cũ.

## Luồng code

- Script train: guard Windows → setup → vòng lặp epoch/batch trực tiếp → validation → checkpoint.
- Forward/loss/backward/step ở ngay trong script; không Trainer hoặc callback.
- Optimizer chỉ nhận các tham số `requires_grad=True`, gồm head và block được mở.
- `model.train()` giữ chính sách frozen backbone/BatchNorm; validation dùng eval/no_grad.
- Chỉ train/validation được duyệt; test loader không được iterate và không có test metrics.
- Chọn best theo val accuracy, hòa thì val loss, vẫn hòa giữ epoch sớm nhất.
- Non-finite loss/metrics báo lỗi. Interrupt/lỗi giữ artifact đã hoàn thành, không ghi Finished.

## Artifact mỗi lượt

Trong `outputs/runs/lan_N/`:

| Artifact | Nội dung |
|---|---|
| `train.log` | Cấu hình đầy đủ, kiến trúc, mỗi epoch và trạng thái cuối |
| `config.json` | Máy/thư viện/device, split hash, transforms, tham số, loss/optimizer thực tế |
| `history.json` | Train/validation loss/accuracy, LR, thời gian từng epoch hoàn tất |
| `best.pt` | State dict, config, weights identifier, epoch, metrics và tên lớp |
| `tensorboard/` | Event files: config/architecture và metrics mỗi epoch |

Best checkpoint được ghi qua file tạm rồi thay thế, tránh ghi đè trực tiếp checkpoint tốt nhất.
Không lưu optimizer state/resume; checkpoint phục vụ đánh giá và tái dựng model.
Log cuối ghi tổng thời gian, số optimizer step, số epoch hoàn tất và GPU memory peak nếu có.
Log flush trong quá trình chạy; TensorBoard writer và log handlers được đóng ở finally.

## Xem TensorBoard

```powershell
.\.venv\Scripts\python.exe -m tensorboard.main --logdir practice_2_pretrained_model/outputs/runs
```

Mở địa chỉ localhost do TensorBoard in ra. Không có dashboard kết quả chính thức khi chưa train.

## Bằng chứng smoke test

- ResNet18 A/E, weights ngẫu nhiên, 2 epoch trên 8 train/4 validation CIFAR-10 thật.
- Reload best checkpoint cho validation loss/accuracy khớp; sáu scalar TensorBoard đủ mỗi epoch.
- Kiểm tra lỗi và interrupt lúc setup: có trạng thái đúng, không Finished giả.
- H2 một epoch và loss options trong config.json đã kiểm tra; ngắt giữa epoch 2 giữ được
  history/checkpoint epoch 1, trạng thái Interrupted đúng, không báo Finished.
- Các kiểm tra ghi vào thư mục tạm đã xóa; không tính vào 18 lượt A–F.
- Đây là kiểm tra luồng, không dùng accuracy để đánh giá chất lượng hoặc chọn model.

## Đánh giá checkpoint

1. Chọn `EVAL_RUN='lan_N'` trong config.py, trỏ đến run có `best.pt`.
2. Giữ `RUN_TEST=False` khi đối chiếu train/validation và chọn cấu hình.
3. Chỉ bật test sau review và khóa ba checkpoint theo kế hoạch; không chọn bằng test.

```powershell
.\.venv\Scripts\python.exe practice_2_pretrained_model/src/3_evaluate.py
```

- Script dùng config trong checkpoint, load state dict strict và không tải lại ImageNet weights.
- Kiểm tra SHA-256 split trước khi chạy; dùng split hiện tại chỉ khi khớp checkpoint.
- Train eval không shuffle/augmentation, model ở eval/no_grad; khác metrics online khi train.
- Ghi thêm `evaluate.log`: máy/device, checkpoint hash, config/weights, kiến trúc, metrics và trạng thái.
- Mỗi split có `{split}_metrics.json`: loss, accuracy, macro-F1, precision/recall/F1/support,
  TP/TN/FP/FN từng lớp, confusion count/chuẩn hóa và tất cả ID/nhãn/dự đoán/confidence.
- Confusion: hàng nhãn thật, cột dự đoán; macro-F1 tính đủ 10 lớp, zero-support dùng zero_division=0.
- Lưu tối đa 24 ảnh sai confidence cao nhất tại `visualization/runs/lan_N/{split}_errors.png`,
  với split, ID gốc, nhãn thật/dự đoán và confidence. Không kết luận nhãn gốc sai từ dự đoán.
- Mỗi báo cáo lưu checkpoint hash và thời gian đánh giá; không đụng báo cáo test cũ khi test tắt.
- Kết quả từng run phục vụ so sánh sau này; chưa thêm bảng so sánh model vào HTML EDA.

## Kiểm tra phần evaluate

- Train/evaluate trên 8 train/4 validation CIFAR thật, weights ngẫu nhiên trong thư mục tạm.
- Loss/accuracy khớp checkpoint; sklearn metrics, TP/TN/FP/FN, zero-support, IDs và PNG sai: qua.
- Test mặc định bị chặn bằng sentinel; nhánh RUN_TEST=True kiểm tra trên 5 ảnh giả, không dùng test thật.
- Split hash sai bị từ chối, báo cáo cũ được giữ. Các artifact tạm đã xóa.
- Chưa đánh giá checkpoint pretrained đã train chính thức; smoke evaluate trước đây dùng CPU.

## Chuyển `.venv` sang GPU — 2026-10-27

- Cài đúng cặp CUDA từ nguồn PyTorch; requirements.txt ghim lại phiên bản đã kiểm tra:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu126
```

- `torch.cuda.is_available()` là True; GPU NVIDIA GeForce RTX 3050 Laptop GPU.
- `num_workers=2`, `pin_memory=False` theo config người dùng. Không đổi LR/head/batch của A.
- Train/evaluate có main guard để worker Windows không chạy lặp toàn bộ script.
- Giới hạn BLAS/OMP/MKL về 1 thread nếu biến môi trường chưa được đặt; logger lưu giá trị thực tế.
  Bổ sung sau lỗi OpenBLAS thiếu bộ nhớ ở lần kiểm tra worker đầu tiên.
- GPU smoke: ResNet18 pretrained A, batch 64, 128 train/64 validation CIFAR thật, 1 epoch.
  Worker 2, checkpoint/TensorBoard qua; peak CUDA allocated khoảng 488,8 MiB.
- Chỉ kiểm tra luồng trên tập nhỏ; không dùng thời gian/accuracy này làm kết quả full epoch.
- `pip check` không có dependency lỗi. Script và artifact smoke tạm đã xóa.
- Chưa chạy lượt GPU 20 epoch; khởi chạy mới sẽ tạo lan_N mới, không resume lượt CPU cũ.
- Trong VS Code chọn interpreter `D:\newtask\DL\.venv\Scripts\python.exe`.
