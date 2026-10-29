# Practice 2 — Vận hành train/evaluate

## 1. Phạm vi

- 2 model (VGG16/DenseNet121) × 3 cấu hình A/B/C, tối đa 20 epoch/lượt.
- A/B đo tác động fine-tune; B/C đo tác động LR; bám checklist và tiết kiệm thời gian.
- Cùng split seed 42, batch 64, Adam/CE, không scheduler/early stopping.

## 2. Các file và thứ tự đọc

| File | Vai trò/đầu ra |
|---|---|
| Practice_2_Report.ipynb | Mục tiêu, EDA, kiến trúc, ma trận, đọc artifact/biểu đồ và kết luận |
| config.py | MODEL_NAME, EXPERIMENT, RESUME_RUN, EPOCHS và các hằng số |
| data.py | Dataset/split/transforms/DataLoader; training=False tắt augmentation/shuffle |
| model.py | Hai backbone, head Linear 10 lớp, freeze/unfreeze và BatchNorm |
| logger.py | Log môi trường/config/TensorBoard và best/last/resume state |
| 1_eda.py | outputs/eda.json và RGB viewer local |
| 2_train.py | Vòng lặp train/validation trực tiếp, metrics theo epoch |
| 3_evaluate.py | Đọc config checkpoint thật, metrics/confusion/per-class/ID/ảnh sai |

Không import runnable script để dùng lại code; không tạo pipeline train trong notebook.

## 3. Notebook và EDA

- Mở Practice_2_Report.ipynb bằng kernel .venv; chạy từ gốc workspace hoặc thư mục practice.
- Run All không train, không tải weights, không evaluate test; thiếu artifact thì báo thiếu.
- Cell kiến trúc chỉ tạo model weights ngẫu nhiên trên CPU, không phải kết quả học.
- Giữ notebook không có outputs ảnh/pixel để Git không chứa dataset dẫn xuất; chạy local để xem.
- Dataset thiếu: tải rõ ràng qua data.load_datasets(download=True) trước, không tự tải khi import.

Chạy EDA từ gốc workspace:

```powershell
.\.venv\Scripts\python.exe practice_2_pretrained_model/src/1_eda.py
```

## 4. Chuẩn bị train

1. Kiểm tra không có tiến trình train khác; dùng .venv CUDA đã cài.
2. Chọn MODEL_NAME='vgg16' hoặc 'densenet121', EXPERIMENT='A'/'B'/'C'.
3. A freeze LR .001; B fine-tune LR .001; C fine-tune LR .0001; tất cả H1/batch64.
4. Lượt mới đặt RESUME_RUN=None; mỗi lượt khởi tạo từ pretrained ImageNet-1K V1.
5. Resume đặt tên run đang dở, giữ đúng model/cấu hình/split/ngân sách của checkpoint.
6. Giữ RUN_TEST=False. Không tự chuyển từ A sang B/C bằng weights đã học của A.

```powershell
.\.venv\Scripts\python.exe practice_2_pretrained_model/src/2_train.py
```

- num_workers=0 tránh lỗi Windows shared mapping 1455; GPU vẫn dùng bình thường.
- Giữ main guard để an toàn khi đổi worker; không cần Trainer/callback framework.
- Cấu hình hiện tại trỏ resume lan_1 VGG16 A đã tạm dừng; không tự chạy từ notebook.
- PretrainedCNN nhận name/fine_tune/pretrained; head cố định Linear 10 lớp, không có lựa chọn head.
- Không chạy B/C cũ theo tên mã: B/C mới khác kế hoạch 18 lượt, kiểm tra model_config thực.

## 5. Log, checkpoint và ngắt phiên

- Đầu lượt: máy/thư viện/device, source/split hash, seed, transforms, kiến trúc, tham số, loss/optimizer.
- Mỗi epoch: train/val loss, accuracy, LR, seconds, best epoch; TensorBoard cùng sáu scalar.
- Không log batch, không tạo live.jsonl/progress.json.
- best.pt: val accuracy cao hơn; hòa chọn loss thấp hơn; vẫn hòa giữ epoch sớm hơn.
- last.pt: model/Adam/random Python/NumPy/Torch/CUDA/DataLoader state ở epoch hoàn tất gần nhất.
- Ghi checkpoint/JSON bằng file tạm rồi replace; không ghi đè trực tiếp file tốt đang có.
- Ngắt giữa epoch: phần epoch đang dở không được lưu làm epoch hoàn tất; resume chạy lại epoch đó.
- Resume bắt buộc last.pt có đầy đủ Adam/RNG; không fallback best hoặc reset optimizer.
- Checkpoint local cần giữ; clone Git không có weights để resume vì weights bị ignore.
- SHA-256 chỉ xác minh split/định danh checkpoint, không tham gia tính loss/accuracy.

## 6. Đánh giá

Chọn EVAL_RUN, đọc best.pt và cấu hình đã lưu; không dùng config model hiện tại thay checkpoint.

```powershell
.\.venv\Scripts\python.exe practice_2_pretrained_model/src/3_evaluate.py
```

- Mặc định chỉ train/validation; training=False dùng preprocessing evaluation và không shuffle.
- model.eval()/no_grad(); lưu *_metrics.json, evaluate.log và ảnh sai local.
- Báo cáo loss, accuracy, macro-F1, precision/recall/F1/support/TP/TN/FP/FN, confusion, ID và confidence.
- Per-class có thêm TPR/FPR/specificity; mẫu số 0 lưu null. Test report lưu probabilities N×10 và probability_class_order, cùng thứ tự predictions; train/validation không lưu ma trận score.
- Notebook phân tích ROC OvR từng lớp, macro/micro AUC và bảng/đường cong so sánh sau test. Báo cáo cũ thiếu score chỉ tính được tỷ lệ confusion, không suy ra AUC.
- PR/AP dùng cùng xác suất test; AP không phải tích phân hình thang. Đối chiếu lỗi ghép theo ID/nhãn, có bảng bốn nhóm, theo lớp và tối đa bốn ảnh mỗi nhóm sai; cặp nhiều checkpoint chọn bằng PAIR_LABELS, không xếp hạng test tự động.
- Khóa một checkpoint/model bằng validation, xác nhận rồi mới bật RUN_TEST và test hai checkpoint.
- Không tự dùng test để chọn epoch/LR, không xem run dở như kết quả hoàn tất.
- Evaluate đọc trusted local checkpoint với weights_only=False vì checkpoint hiện có NumPy RNG.

## 7. TensorBoard

```powershell
.\.venv\Scripts\python.exe -m tensorboard.main --logdir practice_2_pretrained_model/outputs/runs --host 127.0.0.1 --port 6006 --reload_interval 2
```

Mở localhost:6006 hoặc iframe trong visualization/eda.html. Server không cần để notebook đọc history.
Live Server ignore outputs/data/events để giảm reload theo thay đổi artifact; nguyên nhân reload chưa tái hiện bằng browser.

## 8. Bằng chứng kiểm tra và giới hạn

- Trước tái cấu trúc: CPU continuous/ngắt-resume khớp weights; best/last/history qua, config/split sai bị chặn.
- Evaluate smoke khớp sklearn/ID/ảnh sai/test tắt; DataLoader CIFAR thật giữ 45k/5k/10k và tắt augmentation khi eval.
- Notebook: toàn bộ code cell local, nhánh thiếu artifact và config/checkpoint tương thích đã kiểm tra qua; UI IDE chưa kiểm tra.
- Không coi smoke test là kết quả sáu lượt chính thức. Xem progress.md cho kiểm tra mới nhất.
