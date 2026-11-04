# logger.py — Log, checkpoint và resume

Source: [logger.py](../../src/logger.py).
Tập trung phần lưu trữ để vòng train giữ trực tiếp và dễ đọc.
Không tự chạy train/evaluate khi import.

## Run và log

- create_run tìm lan_N đầu tiên chưa tồn tại trong outputs/runs, không ghi đè run cũ.
- open_log tạo FileHandler UTF-8 và StreamHandler; log file được ghi nối.
- close_log đóng/gỡ handler, tránh lặp dòng khi mở lại logger.
- get_environment thu OS/Python/CPU/RAM/PyTorch/torchvision/CUDA/cuDNN/GPU,
  precision và các cờ deterministic/BLAS. Đây là snapshot môi trường, không đo tốc độ.

log_setup ghi config.json, môi trường/kiến trúc/config vào train.log và TensorBoard text.
Ghi tổng/trainable/frozen params, tên tham số trainable, cấu hình optimizer/loss,
loader/transform thực tế. Scheduler=None vì bài không dùng LR scheduler.
log_epoch ghi một dòng epoch và sáu scalar: train_loss, val_loss,
train_accuracy, val_accuracy, lr, seconds; sau đó flush TensorBoard.
Không ghi log từng batch.

## save_epoch

Kiểm tra mọi giá trị row hữu hạn, thêm row vào history.
Best được chọn theo val accuracy lớn nhất, val loss thấp nhất, epoch sớm nhất.
Checkpoint chứa:

| Field | Mục đích |
|---|---|
| model_state, settings, weights, classes | Khôi phục kiến trúc/trọng số và provenance |
| epoch, metrics | Mốc epoch hoàn tất và số đo |
| optimizer_state | Trạng thái Adam |
| random_state, numpy_state, torch_state | RNG trên CPU |
| cuda_state | RNG CUDA nếu train bằng GPU |
| loader_state | Generator của train DataLoader |

Chỉ ghi best.pt khi epoch mới tốt nhất; history.json ghi toàn bộ các epoch;
last.pt cập nhật mỗi epoch. Checkpoint best cũng chứa trạng thái của epoch best.
Dùng file .tmp rồi replace cho từng file, giảm nguy cơ file bị ghi dở.
Các file không phải một transaction chung: mất điện giữa nhiều lần replace
vẫn có thể để history/checkpoint ở mốc khác nhau.

## resume_training

Đọc last.pt lên CPU, kiểm tra model_name/experiment/model_config/train/data/split_sha256
khớp settings hiện tại. Cắt history đến epoch checkpoint, yêu cầu epoch liên tục1..N.
Khôi phục model/weights_name, Adam, RNG và generator; trả history đã đối chiếu.
Train bắt đầu ở len(history)+1. CUDA RNG chỉ khôi phục khi chạy GPU và có state.
Không lưu trạng thái giữa batch hoặc RNG riêng từng worker đang chạy.
Ngắt giữa epoch làm phần chưa lưu của epoch đó phải chạy lại từ cuối epoch trước.
Không cam kết bit-for-bit khi đổi máy, phiên bản thư viện hoặc số workers.

## Tiện ích và lưu ý

save_json ghi UTF-8/indent2, cấm NaN rồi replace file đích.
file_sha256 đọc theo luồng, nhận diện chính xác bytes split/checkpoint,
không tham gia gradient và không làm model học tốt hơn.
Checkpoint chứa NumPy RNG nên resume dùng weights_only=False;
chỉ nạp checkpoint có nguồn tin cậy của project.

Liên quan: [train](2_train.md), [evaluate](3_evaluate.md).
