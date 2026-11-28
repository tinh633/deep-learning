# logger.py — Artifact, bộ nhớ và resume

Source: [logger.py](../../src/logger.py).

| 5W1H | Mô tả |
|---|---|
| What | Log/metadata, JSON, checkpoint, RAM guard và tìm run |
| Why | Loop train dễ đọc, có bằng chứng và khả năng tiếp tục sau ngắt |
| Which | logging/JSON, PyTorch checkpoint, psutil và Windows memory API; ngưỡng 2 GiB RAM/3 GiB commit |
| When | Đầu run, mỗi epoch, trước/sau bước nặng và khi restore |
| Where | outputs/runs/lan_N, checkpoint local và config/log/summary |
| How | logging, ghi tạm/replace, torch.save/load mmap, psutil và Windows memory API |

## Giao diện

- create_run(settings, folder=None): tạo lan_N chưa tồn tại hoặc mở log append cho resume.
- Config gốc không bị create_run ghi đè khi resume; config runtime/source được ghi vào log/checkpoint.
- save_json: từ chối NaN, ghi tạm rồi replace.
- save_checkpoint: model, optimizer, scaler, epoch, settings, RNG và extra history/best score.
- read_checkpoint: trusted pickle local, mmap CPU, kiểm tra model/epoch/settings.
- load_checkpoint: đối chiếu các trường config quan trọng nếu có expected; restore model/optimizer/scaler/RNG.
- latest_checkpoint: chọn epoch mới nhất trong best/last; xử lý ngắt sau best nhưng trước last.
- resolve_run: chỉ chấp nhận lan_N tồn tại, không nhận đường dẫn tùy ý.

Model/optimizer state cũ được giải phóng trước restore để giảm peak bộ nhớ.
Checkpoint là nguồn history khi resume, không dựa vào history.json có thể chưa kịp ghi sau ngắt.
Epoch budget/LR/phase/split/preprocessing/batch/seed không được đổi ngầm qua resume.
Không resume giữa batch; epoch chưa hoàn tất sẽ chạy lại từ boundary đã lưu.

## Bộ nhớ và kiểm chứng

memory_snapshot ghi RAM trống, process RSS và Windows commit trống.
guard_memory giữ ít nhất 2 GiB RAM/3 GiB Windows commit, thử gc/empty_cache rồi kiểm tra lại.
Không hạ ngưỡng để ép chạy; guard không bảo đảm tuyệt đối tránh hệ điều hành hết bộ nhớ.
state_fingerprints đọc từng tensor để xác minh freeze, không clone toàn backbone.
Best/last chia sẻ bằng Git LFS; ngoại lệ giữ replay_reference của gói B tại lan_20 qua LFS, các replay/cache khác vẫn ignore. Chỉ nạp checkpoint tin cậy.
Audit Markdown ghi kết quả; changelog ghi sửa pipeline, không nhân bản hai vai trò vào log.
