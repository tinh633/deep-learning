# AI Usage Step — Practice 2

## 2026-10-29 — Làm gọn src

- Người dùng chấp nhận tạm và yêu cầu push rồi train thử. Kiểm tra phát hiện outputs/runs trống, không còn checkpoint để resume; hỏi lựa chọn train mới/smoke. Compile/diff check qua; commit/push được cho phép.

- Sau yêu cầu dễ nắm khi vấn đáp: sửa riêng config/train, dùng hằng số có tên, tham số model truyền bằng keyword, config.get_settings() tạo metadata tương thích cũ; vòng train/validation giữ trực tiếp và chú thích tiếng Việt. Chờ review trước phần tiếp theo.
- Đã kiểm tra config khớp checkpoint lan_2; smoke CPU train liên tục/ngắt-resume khớp trọng số, best/last/history qua; compile/diff check qua. Không train thật, chưa commit; đã tổng hợp Full.

- Người dùng tiếp tục phản hồi code chưa dễ đọc: viết lại train, đưa restore/save checkpoint sang hai hàm trong logger, giữ train/validation trực tiếp; không thêm lớp/framework/file.
- Smoke CPU lần mới qua: fresh run, best/last, ngắt/resume khớp trọng số liên tục, cấu hình resume đổi bị từ chối. Không khởi chạy run thật; đã tổng hợp Full.

- Người dùng yêu cầu giải thích SHA-256 và chỉnh code gọn, rõ luồng. Giữ 7 module và checkpoint/report hiện có.
- Gom JSON, đóng log, hash vào logger; bỏ registry không còn viewer sử dụng; chia bước train/EDA/config, tách vẽ ảnh sai trong evaluate.
- Evaluate dùng DataLoader không augmentation/shuffle và đọc được checkpoint có NumPy RNG.
- CPU smoke: train liên tục và ngắt/resume khớp trọng số; evaluate khớp sklearn/ID, PNG, test tắt, từ chối split sai. Không train lượt chính thức.
- Đã tổng hợp giai đoạn này vào Full; thay đổi chưa commit/push.

Ghi theo từng giai đoạn: yêu cầu, hành động, file thay đổi, kiểm tra/kết quả và việc còn lại.
Cuối giai đoạn hoặc khi bàn giao, tổng hợp nội dung đã xác minh vào
[AI_Usage_Full.md](docs/AI_Usage_Full.md) và đánh dấu đã chuyển; không xóa nội dung chưa tổng hợp.
Full lưu toàn bộ lịch sử; Step phục vụ theo dõi và review các giai đoạn đang làm.
Trạng thái vận hành hiện tại: [progress.md](progress.md).
Các mục dưới đây đã được tổng hợp vào Full trong lần đổi tên này, giữ lại để tiếp tục review.

## 2026-10-28 — Bắt đầu train chính thức ResNet18 A

- Người dùng yêu cầu bắt đầu tiến trình train; phạm vi là một lượt baseline A, không tự chạy 18 lượt.
- Codex đọc tiến độ, kiểm tra config, CUDA/dataset/split và không có lượt train khác đang chạy.
- Khởi chạy `.venv/Scripts/python.exe -u src/2_train.py` từ gốc workspace bằng đường dẫn project đầy đủ.
- Run mới `outputs/runs/lan_1`: ResNet18 pretrained ImageNet V1, H1, freeze backbone,
  Adam LR 0,001, batch 64, worker 2, 20 epoch; giữ RUN_TEST=False.
- Log/config thực tế xác nhận device=cuda, train 45.000/validation 5.000 và 5.130 tham số trainable.
- NVIDIA-SMI kiểm tra sau khởi động: GPU 58%, bộ nhớ 786 MiB ở thời điểm quan sát.
- Tiến trình tiếp tục chạy; chưa xác nhận epoch/run hoàn tất, chưa evaluate hay chọn model.
- Công cụ: đọc file, kiểm tra tiến trình/GPU và exec session 55483; không chỉnh source/config.
- Cập nhật progress; lưu trữ AI usage cũ vì đã chạm giới hạn 150 dòng, không xóa lịch sử.
- Chưa commit/push artifact hoặc tài liệu mới.
- Cập nhật: lan_1 hoàn tất 6 epoch, lỗi giữa epoch 7 với shared memory mapping/Windows 1455.
  Tiến trình exit 1; best.pt xác minh đọc được epoch 5, val 88,20%, loss 0,339812.
- Ghi lại thất bại và chẩn đoán từ traceback; chưa đổi worker/pagefile hoặc chạy lại.
- Theo yêu cầu tiếp theo, đổi worker 2 → 0 và khởi động lại GPU ResNet18 A 20 epoch tại lan_2.
  Không resume, không đổi LR/head/batch; giữ lan_1. Config/log lan_2 xác nhận CUDA và worker 0.

## Hoàn tất ResNet18 A, chuẩn bị VGG16 A

- Người dùng yêu cầu tiếp tục, xóa lượt lỗi lan_1 và đổi tên lan_2 thành lan_1.
- Trước khi xóa/đổi tên, đối chiếu 20 history rows, Finished và checkpoint ResNet18 A đọc được;
  best epoch 18, val 88,54%, loss 0,337606. Xác minh đúng đường dẫn workspace trước thao tác.
- Giữ nguyên log lịch sử và thêm ghi nhận đổi tên. Chuyển config sang VGG16 A, worker 0.
- Khởi chạy evaluate train/validation ResNet18 theo flow trong lúc tải weights VGG16; test tắt.
- Tải weights từ nguồn PyTorch chính thức; dùng HTTP ranges do tải đơn chậm, kiểm tra hash trước cache.
- Chưa commit/push. VGG16 chỉ được khởi động sau khi cache và GPU preflight được xác minh.
- Evaluate ResNet18 xong: train 89,6533%, val 88,54%, val macro-F1 0,885577; counts/IDs/lỗi khớp,
  không có test report. Weights VGG16 tải chậm, chuyển sang CDN download-r2.pytorch.org có kiểm tra hash.
- Giữ một download session 57711; session 66286 chờ cache, chạy GPU preflight rồi tự khởi động train.
  Các phép thử tải khác đã dừng để tránh lặp; pipeline chưa vào epoch VGG16 khi weights chưa đầy đủ.

## 2026-10-28 — Thống nhất nhật ký Full/Step

- Theo người dùng, đổi log trong docs thành AI_Usage_Full.md, log ngoài thành AI_Usage_Step.md.
- Đã tổng hợp các giai đoạn hiện có vào Full, giữ nội dung lịch sử và cập nhật liên kết/quy ước.
- Bước tài liệu này đã tổng hợp vào Full; không dừng pipeline VGG16 hoặc commit/push.

## 2026-10-28 — Không lưu data trên Git

- Người dùng yêu cầu không commit data và gỡ data đã push trước đó.
- Bỏ ngoại lệ CIFAR10_web trong .gitignore, gỡ 6 RGB gzip khỏi Git index bằng --cached, giữ bản local.
- Chỉ commit/push việc loại data và .gitignore; các thay đổi source/log/tài liệu khác giữ riêng.
- Đồng bộ yêu cầu vào README/progress và Full; không rewrite lịch sử hoặc xóa dataset trên máy.
- Kết quả: commit 4bdb86b đã push Thái, 6 file data không còn trong cây Git hiện tại; bản local giữ nguyên.

## 2026-10-28 — Làm rõ weights trong kế hoạch

- Theo yêu cầu, ghi rõ ImageNet-1K (1.000 lớp), IMAGENET1K_V1 cho cả ba model và nguồn torchvision chính thức.
- Bổ sung weights vào bảng kiến trúc/checklist, phân biệt 1K/V1 và head CIFAR-10; sửa trạng thái train cũ trong kế hoạch.
- Đã tổng hợp bước tài liệu vào Full; kiểm tra enum/metadata local, giới hạn dòng và liên kết. Không đổi model/source hoặc commit/push.

## 2026-10-28 — VGG16 A bắt đầu lan_2

- Weights VGG16 đã tải xong, hash 397923af... khớp; pipeline cũ lỗi quyền đọc cache trong sandbox.
- Khởi chạy lại với quyền phù hợp theo yêu cầu train trước đó; GPU preflight batch64 qua, peak 2445,9 MiB.
- Log xác nhận VGG16 A pretrained CUDA, worker 0, 20 epoch tại lan_2 lúc 16:00:25; session 95990.
- Đã tổng hợp vào Full; chưa xác nhận epoch/run hoàn tất, chưa commit/push.

### Resume checkpoint — 2026-10-28

- Theo yêu cầu người dùng: thêm last.pt và resume cùng run; best dùng validation, last dùng tiếp tục train.
- Checkpoint mới lưu Adam và RNG; checkpoint lan_2 cũ epoch 2 thiếu Adam nên reset có cảnh báo. Không sửa lan_1.
- Smoke CPU tạm kiểm tra save/resume/RNG/legacy qua; chưa tuyên bố tái lập chính xác GPU bit-for-bit.

- Bỏ logger.info Live batch theo yêu cầu; kiểm tra logger không ghi dòng batch, TensorBoard vẫn nhận scalar. Không restart train đang chạy tại bước này.

- Sau đó người dùng yêu cầu dừng/xóa lan_2 và train mới: đã dừng PID 12232/4736, xóa đúng outputs/runs/lan_2; RESUME_RUN=None, giữ VGG16 A và lan_1.

- Theo yêu cầu: HTML nhúng TensorBoard trực tiếp, có lệnh khởi động/tải lại/tab riêng; bỏ dashboard riêng và external JS.
- Sửa ký tự tiếng Việt bị ghi thành dấu hỏi; thêm Live Server ignore outputs/data để tránh reload theo artifact train. Notebook chỉ là đề xuất, chưa tạo.

- 2026-10-28: user stopped training for later. Removed per-batch logging/TensorBoard updates, live.jsonl and progress.json. Epoch-only history/log/TensorBoard retained. Verified best.pt/last.pt epoch 1 with Adam state; RESUME_RUN=lan_2, next epoch=2. No restart.

- 2026-10-29: user authorized Git commit/push; checked branch Thai, excluded local weights/events/data-derived images, repaired remaining damaged handoff/resume documentation.
