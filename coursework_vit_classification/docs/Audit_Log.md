# Audit log — Checklist smoke run

Checklist ngày 10/11/2026 được giữ làm kế hoạch; kết quả thực ngày 14/11 nằm ở phần cuối.
Chiến lược: [Training_Strategy.md](design/Training_Strategy.md).
Thay đổi code/pipeline ghi ở [CHANGELOG.md](CHANGELOG.md); việc tiếp theo ghi ở [progress.md](progress.md).

## Quy tắc ghi nhận

- Chỉ ghi cấu hình, kiểm tra đã chạy, bằng chứng và kết luận smoke; không kể quá trình sửa code.
- Trạng thái: PASS, FAIL hoặc NOT_RUN; không xem chạy không lỗi là mọi kiểm tra đều đạt.
- Mỗi lượt thực có ngày, run ID/thư mục, phiên bản source, model/A–D, seed, split và cấu hình.
- Ghi máy/GPU, precision, batch, số batch/epoch thực chạy và đường dẫn log/checkpoint nguồn.
- Khi kiểm tra thất bại, ghi FAIL và liên kết thay đổi/kiểm chứng lại trong changelog, không xóa kết quả cũ.
- “Ổn định” chỉ trong phạm vi số batch đo: có/không NaN, Inf, OOM, biến động thời gian/VRAM.
- Smoke ngắn không chứng minh hội tụ hoặc khả năng tổng quát hóa; không dùng test để smoke/tuning.
- Audit Markdown là bản tóm tắt; bằng chứng chi tiết đọc từ artifact thật, không bắt buộc thêm audit.py/JSON riêng.

## Checklist trước smoke — Chưa thực thi

A: ResNet freeze; B: ResNet fine-tune; C: ViT freeze; D: ViT fine-tune.
Các cột chỉ được cập nhật khi có run và bằng chứng tương ứng.

| Kiểm tra | A | B | C | D |
|---|---|---|---|---|
| Nguồn weights, class order, split/config đúng | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| Input/output shape và logits 100 lớp | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| Loss hữu hạn, không NaN/Inf | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| Trainable parameters đúng chiến lược | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| Head/backbone gradient đúng | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| Optimizer chứa đúng tham số được mở | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| Tham số/buffers thay đổi đúng policy freeze/BN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| Val eval/no-gradient, không thay state | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| Save/load tái tạo logits trong tolerance đã ghi | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| Resume epoch/optimizer/RNG đúng cấu hình | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| Thời gian, peak VRAM và số batch đo được ghi | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| Không NaN/Inf/OOM trong số batch đã công bố | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |

## Mẫu ghi một lượt thực tế

Khi chạy, thêm mục theo ngày/run và liên kết thư mục bằng chứng thực sự tồn tại:

1. Nhận diện: run ID, model/chiến lược, source revision/hash, weights và split hash.
2. Điều kiện: máy, GPU, batch/precision, seed và phạm vi thực thi.
3. Kết quả: từng check, trạng thái, số đo/tolerance và bằng chứng; ghi rõ check chưa chạy.
4. Ổn định: số batch quan sát, NaN/Inf/OOM, thống kê thời gian/VRAM nếu đã đo.
5. Kết luận: có đủ điều kiện tuning chưa, còn giới hạn nào; liên kết recheck nếu cần.

Ở thời điểm tạo checklist 10/11 chưa có run ID hoặc kết luận PASS; không áp dụng nhận định đó cho lượt chạy sau.
Khi gần 150 dòng, chuyển lượt cũ sang docs/archive/audit_<giai_doan>.md và liên kết, không xóa lịch sử.

## 14/11/2026 — Kết quả smoke thực và khôi phục phiên

Máy: RTX3050 Laptop 4 GB, RAM 16 GB; torch 2.11+cu126, torchvision 0.26, AMP float16.
Batch 2, workers 0, seed 42; cùng 20 train/4 val từ split hiện có, không mở test.
LR smoke freeze=1e-3, fine-tune=1e-4; AdamW WD=0, không scheduler. Chưa tuning các giá trị này.
Mỗi lượt hoàn tất gồm 10 batch train, 2 batch val và hai nhánh so sánh một bước resume.
Mini-epoch không đại diện một epoch đầy đủ CIFAR-100; xem config từng run để đối chiếu source hashes.

| Cấu hình | Bằng chứng | Kết quả | Batch train hoàn tất | Peak train / workflow MiB |
|---|---|---|---:|---:|
| A ResNet freeze | [lan_4](../outputs/runs/lan_4/summary.json) | PASS | 10 | 121,85 / 139,54 |
| B ResNet fine-tune | [lan_8](../outputs/runs/lan_8/summary.json) | PASS | 10 | 460,17 / 460,17 |
| C ViT freeze | [lan_5](../outputs/runs/lan_5/summary.json) | PASS | 10 | 365,39 / 367,39 |
| D ViT fine-tune | [lan_7](../outputs/runs/lan_7/summary.json) | FAIL — memory guard | 1 | Chưa ghi đủ |

| Kiểm tra đã lưu | A / lan_4 | B / lan_8 | C / lan_5 | D / lan_7 |
|---|---|---|---|---|
| Nguồn/split/config | PASS | PASS | PASS | PASS |
| Trainable parameters | PASS | PASS | PASS | PASS |
| Optimizer parameter groups | PASS | PASS | PASS | PASS |
| Shape / finite loss qua đủ 10 batch | PASS | PASS | PASS | NOT_RUN |
| Gradient qua đủ 10 batch | PASS | PASS | PASS | NOT_RUN |
| Backbone/head/buffers cập nhật đúng policy | PASS | PASS | PASS | NOT_RUN |
| Validation eval/no-gradient/buffers | PASS | PASS | PASS | NOT_RUN |
| Save/load logits trong tolerance | PASS | PASS | PASS | NOT_RUN |
| Resume mini-boundary + bước tiếp theo | PASS | PASS | PASS | NOT_RUN |
| Điều kiện bộ nhớ hệ thống | PASS | PASS | PASS | FAIL |
| Hoàn tất phạm vi ổn định đã công bố | PASS | PASS | PASS | NOT_RUN |

A/B/C: checkpoint reload và state sau replay khớp rtol=1e-5, atol=1e-6; chênh loss <1e-5.
Không NaN/Inf/OOM ở phạm vi hoàn tất của A/B/C; không kết luận về hội tụ hoặc train dài.
D dừng sau một update vì Windows commit còn 2553,90 MiB <3072 MiB, RAM trống 3934,05 MiB.
Guard vẫn giữ ngưỡng 2 GiB RAM/3 GiB commit sau thu hồi cache; chưa đủ điều kiện tiếp tục D.
NOT_RUN ở D nghĩa kiểm tra đầy đủ chưa hoàn tất/lưu PASS, không suy đoán từ một bước đã chạy.

Lịch sử giữ nguyên: [lan_1 A](../outputs/runs/lan_1/summary.json), [lan_2 B](../outputs/runs/lan_2/summary.json)
đã PASS ở source cũ; [lan_3 C](../outputs/runs/lan_3/summary.json) bị ngắt khi loading, không có kết quả.
[lan_6 D](../outputs/runs/lan_6/run.log) dừng do memory guard, không tính là một lượt đạt.
Review sau chạy: B/lan_8 kết thúc còn 2952,80 MiB Windows commit, thấp hơn ngưỡng 3072 MiB.
PASS trong summary chỉ phản ánh các check đã cài; guard chưa bao phủ cuối save/load/replay.
Thời gian batch có warm-up CUDA và kiểm tra gradient, không dùng để xếp hạng tốc độ model.
Kết luận: A/B/C hoàn tất smoke hiện có; D chưa đạt. Chưa xác nhận bộ nhớ ổn định toàn workflow.
Trước tuning/final cần bổ sung guard cuối workflow và đủ RAM/commit để recheck, nhất là D.
