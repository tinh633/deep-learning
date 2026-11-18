# Exp A — ConvNeXt-Base freeze

Hoàn tất ngày 18/11/2026. Đây là kết quả validation; **chưa đánh giá tập test**.
Protocol: [Training_Strategy mục 9](design/Training_Strategy.md).
Nhật ký phiên: [experiment_A.json](../outputs/experiment_A.json).

## 1. Cấu hình

- ConvNeXt-Base, pretrained ImageNet-1K V1; freeze cả backbone và LayerNorm cuối.
- Chỉ train classifier[2] Linear(1024,100): 102.500/87.668.964 tham số.
- CIFAR-100: 45.000 train/5.000 validation, seed 42, resize 224 và ImageNet normalization.
- Batch 16, workers 0, AdamW, WD=0, LR cố định, không scheduler hoặc early stopping.
- Train AMP; validation FP32. GPU RTX 3050 Laptop 4 GB.
- Giữ nguyên split, công bố 12 nhóm ảnh trùng xuyên split; không thay dữ liệu sau khi xem điểm.

## 2. Hai pilot LR

| Run | LR | Epoch | Best epoch | Val accuracy | Val loss | Train+val đã ghi |
|---|---|---|---|---|---|---|
| lan_12 — LR1 | 1e-3 | 5 | 5 | 80,64% | 0,6950 | 28,28 phút |
| lan_13 — LR2 | 3e-4 | 5 | 5 | 80,78% | 0,6335 | 28,17 phút |

Chọn LR2 vì validation accuracy cao hơn 0,14 điểm phần trăm, theo tiêu chí công bố trước.
Chênh lệch nhỏ và chỉ một seed; không kết luận LR2 tốt hơn trong mọi điều kiện.
Không dùng tập test để lựa chọn, không kéo dài pilot sau khi xem kết quả.

## 3. Final

Final [lan_14](../outputs/runs/lan_14/config.json) khởi tạo mới từ pretrained và head mới cùng seed.
Chạy đủ 20 epoch, LR=3e-4; không warm-start từ pilot.

| Checkpoint | Epoch | Train accuracy | Val accuracy | Val loss |
|---|---|---|---|---|
| best.pt | 9 | 88,57% | **81,46%** | **0,6138** |
| last.pt | 20 | 92,77% | 80,94% | 0,6474 |

Train tiếp tục tăng trong khi validation không vượt epoch 9; ưu tiên best cho đánh giá cuối.
Gap train/val ở epoch 20 khoảng 11,83 điểm phần trăm; không coi accuracy train là khả năng tổng quát hóa.
Train+val của 20 epoch đã lưu: 112,08 phút, chưa tính tải model/checkpoint IO hoặc phần epoch bị bỏ khi dừng.

## 4. Dừng bảo vệ và resume

- Đầu phiên thiếu Windows commit; đã đóng ứng dụng không thiết yếu theo phạm vi được cho phép.
- Smoke lan_11 batch16 PASS đủ checklist trước pilot; xem [Audit_Log](Audit_Log.md).
- Giữa epoch 20, guard dừng ở 3061,45 MiB commit trống, dưới ngưỡng 3072 MiB.
- Checkpoint epoch 19 còn nguyên; đóng ứng dụng phát sinh, đo lại commit khoảng 6,7 GiB rồi resume cùng run.
- Log xác nhận resume last.pt bắt đầu epoch 20; optimizer/scaler/RNG và loader generator được khôi phục.
- Chỉ epoch 20 chạy lại; history cuối đủ 1–20, không lặp hoặc mất các epoch đã lưu.
- Giữ [interruptions.json](../outputs/runs/lan_14/interruptions.json) và traceback trong run.log, không xóa dấu vết lỗi.
- GPU samples toàn phiên: tối đa 876 MiB và 78°C; là mẫu định kỳ, không phải peak allocator.

## 5. Kiểm chứng và bàn giao

[experiment_A_check.json](../outputs/validation/experiment_A_check.json) ghi kết quả kiểm tra thực:

- Cả ba run completed; history, config, rule chọn best và epoch checkpoint khớp.
- Mọi tensor ngoài Linear mới bằng weights ImageNet gốc ở cả best/last của ba run, gồm LayerNorm cuối.
- Hai pilot không AMP skip; final có 18 skip do GradScaler, loss đã lưu đều hữu hạn.
- Last optimizer đủ 56.242 update = 20×2813 batch −18 skip; không mất trạng thái khi resume.
- Checkpoint có optimizer, scaler, epoch, history và các RNG; SHA-256/dung lượng trong manifest.
- Năm epoch đầu final khớp metric của LR2 pilot cùng seed, phù hợp khởi tạo lại theo cùng protocol.
- Không có artifact đánh giá test trong các run A.

File dùng tiếp: [best.pt](../outputs/runs/lan_14/best.pt), [last.pt](../outputs/runs/lan_14/last.pt),
[history](../outputs/runs/lan_14/history.json), [run.log](../outputs/runs/lan_14/run.log).
Mở Training_Monitor.ipynb, lọc lan_12/13/14 để xem pilot/final; không trộn smoke vào đường epoch.
Config giữ A final/resume và EVAL_RUN=lan_14, RUN_TEST=False; chặn train lặp bằng TRAINING_APPROVED=False.
Kết quả/checkpoint đã push cousework ngày 18/11 tại commit 68641b3, đủ 7 LFS mới; B/C/D chờ kết quả nhóm.
