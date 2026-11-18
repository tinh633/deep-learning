# Tiến độ coursework

Cập nhật 18/11/2026: **Exp A hoàn tất 2 pilot và final 20/20 epoch; checkpoint đã kiểm chứng.**
Lịch sử: [progress_before_A_2026-11-18.md](progress_before_A_2026-11-18.md).

## Phạm vi đã chốt

- CIFAR-100, ConvNeXt-Base A/B và ViT-B16 C/D, pretrained ImageNet-1K V1; không kết hợp deepfake.
- A chạy máy hiện tại; ba thành viên khác nhận B/C/D, mỗi người 2 pilot + 1 final.
- A: 2 pilot × 5 epoch (LR 1e-3, 3e-4), chọn bằng validation; final 20 epoch khởi tạo mới.
- Batch 16, AdamW, WD=0, không scheduler, seed 42, AMP, workers 0; không mở test.
- Giữ split 45k/5k/10k và công bố 12 nhóm ảnh trùng xuyên split; không đổi dữ liệu giữa các lượt.
- Quy trình: [Training_Strategy](design/Training_Strategy.md), [Work_Allocation](design/Work_Allocation.md).

## Đã hoàn thành

- Code data/model/smoke/train/resume/evaluate/demo; EDA HTML; notebook theo dõi log epoch.
- 3 kiểm thử cấu trúc meta + integration CPU đạt trước khi đổi config train; có hash theo phiên bản kiểm tra.
- Xóa riêng lan_9 ResNet cũ theo yêu cầu; bằng chứng cũ ở Git 6bf7dcc, không phải ConvNeXt.
- Lan_9 mới: smoke lỗi tải mạng, 0 batch. Tải weights chính thức, checksum 6075fbad thành công.
- Lan_10: smoke dừng bảo vệ sau 1 batch vì Windows commit 2942,83 MiB (<3072).
- Được phép đóng ứng dụng khác: đã đóng Zalo/Discord; Chrome không đủ quyền, giữ VS Code/hệ thống.
- Lan_11: smoke A batch 16, 32 train/16 val, PASS đủ 12 checklist, save/load/replay/final guard.
- Peak train/workflow 511,22/607,22 MiB; commit cuối 3716,92 MiB. Không phải chất lượng train đầy đủ.

## Kết quả Exp A và trạng thái hiện tại

- Lan_12 LR1=1e-3 và lan_13 LR2=3e-4 đều đủ 5 epoch; best val 80,64% / 80,78%, chọn LR2.
- Lan_14 final đủ 20 epoch; best val 81,46% epoch 9, last val 80,94% / train 92,77%.
- Guard dừng giữa epoch 20 vì commit 3061,45 MiB; đã giải phóng bộ nhớ và resume từ last epoch 19.
- Lần dừng được lưu interruptions.json; history đủ 1–20, optimizer last đủ 56.242 update, tính cả 18 AMP skip.
- [Kiểm chứng thực](../outputs/validation/experiment_A_check.json): best/last, frozen tensors và RNG/history đạt.
- [experiment_A.json](../outputs/experiment_A.json) trạng thái completed; process train/monitor đã thoát, GPU về 0 MiB.
- [Kết quả A](A_Results.md); Training_Monitor.ipynb đọc được lan_12/13/14. Không mở test.
- Config giữ final/resume lan_14 nhưng TRAINING_APPROVED=False để tránh chạy lặp; EVAL_RUN=lan_14, validation.
- Manifest 12 checkpoint, 3,32 GiB; kết quả A đã push cousework tại 68641b3, đủ 7 LFS mới. B/C/D chờ kết quả nhóm.

## Bàn giao khi bị ngắt

1. Kiểm tra experiment_A.json, PID thật, summary/history trước khi chạy mới; không chạy song song.
2. Run bị ngắt có last.pt: dùng RESUME_RUN cùng cấu hình/budget; không đổi LR hoặc train lại từ đầu.
3. Không chạy lại controller từ đầu khi đã có pilot; tiếp tục đúng bước còn thiếu.
4. Guard thiếu bộ nhớ: dừng, không hạ ngưỡng hoặc tự thay batch đã chốt.
5. A đã đủ budget và đã xuất bản, không chạy lại controller; tiếp theo tích hợp B/C/D rồi test sau khóa protocol.

Audit: [Audit_Log](Audit_Log.md). Thay đổi pipeline: [CHANGELOG](CHANGELOG.md).
