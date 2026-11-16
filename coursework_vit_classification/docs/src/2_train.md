# 2_train.py — Tuning/final và resume

Source: [2_train.py](../../src/2_train.py). Chiến lược: [Training_Strategy](../design/Training_Strategy.md).

| 5W1H | Mô tả |
|---|---|
| What | Train một pilot hoặc một final của A–D |
| Why | Cùng loop, có val, best/last và cấu hình tái lập |
| Which | Một cấu hình A–D, phase tuning/final, AdamW, CrossEntropyLoss và checkpoint best/last |
| When | Smoke đã đủ, config được duyệt; không tự chạy cả 12 lượt |
| Where | outputs/runs/lan_N; resume giữ thư mục run hiện có |
| How | Train/val trực tiếp, AdamW, AMP tùy CUDA, checkpoint epoch |

## Trước khi chạy

PHASE phải tuning/final; TRAINING_APPROVED=True, DUPLICATE_POLICY='keep_reported'.
Đây là các lựa chọn cần thống nhất; code không tự chấp nhận duplicate policy thay nhóm.
Khai báo BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, EPOCHS và EXPERIMENT.
Tuning: CANDIDATE=LR1/LR2, SELECTED_FROM=None.
Final: CANDIDATE=None, SELECTED_FROM=lan_N pilot đã completed, cùng LR/protocol được chọn.
Final từ chối pilot có purpose=resource_probe; đây chỉ là lượt kiểm tra tài nguyên, không chọn LR.
Đổi optimizer/scheduler không nằm trong vòng tuning hiện tại; AdamW và LR cố định.

```powershell
.venv/Scripts/python.exe -u coursework_vit_classification/src/2_train.py
```

Mặc định bị chặn vì PHASE=smoke và cấu hình chưa được duyệt; smoke chạy 0_smoke.py riêng.

## Luồng

1. Kiểm tra config, seed, memory; tạo train/val loaders, ghi metadata và model architecture.
2. New run khởi tạo từ pretrained; final không tiếp tục weights pilot.
3. Mỗi epoch train trên logits/CE, cập nhật trainable parameters, ghi skipped AMP steps nếu có.
4. Val eval/inference_mode; loss tính theo số ảnh, không lấy trung bình batch không trọng số.
5. Best theo val accuracy, hòa thì loss thấp hơn; tie hoàn toàn giữ epoch sớm hơn.
6. Best/last đều full state; guard trước/sau ghi, history JSON và một dòng log/epoch.
7. Summary completed/failed/interrupted; không ghi checkpoint giả ở giữa batch khi bị ngắt.

## Resume

RESUME_RUN=lan_N; chọn epoch mới nhất trong best/last, có thể dùng best nếu last chưa kịp ghi.
Khôi phục history/best score, optimizer/scaler, RNG và train loader generator.
Đối chiếu phase, model, LR, seed, batch, preprocessing, split, epoch budget và các trường protocol.
Không đổi EPOCHS để kéo dài run ngầm; muốn đổi ngân sách phải thiết kế lượt mới.
Chưa có gradient accumulation, early stopping, scheduler hoặc tự chọn LR thắng giữa hai pilot.
Full run trên CIFAR-100 chưa thực hiện; kiểm tra tổng hợp CPU không đại diện chất lượng model.
