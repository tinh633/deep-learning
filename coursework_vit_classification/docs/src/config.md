# config.py — Cấu hình

Source: [config.py](../../src/config.py).

| 5W1H | Mô tả |
|---|---|
| What | Đường dẫn, data, A–D, smoke, tuning/final, evaluate và demo |
| Why | Một nguồn cấu hình cho mọi bước, không dùng ngầm tham số smoke cho train dài |
| Which | CIFAR-100, ConvNeXt-Base/ViT-B/16, chiến lược A–D và các giá trị cấu hình tương ứng |
| When | Khi import; không tự tải hoặc thực thi |
| Where | data/, outputs/, split, tên run và đường dẫn ảnh demo |
| How | Hằng số Python và dictionary tường minh |

## Các nhóm cấu hình

- Data: seed 42, input 224, ImageNet mean/std, split 45k/5k/10k.
- EXPERIMENTS: A/C freeze, B/D fine-tune; ConvNeXt-Base và ViT-B/16.
- Smoke: 20 train/4 val, batch 2, workers 0, AMP CUDA; LR 1e-3/1e-4 và WD=0 chỉ để kiểm tra.
- Train: PHASE=tuning/final, EXPERIMENT, BATCH_SIZE/LR/WD/EPOCHS phải được khai báo rõ.
- TRAINING_APPROVED=False và DUPLICATE_POLICY=None mặc định chặn train dài.
- Tuning cần CANDIDATE=LR1/LR2; final cần SELECTED_FROM là pilot đã hoàn tất và CANDIDATE=None.
- RESUME_RUN là tên lan_N; resume không tự đổi LR, epoch budget hoặc phase.
- RUN_PURPOSE=resource_probe đánh dấu lượt đo khả năng chạy, không dùng chọn LR cho final.
- Evaluate: EVAL_RUN, checkpoint best.pt/last.pt, split, MAX_EVAL_IMAGES tùy chọn.
- RUN_TEST=False, ALLOW_SMOKE_CHECKPOINT=False: chặn test ngầm và dùng model smoke như model cuối.
- Demo: DEMO_IMAGE là file ảnh local; không có ảnh mặc định giả định.

Giá trị None là quyết định chưa chốt, không phải file code chưa triển khai.
Scheduler hiện cố định None; chưa có grid search, accumulation hoặc early stopping.
Duplicate policy hiện chỉ hỗ trợ giữ split và công bố; không có code lọc ảnh tự động.
