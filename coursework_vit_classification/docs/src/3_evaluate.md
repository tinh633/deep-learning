# 3_evaluate.py — Đánh giá checkpoint

Source: [3_evaluate.py](../../src/3_evaluate.py).

| 5W1H | Mô tả |
|---|---|
| What | Tính metric/predictions và phân tích theo lớp trên một split |
| Why | Kết quả có checkpoint nguồn và tái tính được |
| Which | best.pt/last.pt của run được chọn, split train/validation/test và metric sklearn |
| When | Validation để phân tích; test chỉ sau khóa cấu hình cuối |
| Where | Run nguồn/evaluation/<split>_<checkpoint>_N, không ghi đè kết quả cũ |
| How | Model từ checkpoint, transform xác định, FP32 inference_mode và sklearn |

## Cấu hình và cách chạy

Đặt EVAL_RUN, EVAL_CHECKPOINT=best.pt/last.pt, EVAL_SPLIT=train/validation/test.
Test cần RUN_TEST=True và checkpoint phase=final; không test pilot/smoke.
Smoke bị từ chối mặc định; ALLOW_SMOKE_CHECKPOINT chỉ dành kiểm tra kỹ thuật có chủ đích.
MAX_EVAL_IMAGES giới hạn subset để kiểm tra; report luôn ghi n và giới hạn đó.

```powershell
.venv/Scripts/python.exe -u coursework_vit_classification/src/3_evaluate.py
```

## Đối chiếu và output

- Từ chối preprocessing thiếu/khác, split hash khác hoặc class order khác checkpoint.
- Giữ head 100 lớp, không tải pretrained lần nữa; đọc checkpoint local đáng tin.
- metrics.json: loss, accuracy/correct, Top-5, macro/weighted F1 và confusion 100×100 chuẩn hóa theo hàng.
- Mỗi lớp có TP/TN/FP/FN, precision/recall/F1, TPR/FPR/specificity, OvR ROC-AUC/AP.
- Lưu ROC fpr/tpr theo lớp nếu có cả positive và negative.
- AUC không xác định khi thiếu một phía; AP thiếu positive là null, không thay bằng số 0 giả.
- Macro AUC/AP chỉ trên các lớp xác định, ghi auc_classes/ap_classes; macro-F1 trên 100 nhãn.
- predictions.json giữ ID nguồn, true/pred và vector 100 xác suất; không chứa pixel ảnh.
- evaluate.log ghi tóm tắt; checkpoint SHA/epoch/split/preprocessing nằm trong report.

evaluation_seconds gồm loader/transform/transfer/forward/guard, không phải benchmark forward-only.
HTML trình bày đã đọc history A/B/C; chưa có notebook báo cáo hoàn chỉnh hoặc giao diện phân tích predictions theo ảnh.
Không dùng test report để thay đổi LR, checkpoint hoặc số epoch của thí nghiệm đang so sánh.
