# 0_smoke.py — Kiểm tra kỹ thuật

Source: [0_smoke.py](../../src/0_smoke.py). Kết quả thực: [Audit_Log](../Audit_Log.md).

| 5W1H | Mô tả |
|---|---|
| What | Smoke A–D với miniature epoch và checkpoint replay |
| Why | Phát hiện lỗi pipeline/tài nguyên trước các lượt train dài |
| Which | Bốn cấu hình A–D, AdamW/AMP, mẫu 20 train/4 val và checklist trong summary/audit |
| When | Sau EDA/model; PHASE phải smoke, workers phải 0 |
| Where | Mỗi chiến lược một outputs/runs/lan_N |
| How | Loop train/val trực tiếp, AdamW/AMP và các assert có bằng chứng |

```powershell
.venv/Scripts/python.exe -u coursework_vit_classification/src/0_smoke.py
```

Trước đây smoke nằm trong 2_train.py; tách riêng để loop train chính thức dễ đọc.
Mặc định 20 train/4 val, batch 2: 10 batch train, 2 batch val; không dùng official test.
Kiểm tra source/split, trainable, optimizer, shape, finite loss/gradient, buffers và state update.
Save/load logits rồi replay một bước sau mini-boundary; so loss và toàn model state trong tolerance.
Guard sau save, load, replay load/step, reference save và cuối workflow; không chỉ trong train loop.

## Artifact và giới hạn

- config.json, model.txt, run.log, history.json, summary.json.
- last.pt và replay_reference.pt chỉ kiểm chứng kỹ thuật, không là model final.
- Source hashes và scope trong config phân biệt các phiên bản/lượt chạy.
- PASS/FAIL/NOT_RUN ở summary được đối chiếu khi ghi audit, không tự sửa các report lịch sử.
- LR smoke không phải LR đã tuning; mini-loss không dùng xếp hạng kiến trúc.
- Kết quả A/B/C trước đây thuộc source cũ; việc sửa guard không tự làm D đạt hoặc tái xác nhận GPU.
- A/B mới dùng ConvNeXt-Base: chỉ classifier[2] được đổi khi freeze; LayerNorm cuối phải giữ bất biến.
- Code mới cần recheck thực khi RAM/commit đủ; không chạy D liên tục để ép PASS.
