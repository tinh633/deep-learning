# test_pipeline.py — Kiểm tra hồi quy pipeline trên CPU

Source: [tests/test_pipeline.py](../../tests/test_pipeline.py).
File thuộc tests/, không phải bước huấn luyện hoặc đánh giá model thật.

## 1. Mục đích và 5W1H

| Mục | Mô tả |
|---|---|
| What | Kiểm tra các script phối hợp đúng: smoke, train/resume, evaluate và demo |
| Why | Phát hiện lỗi code trước khi dùng pretrained model và dữ liệu thật tốn tài nguyên |
| Which | unittest/mock, SmallClassifier, tensor tổng hợp và hai kiến trúc torchvision trên meta |
| When | Khi thay đổi logic liên quan; không bắt buộc trước mỗi lần train |
| Where | tests/; dữ liệu và artifact tạm dưới data/pipeline_check_* |
| How | unittest, model nhỏ, tensor tổng hợp và mock thay các phụ thuộc nặng |

## 2. Dữ liệu và model thay thế

- SmallClassifier: global average pooling → Conv 1×1 → Linear 100 lớp.
- Width 1024/768 để kiểm tra head tương ứng; đây không phải ConvNeXt-Base hoặc ViT-B/16.
- Ảnh là tensor ngẫu nhiên 3×224×224 với seed cố định, nhãn tổng hợp 0/1.
- Các tên class_0…class_99 chỉ dùng kiểm tra cấu trúc đầu ra 100 lớp.
- CUDA bị mock thành không khả dụng để chạy CPU; không tải weights hoặc mở CIFAR dataset.
- Memory guard được mock để thử luồng xử lý, không cố làm máy thiếu bộ nhớ thật.

## 3. Ba nhóm kiểm tra

### test_real_architecture_contracts

Dựng ConvNeXt-Base và ViT-B16 thật trên meta device; không cấp phát weights, không tải pretrained.
Kiểm tra tổng/trainable parameters ở freeze/fine-tune, output 2×100 và đường dẫn head.
ConvNeXt: LayerNorm cuối freeze, stochastic depth đúng mode, không có BatchNorm.
Tên resnet50 bị từ chối; đây là kiểm tra cấu trúc, không phải numerical inference.

### test_smoke_and_post_replay_guard

1. Chạy 0_smoke.py với model/loader tổng hợp cho bốn cấu hình A–D.
2. Kiểm tra cả bốn có summary passed trong phạm vi tổng hợp.
3. Xác nhận ghi nhận guard cuối workflow và sau save/load/replay load.
4. Chạy thêm một lượt, giả lập thiếu commit tại điểm guard sau replay.
5. Pipeline phải dừng, summary failed và failed_check=resources.

Điểm giả lập dùng thứ tự gọi guard hiện tại; nếu tổ chức lại guard cần cập nhật test tương ứng.

### test_train_resume_evaluate_demo

1. TRAINING_APPROVED=False phải chặn 2_train.py.
2. Giả lập KeyboardInterrupt sau khi lưu last epoch 1; summary phải ghi interrupted.
3. Resume tới epoch 2; history không mất hoặc lặp epoch, trạng thái trở thành completed.
4. Từ chối resume khi đổi LR.
5. Chạy final mới từ cấu hình pilot; so toàn state_dict với nhánh resume để kiểm chứng tái lập.
6. Evaluate validation: số ảnh, 100 lớp và accuracy phải khớp predictions.
7. AUC của lớp không có mẫu phải là null, không tự tạo giá trị giả.
8. RUN_TEST=False hoặc checkpoint pilot phải chặn test trước khi gọi loader.
9. Demo ảnh tổng hợp phải ghi năm dự đoán Top-5 đúng cấu trúc.
10. Từ chối chọn pilot resource_probe hoặc kiến trúc ResNet cũ cho final ConvNeXt.

Các bước tuning/final ở đây chỉ là nhãn luồng kiểm tra; không phải pilot/final của coursework thật.

## 4. Cách chạy và đọc kết quả

Từ thư mục gốc workspace, dùng môi trường có requirements của coursework:

```powershell
.venv/Scripts/python.exe coursework_vit_classification/tests/test_pipeline.py
```

- Thành công: unittest in `Ran 3 tests`, `OK`, tiến trình trả mã 0.
- Thất bại: đọc FAIL/ERROR và traceback để tìm assertion không đạt.
- Trong log có thể xuất hiện traceback ngắt/memory guard do tình huống cố ý giả lập.
- Không chỉ thấy traceback rồi kết luận test lỗi; xem kết luận unittest và exit code cuối.
- Không chạy tuning/model GPU thật để xử lý một lỗi chỉ thuộc test tổng hợp.

## 5. Artifact và dọn dẹp

setUp tạo thư mục tạm riêng trong data/, chuyển output và config sang phạm vi test.
Mỗi config run có test_scope ghi rõ synthetic CPU, không dùng làm số liệu CIFAR-100.
tearDown khôi phục các mock/config, đóng log rồi xóa đúng thư mục tạm do test tạo.
Run thật trong outputs/runs/lan_N và dataset local không bị xóa hoặc ghi đè.

Lần kiểm chứng 16/11/2026 có [report](../../outputs/validation/source_pipeline_check.json)
và log local data/pipeline_test.log do lệnh kiểm chứng ngoài test lưu lại.
Chạy file test trực tiếp không tự cập nhật report này; không mặc định report cũ chứng minh code mới đã đạt.
Kiểm chứng cặp mới: [convnext_pipeline_check.json](../../outputs/validation/convnext_pipeline_check.json),
log local data/convnext_pipeline_test.log; phân biệt meta/synthetic với pretrained GPU thực.

## 6. Giới hạn

PASS chứng minh logic đã kiểm tra chạy đúng trong điều kiện tổng hợp đã định.
Không chứng minh accuracy, convergence, tốc độ, CUDA/AMP hoặc VRAM của ConvNeXt/ViT thật.
Không kiểm tra đầy đủ DataLoader worker Windows, tải weights hoặc chất lượng dữ liệu CIFAR-100.
Không thay smoke thật ở [0_smoke.md](0_smoke.md); D/GPU vẫn phải recheck khi đủ RAM/commit.
Ghi kết quả kiểm tra ở [Audit_Log](../Audit_Log.md), sửa code ở [CHANGELOG](../CHANGELOG.md).
