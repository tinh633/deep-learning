# Source workflow — 5W1H theo từng file

Cập nhật: 14/11/2026. Mô tả code đang có; tuning/final, evaluate và demo chưa triển khai.
Thiết kế: [Implementation_Plan](design/Implementation_Plan.md). Kiểm chứng: [Audit_Log](Audit_Log.md).

## 1. config.py — Cấu hình dùng chung

Source: [config.py](../src/config.py).

| Mục | Mô tả |
|---|---|
| What — Là gì? | Đường dẫn, split, input/normalization, ma trận A–D và thông số smoke |
| Why — Vì sao? | Tránh khai báo khác nhau giữa data, model và script chạy |
| Who — Ai dùng? | Các module/script trong src; thành viên chỉnh cấu hình trước chạy |
| When — Khi nào? | Khi import, chỉ khai báo giá trị; không tải hoặc train |
| Where — Ở đâu? | ROOT project, data/, outputs/ và đường dẫn split |
| How — Thế nào? | pathlib và hằng số/dictionary tường minh; PHASE hiện chỉ smoke |

LR freeze 1e-3/fine-tune 1e-4 và WD=0 hiện phục vụ smoke, không phải kết quả tuning.
Batch 2, workers 0, AMP khi CUDA; 20 ảnh train và 4 val là mẫu kỹ thuật.

## 2. data.py — Dataset, split và DataLoader

Source: [data.py](../src/data.py).

| Mục | Mô tả |
|---|---|
| What | Nạp CIFAR-100, lưu/đọc split và tạo train/val loaders |
| Why | Dùng cùng IDs và preprocessing nhất quán giữa các cấu hình |
| Who | 1_eda.py và 2_train.py; phần evaluate sau này sẽ tái dùng |
| When | Khi gọi load_datasets, load_split hoặc build_loaders; không có side effect lúc import |
| Where | CIFAR-100 local trong data/, split ở outputs/cifar100_split.json |
| How | torchvision, stratified split seed 42, Subset và DataLoader |

load_datasets trả official train/test cho EDA. build_loaders chỉ nạp official train,
tách train/val theo IDs, không truy cập official test trong smoke.
Train resize/flip/tensor/normalize; val resize/tensor/normalize cố định.
Validation dùng bản sao nhẹ, chia sẻ mảng pixel gốc để tránh nhân đôi dataset trong RAM.

## 3. model.py — Backbone, head và chế độ train

Source: [model.py](../src/model.py). Kiến trúc: [ResNet50](design/ResNet50.md), [ViT](design/ViT_B16.md).

| Mục | Mô tả |
|---|---|
| What | Tạo ResNet-50 hoặc ViT-B/16 với head Linear 100 lớp |
| Why | So sánh đúng hai model và hai mức cập nhật backbone |
| Who | 2_train.py gọi build_model/get_head/set_train_mode; evaluate/demo dùng sau |
| When | Tạo model trước optimizer; đặt mode mỗi khi bước vào train |
| Where | Pretrained ImageNet1K_V1, cache ở data/torch_cache/checkpoints |
| How | torchvision, kiểm tra checksum, meta initialization + mmap/assign khi nạp pretrained |

Meta/mmap giảm bản sao RAM khi nạp weights, không thay kiến trúc hoặc số tham số.
Freeze: chỉ head requires_grad, backbone eval. Fine-tune: toàn bộ trainable và model.train().
pretrained=False tạo kiến trúc không tải weights; chưa có pipeline evaluate tự nạp checkpoint.

## 4. logger.py — Logging, bộ nhớ và checkpoint

Source: [logger.py](../src/logger.py).

| Mục | Mô tả |
|---|---|
| What | Tạo run, ghi config/log/JSON, kiểm tra RAM và save/load trạng thái |
| Why | Giữ xử lý artifact tách khỏi loop và có bằng chứng kiểm chứng |
| Who | 2_train.py; các bước khác chỉ tái dùng khi có nhu cầu |
| When | Đầu run, trước bước nặng, tại boundary checkpoint và khi resume |
| Where | outputs/runs/lan_N; run.log, config.json, summary/history và checkpoint local |
| How | logging, ghi tạm rồi replace, torch.save/load mmap, trạng thái RNG và loader generator |

Checkpoint gồm model/optimizer/scaler, epoch, settings và Python/NumPy/Torch/CUDA RNG.
Khi restore, giải phóng gradient/optimizer state cũ trước khi nạp để giảm peak bộ nhớ.
Guard dừng nếu RAM trống dưới 2 GiB hoặc Windows commit còn dưới 3 GiB; không bảo đảm tuyệt đối tránh OOM.
Khi thiếu bộ nhớ, guard thử thu gom tham chiếu rác và giải phóng CUDA cache rồi kiểm tra lại, không hạ ngưỡng.
Fingerprint từng tensor phục vụ kiểm tra backbone bất biến, không clone nguyên model để so sánh.
Chỉ nạp checkpoint local đáng tin; trạng thái RNG yêu cầu weights_only=False khi restore.

## 5. 1_eda.py — Tải và khảo sát dữ liệu

Source: [1_eda.py](../src/1_eda.py). Chi tiết: [Data_EDA](design/Data_EDA.md).

| Mục | Mô tả |
|---|---|
| What | Tải/kiểm tra dữ liệu, thống kê lớp/RGB/độ sáng/trùng ảnh, chiếu PCA/t-SNE |
| Why | Hiểu dữ liệu trước train và cung cấp report cho HTML |
| Who | Người thực hiện chạy script; explorer.html đọc artifact |
| When | Trước model/train; chạy lại khi cần tái sinh thống kê hoặc pixel chunks |
| Where | eda.json, split JSON, projection.json trong outputs/; RGB gzip trong data/ |
| How | NumPy thống kê, PCA fit train, t-SNE mẫu train và gzip pixel local |

Report mô tả đủ 60.000 ảnh; phép chiếu dùng RGB pooled, chưa phải embedding model.
Script ghi nhận duplicate nhưng không tự xóa ảnh hoặc sửa split.

## 6. 2_train.py — Smoke kỹ thuật hiện tại

Source: [2_train.py](../src/2_train.py). Kế hoạch: [Training_Strategy](design/Training_Strategy.md).

| Mục | Mô tả |
|---|---|
| What | Kiểm tra A–D bằng mini-epoch, validation và checkpoint replay |
| Why | Xác nhận pipeline trước tuning và thu số đo tài nguyên có phạm vi rõ ràng |
| Who | Thành viên chạy script; audit đọc log/summary làm bằng chứng |
| When | Sau data/model; hiện từ chối PHASE khác smoke và workers khác 0 |
| Where | Run mới cho mỗi chiến lược, config/log/history/summary và checkpoint local |
| How | Loop trực tiếp, CrossEntropyLoss, AdamW/AMP; 10 batch train, 2 batch val và replay |

Kiểm tra shape, finite loss/gradient, optimizer, freeze/buffers, eval, save/load và bước kế tiếp sau restore.
Replay so loss/state trong tolerance; miniature epoch không phải epoch đủ 45.000 ảnh.
Không dùng test, không tuning LR, không tạo best final và không train đủ E epoch.
Sửa code/pipeline ghi CHANGELOG; audit chỉ ghi kết quả có bằng chứng, không kể quá trình sửa.

## 7. 3_evaluate.py — Chưa triển khai

Source: [3_evaluate.py](../src/3_evaluate.py), hiện rỗng.
What/Why: dự kiến đánh giá checkpoint bằng metric và predictions tái tính được.
Who/When: người thực hiện sau khi khóa cấu hình, test chỉ dùng ở đánh giá cuối.
Where/How: dự kiến outputs/runs/lan_N/evaluation/, model.eval và inference_mode.
Các mô tả ở mục này là thiết kế, không phải chức năng đã có hoặc đã chạy.

## 8. 4_demo.py — Chưa triển khai

Source: [4_demo.py](../src/4_demo.py), hiện rỗng.
What/Why: dự kiến minh họa suy luận Top-1/Top-5 của checkpoint CIFAR-100.
Who/When: thành viên trình bày và giảng viên sau khi có checkpoint được chọn.
Where/How: dự kiến đọc ảnh/model local, dùng chung preprocessing và class order, không train ngầm.
Demo chưa có giao diện thực thi; không xem file rỗng là một sản phẩm đã hoàn tất.
