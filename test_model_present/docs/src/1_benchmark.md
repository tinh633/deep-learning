# 1_benchmark.py — Đánh giá pretrained model và tài nguyên

Source: [1_benchmark.py](../../src/1_benchmark.py).
Đọc [config](config.md) và hoàn tất [chuẩn bị dữ liệu](0_preparedata.md) trước.

## Phạm vi

Nạp sáu model pretrained, chạy eval/inference_mode, không train hoặc fine-tune.
Không có optimizer/backward; giữ nguyên classifier 1.000 lớp và transforms theo weights.
Notebook chỉ đọc report đã lưu, không tự chạy benchmark khi Run All.

```powershell
.venv/Scripts/python.exe test_model_present/src/1_benchmark.py --smoke
.venv/Scripts/python.exe test_model_present/src/1_benchmark.py
```

--model <name> <folder> là chế độ worker nội bộ do subprocess gọi.
Chạy benchmark tạo lượt mới; không ghi đè hoặc resume lượt cũ.

## 1. Kiểm tra dữ liệu trước khi chạy

read_manifest() xác nhận file CSV tồn tại và có id/path/label/split.
Mỗi ảnh phải tồn tại, label 0–999, đúng split validation; ID/path không trùng.
Nếu có image_sha256 thì bytes phải khớp; kiểm tra đủ đúng 10 lớp, 50 ảnh/lớp.
Sau kiểm tra, MAX_IMAGES có thể giới hạn mẫu; smoke lấy 16 dòng đầu của manifest.
Smoke không bảo đảm cân bằng 10 lớp và không dùng để xếp hạng accuracy.

## 2. Điều phối các model

- Tạo outputs/lan_N hoặc outputs/smoke_N với số thứ tự chưa tồn tại.
- Lưu manifest JSON, run_settings và thứ tự model xáo bằng seed 42.
- Chạy từng model tuần tự, mỗi model trong một subprocess để tách vòng đời bộ nhớ.
- Ghi stdout/stderr vào <model>.log, lưu returncode vào status.json.
- Model lỗi không xóa kết quả model đã hoàn tất; cuối lượt báo lỗi nếu có worker thất bại.
- results.csv chỉ tổng hợp các worker hoàn tất thành công.

## 3. Nạp model và tính accuracy

Worker đọc runtime đã lưu, đặt torch seed, tắt TF32 và chọn CUDA nếu có, nếu không dùng CPU.
Cache weights ở data/torch_cache/; có thể dùng lại cache Torch local nếu hash prefix khớp.
Nạp đúng enum weights, chuyển model sang device và eval; đối chiếu SHA-256 prefix URL weights.
LabeledImages đọc ảnh bằng PIL, chuyển RGB và dùng weights.transforms().
DataLoader dùng batch 16, shuffle=False, num_workers=0.

Trong torch.inference_mode():

1. Forward và kiểm tra logits có shape batch × 1.000.
2. Softmax rồi lấy 5 dự đoán cao nhất.
3. Đếm Top-1 đúng và nhãn thật xuất hiện trong Top-5.
4. Lưu ID, nhãn thật, 5 index dự đoán và xác suất tương ứng.

Accuracy tính trên toàn bộ nhãn 1.000 lớp; không cắt đầu ra thành 10 lớp đánh giá.
evaluation_seconds bao gồm loader/transform/transfer/inference trong vòng đánh giá,
không gồm tải weights và không được gọi là phép đo end-to-end I/O lạnh có kiểm soát.

## 4. Đo latency, throughput và VRAM

measure() forward warm-up, đồng bộ CUDA rồi reset peak memory khi dùng GPU.
Mỗi request đồng bộ trước/sau forward và đo bằng perf_counter.
Protocol chính: 20 warm-up, 3 nhóm × 100 request = 300 số đo/phép đo.
Tensor đầu vào đã preprocess và nằm trên device; không tính decode/transform/transfer.

- Latency: một ảnh, batch 1; lưu mean/median/P95 và toàn bộ thời gian từng request.
- Throughput: batch 16 ảnh, tính batch / mean thời gian batch.
- Timing dùng lại các ảnh đầu manifest, không đo một lượt độc lập trên từng ảnh trong 500 ảnh.
- MiB = bytes / 2^20; allocated là bộ nhớ tensor do PyTorch cấp phát, reserved là bộ nhớ allocator giữ.
- Peak được reset sau warm-up; reserved vẫn có thể gồm cache tạo từ accuracy pass trước đó.
- Không cộng allocated với reserved, không đồng nhất với VRAM toàn hệ thống trên Task Manager.
- Trên CPU, các trường GPU/peak VRAM là null; không diễn giải thành 0 MiB.
- Phép đo FP32 không có gradient/optimizer, không đại diện bộ nhớ khi train/fine-tune.

## 5. File kết quả

| File trong một run | Nội dung |
|---|---|
| manifest.json | Bản sao các ảnh/nhãn và đường dẫn dùng cho run |
| run_settings.json, model_order.json | Profile timing và thứ tự model |
| <model>.log | Tiến trình tải/nạp, accuracy pass và timing, lỗi nếu có |
| <model>.json | Chỉ số, weights/hash, transforms, môi trường, raw timing |
| <model>_predictions.json | ID, true, top5, scores để tái tính accuracy/phân tích |
| status.json | Trạng thái và returncode từng subprocess |
| results.csv | Bảng tổng hợp các model hoàn tất |

save_json() ghi file tạm rồi replace và từ chối NaN; không ghi checkpoint model mới.
source_sha256 ghi source thực sự chạy; sửa source sau này không được sửa ngược hash lịch sử.

## 6. Đọc bảng notebook và CSV

| Cột CSV → bảng notebook | Ý nghĩa / chiều tốt hơn |
|---|---|
| n → Tổng ảnh | Số ảnh accuracy pass, chính thức 500 |
| top1_correct / top1_wrong → Đúng / Sai | Số dự đoán Top-1 đúng/sai, tổng bằng n |
| top1_percent → Accuracy (%) | 100 × đúng / n; cao hơn tốt hơn |
| top5_correct, top5_percent | Số/tỷ lệ nhãn thật trong 5 dự đoán; cao hơn tốt hơn |
| latency_ms → ms/ảnh (B1) | Mean forward batch 1; thấp hơn tốt hơn |
| p95_ms | Phân vị 95 thời gian batch 1; thấp hơn tốt hơn |
| images_per_second → ảnh/s (B16) | Throughput batch 16; cao hơn tốt hơn |
| peak_allocated_mib → VRAM MiB (B16) | Peak allocated của throughput; thấp hơn tiết kiệm hơn |
| peak_reserved_mib | Peak memory allocator giữ, không cộng với allocated |
| parameters_m | Tổng tham số / một triệu; không trực tiếp xác định tốc độ |

Không lấy 1000/latency_ms để đối chiếu trực tiếp throughput vì batch khác nhau.
Notebook đọc results.csv cho bảng/biểu đồ và predictions cho recall/confusion theo lớp.
Confusion 10 × 11 gộp 990 lớp ngoài thành một cột chỉ để trình bày, không đổi cách chấm.

## Kiểm chứng và giới hạn

lan_1 hoàn tất 6 model trên cùng 500 ảnh; có kiểm tra Top-1/Top-5 từ predictions,
hash manifest/weights, thời gian hữu hạn/dương và reserved ≥ allocated; xem [progress](../../progress.md).
smoke_1 lịch sử dùng ImageNetV2; smoke_2 mới thuộc ImageNet-1K, không trộn hai nguồn kết quả.
Chỉ một mẫu 10 lớp, một máy và weights/recipe khác nhau: không khái quát thứ hạng cho mọi dữ liệu.
Tài liệu được đối chiếu source; lần bổ sung tài liệu không chạy inference/timing lại.
