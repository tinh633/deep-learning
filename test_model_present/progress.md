# Tiến độ — benchmark hoàn tất

Cập nhật2026-11-02.

## Dữ liệu và model

- ImageNet-1K validation:10 lớp xác định trước ×50 ảnh, tổng500; không lấy train.
- HF revision49e2ee26f3810fb5a7536bbf732a7b07389a47b5; đã đọc49.948 bản ghi để thu đủ mẫu.
- Manifest data/imagenet1k/manifest.csv:500 ID riêng biệt và SHA ảnh đã đối chiếu.
- Sáu model weights ImageNet1K V1, classifier gốc1.000 lớp, FP32/TF32 off.
- Máy RTX3050Laptop, torch2.11+cu126/torchvision0.26; không train/fine-tune.

## Thực chạy

- Smoke_1 lịch sử thuộc ImageNetV2, không dùng làm kết quả ImageNet1K.
- Smoke_2:6/6 model pretrained qua trên16 ảnh của bộ ImageNet1K mới.
- Lan_1:6/6 model hoàn tất đủ500 ảnh, không OOM; model chạy riêng từng subprocess.
- Latency batch1 và throughput batch16,20 warm-up rồi3×100 request/phép đo.
- EfficientNet-B0 Top1=76,2%, Top5=95,2%, cao nhất trong mẫu500 ảnh.
- MobileNetV2 latency5,96ms, throughput842,3 ảnh/s; nhanh nhất trong lần đo này.
- GoogLeNet peak allocated batch16=152,9MiB, thấp nhất; VGG16=939,7MiB.

## Kiểm tra đã qua

- Tái tính Top1/Top5 từ prediction của500 ID và nhãn, khớp6 report.
- Cùng manifest/source SHA; weights URL/hash và1000 lớp được xác nhận.
- Profile benchmark, CUDA, FP32 và timing300 request/phép đo đúng cấu hình.
- Timing hữu hạn/dương; peak reserved≥allocated và allocated>0.
- Notebook4 code cells đã chạy qua kernel .venv, lưu bảng, metric plots và confusion10×11.
- Confusion giữ nhãn1.000 lớp, gộp990 lớp ngoài thành một cột chỉ để trình bày.

## Bàn giao

Mở Model_Benchmark.ipynb hoặc bảng kết quả trong README.
CSV/JSON/predictions/status/log ở outputs/lan_1; không cần chạy benchmark lại để xem.
Dataset/weights cache giữ local, không commit ảnh. Audit/presentation riêng đã bỏ.
Validation có thể được dùng để phát triển model; không bảo đảm hoàn toàn chưa từng thấy.
Mẫu10 lớp và một máy không đại diện toàn ImageNet; không chọn ảnh theo kết quả.

Không commit/push. Còn lại review giao diện notebook trong IDE nếu cần.

- Added explicit Top1 correct/wrong and Top5 correct counts to reports/CSV and simplified notebook main table. Counts derived from saved predictions; no inference or timing rerun. Explained common ImageNet1K pretraining dataset versus differing recipes and identical evaluation images.

- README: added benchmark motivation after architecture presentation, practical evaluation questions and accuracy/latency/throughput/VRAM trade-offs. Documentation only; no benchmark rerun.

- 2026-11-04: Publication authorized: test_model_present source/notebook/results to Thai branch; dataset and weights cache excluded.

## 07/11/2026 — Tài liệu source

- Thêm docs/src/config.md, 0_preparedata.md và 1_benchmark.md; liên kết từ README.
- Đối chiếu trực tiếp ba file Python: config, streaming/mapping/manifest, subprocess và protocol đo.
- Giải thích bảng chỉ số, hash, cache weights, chế độ smoke và giới hạn VRAM/timing.
- Ghi rõ smoke_1 là ImageNetV2, smoke_2/lan_1 là ImageNet-1K.
- Kiểm tra UTF-8, giới hạn 150 dòng và liên kết nội bộ; không chạy benchmark hoặc đổi kết quả.
- Chưa commit/push phần tài liệu mới.
- Thêm docs/Models.md: sáu kiến trúc, đặc trưng, head/pretrained, số tham số và cách diễn giải kết quả.
- 07/11/2026: chuẩn bị commit/push riêng test_model_present lên Thái; không kèm thay đổi coursework.
