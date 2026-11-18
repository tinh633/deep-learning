# Ghi nhận hỗ trợ AI

## 05/11/2026

- Phạm vi hỗ trợ: tổng hợp yêu cầu, thiết kế bốn thí nghiệm, kế hoạch và cấu trúc dự án.
- Các quyết định được thống nhất: CIFAR-100, CNN/ViT với freeze và fine-tune toàn bộ.
- Hướng hiện tại: ConvNeXt-Base và ViT-B/16; ghi nhận ResNet bên dưới là lịch sử trước 18/11.
- Đã tạo file nguồn rỗng và notebook rỗng hợp lệ; chưa sinh pipeline huấn luyện.
- Kiểm chứng: cấu trúc file, UTF-8, số dòng Markdown, liên kết và notebook JSON.
- Chưa có kết quả accuracy, thời gian hoặc VRAM của coursework.

## 05/11/2026 — Dữ liệu và EDA

- Hỗ trợ triển khai config/data/EDA và HTML explorer không dùng CDN.
- Tách report thống kê khỏi RGB gzip local; giữ dataset/pixel trong data/ được ignore.
- Sửa stdout cp1252; tải chính thức/range chậm, chuyển mirror archive và xác minh MD5 chính thức.
- Đã kiểm tra syntax Python/JavaScript; số đo dữ liệu thực được ghi vào outputs/ sau EDA.
- Cập nhật hướng dẫn chạy và tài liệu workflow; không sửa pipeline các lab cũ.
- EDA thực đủ 60.000 ảnh; kiểm tra split, RGB gzip, nhãn và thống kê train đạt.
- Phát hiện 26 nhóm trùng pixel, 12 nhóm xuyên split; không tự xóa hoặc thay split.
- Edge headless: đã xem ảnh giao diện; bộ lọc split/lớp/tên, sắp xếp và phân trang đạt.

## 05/11/2026 — PCA và t-SNE

- Bổ sung phép chiếu vào 1_eda.py và HTML hiện có; không tạo thêm pipeline src.
- PCA fit train cho toàn bộ dataset; t-SNE dùng mẫu train 50 ảnh/lớp, nhãn không tham gia tính tọa độ.
- Kiểm chứng số điểm, finite coordinates, train-only sampling và class counts đạt.
- Edge kiểm tra filter, nền xám, preview pixel thật và zoom/reset; không dùng phép chiếu làm metric model.

## 10/11/2026 — Phạm vi nhánh

- Bỏ Practice 2 khỏi index nhánh cousework bằng git rm --cached, giữ nguyên bản local.
- Đồng bộ README gốc, ignore, attributes và chỉ mục tiến độ theo coursework.
- Kiểm tra file local và đường dẫn Git sau thao tác; chưa commit/push.
- Phân nhóm tài liệu thành docs/define và docs/design; cập nhật liên kết/cây thư mục, giữ nhật ký ở gốc.
- Viết lại Define/Design theo bố cục yêu cầu; thêm hai tài liệu model, đối chiếu torchvision chính thức.
- Ghi rõ ResNet torchvision v1.5, nguồn recipe ViT và giao diện nội bộ chỉ là đề xuất; chưa sinh code train.
- Soạn Training_Strategy.md: tách smoke/tuning/final/test, bảng 8 pilot + 4 final, fixed variables và chọn LR bằng val.
- Chuyển nhật ký vào docs và bổ sung Audit_Log/CHANGELOG theo vai trò riêng; chưa chạy smoke hoặc sửa code train.
- Chốt tài liệu và trạng thái chờ giảng viên xác nhận ViT/Anti Face Deepfake; không triển khai tiếp trong lúc chờ.
- Chuẩn bị xuất bản nhánh cousework theo phạm vi được duyệt, loại lab/dataset/cache, giữ file local.

## 14/11/2026 — Khôi phục task trước

- Đối chiếu log/summary và code đã lưu; nhận diện C bị ngắt, không tự đánh dấu PASS.
- Bổ sung model/loader/logger/smoke, giảm bản sao RAM và guard tài nguyên; giữ artifact cũ.
- docs/src.md mô tả 5W1H theo yêu cầu; cập nhật README/Define/Design với phạm vi đã xác nhận.
- Recheck A/B/C trên CUDA PASS; D tự dừng khi thiếu Windows commit, không hạ ngưỡng bảo vệ.
- Audit lưu kết quả thực; changelog ghi sửa code/pipeline; chưa tuning/train chính thức hoặc push.
- Review nêu giới hạn guard sau replay và bias thời gian batch; giữ raw report, cập nhật chú thích audit.
- Chuẩn bị xuất bản coursework theo phạm vi được duyệt, giữ HTML mới trên remote bằng fast-forward.

## 16/11/2026 — Source và tài liệu per-file

- Hoàn thiện code train/evaluate/demo, tách smoke, bổ sung guard cuối workflow và kiểm tra resume.
- Chuyển mô tả tổng hợp thành docs/src/README.md và từng file Markdown 5W1H theo source.
- Thêm kiểm tra hồi quy CPU bằng model/dữ liệu tổng hợp; không coi đó là lượt train/test CIFAR-100.
- Chưa chạy tuning/final, test thật hoặc commit/push giai đoạn mới.
- Bổ sung mô tả tests/test_pipeline.py trong docs/src/test_pipeline.md, đối chiếu code và report hiện có; chỉ đổi tài liệu.
- Rà soát Markdown và chuyển mục Which sang mô tả model/dataset/công cụ/cấu hình; đồng bộ quy tắc, không sửa code thực thi.
- Chạy thử A một epoch đầy đủ theo phạm vi resource_probe; không dùng để chọn LR, test hoặc kết quả cuối.
- Lan_9 hoàn tất 45k/5k, val 64,16%; đối chiếu checkpoint/backbone/BN và 2813 optimizer steps, lưu verification và telemetry.
- Ghi quy trình tuning → chọn val → final khởi tạo lại, phân vai 6 người và tiêu chí bàn giao trong Work_Allocation.md; chưa gán tên hoặc chạy thêm.
- Xuất bản theo phạm vi được duyệt: coursework và quy tắc liên quan trên cousework; chỉ text artifact, không dataset/weights.
- Yêu cầu bổ sung checkpoint: kiểm tra 7 best/last, head 100 lớp và cấu trúc resume; tạo manifest SHA-256, cấu hình LFS và hướng dẫn lấy về.

## 18/11/2026 — Đồng bộ cặp model gần quy mô

- Thay ResNet-50 bằng ConvNeXt-Base theo lựa chọn mới, giữ ViT-B16 và ma trận A–D.
- Sửa head, freeze LayerNorm, stochastic depth, smoke fingerprint và chặn pilot khác kiến trúc.
- Viết ConvNeXt_Base.md, cập nhật Define/Design/source docs và ghi rõ checkpoint ResNet là lịch sử.
- Kiểm chứng cấu trúc torchvision trên meta và hồi quy CPU tổng hợp; không tải weights hoặc chạy GPU mới.
- Chưa chốt budget hoặc train mới; đã duyệt commit/push cousework, kiểm tra lại evidence/hash và phạm vi trước xuất bản.
- Push d5e6dfc thành công lên cousework ngày 18/11, upload đủ 7 checkpoint LFS (1,3 GB); không còn blocker upload cũ.
- Thêm Training_Monitor.ipynb đọc JSON, tách train/val, so sánh run, bảng trạng thái và refresh tùy chọn; không nạp model.
- Chạy mọi code cell (refresh mặc định tắt), kiểm tra history thật, hai run tổng hợp, bộ lọc và JSON lỗi; xem ảnh biểu đồ; đã duyệt commit/push notebook lên cousework.
- Xóa lan_9 ResNet theo yêu cầu, cập nhật manifest; chạy A smoke lan_9 lỗi mạng, tải weights chính thức rồi lan_10 dừng đúng guard commit 2942,83 MiB.
- Đã duyệt A 2×5 pilot và final 20 epoch; chưa bắt đầu do smoke thiếu tài nguyên, đang chờ giải phóng bộ nhớ; không tự đóng ứng dụng hoặc hạ ngưỡng.
- Được phép đóng ứng dụng: đã đóng Zalo/Discord, Chrome bị từ chối quyền; smoke lan_11 batch16 đạt, bắt đầu A_LR1 lan_12.
- Controller local theo dõi nhiệt/disk, dừng theo lỗi và chọn LR bằng val rồi khởi tạo final mới; không test hoặc tự commit/push.
- Hoàn tất pilot lan_12/13, chọn LR2; final lan_14 bị guard dừng giữa epoch20, giữ epoch19, giải phóng bộ nhớ rồi resume riêng epoch20.
- Đủ 20 epoch, best val81,46% epoch9; kiểm chứng 6 best/last, frozen tensors, optimizer/RNG/history và reset final, cập nhật manifest và A_Results.md.
- Giữ traceback/interruptions.json, GPU trở về 0 MiB; khóa train lặp bằng config, chưa test/commit/push kết quả mới.
- Chạy lại notebook monitor trên 3 history thật (5/5/20 epoch), kiểm tra bảng 81,46% và xem ảnh biểu đồ; đồng bộ trạng thái Define/Design/README.
- Được duyệt push kết quả A: kiểm tra lại SHA-256 toàn bộ 12 checkpoint, trạng thái completed, liên kết/Markdown/AST; chỉ xuất coursework và chỉ mục liên quan.
- Push 68641b3 thành công lên cousework, upload đủ 7 LFS mới (~2,46 GB); đồng bộ ghi nhận xuất bản, không kèm dataset/cache.
