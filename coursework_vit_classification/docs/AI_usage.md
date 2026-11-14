# Ghi nhận hỗ trợ AI

## 05/11/2026

- Phạm vi hỗ trợ: tổng hợp yêu cầu, thiết kế bốn thí nghiệm, kế hoạch và cấu trúc dự án.
- Các quyết định được thống nhất: CIFAR-100, CNN/ViT với freeze và fine-tune toàn bộ.
- Hướng model được ghi trong kế hoạch: ResNet-50 và ViT-B/16; tài nguyên cần smoke kiểm chứng.
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
