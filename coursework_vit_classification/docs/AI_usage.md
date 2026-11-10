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
