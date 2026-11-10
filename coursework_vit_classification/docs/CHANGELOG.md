# Changelog — Thay đổi đáng kể về code và pipeline

Ghi vấn đề/nhu cầu, thay đổi, lý do và cách kiểm chứng; không ghi từng chỉnh sửa nhỏ.
Trạng thái hiện tại: [progress](progress.md). Checklist smoke: [Audit_Log](Audit_Log.md).
Audit chỉ ghi kết quả kiểm tra; lịch sử hỗ trợ AI nằm ở [AI_usage](AI_usage.md).

## 05/11/2026 — Chuẩn bị dữ liệu và viewer

- Nhu cầu: dữ liệu/split nhất quán và quan sát toàn CIFAR-100 trước train.
- Thay đổi: config/data/EDA, split phân tầng seed 42 và HTML đọc report/RGB gzip local.
- Sửa stdout Windows sang UTF-8 để script không lỗi khi in tiếng Việt.
- Kiểm chứng: checksum archive, IDs/cân bằng split, toàn bộ RGB bytes, labels và thống kê train đạt.
- Browser: gallery, bộ lọc, phân trang và ảnh nguồn đã kiểm tra; không chạy model/train.

## 05/11/2026 — Phép chiếu dữ liệu

- Nhu cầu: quan sát phân bố ảnh và mức chồng lấn lớp trong không gian 2D.
- Thay đổi: PCA fit train/transform đủ 60.000 ảnh; t-SNE 5.000 train, 50 ảnh/lớp.
- Thêm chọn lớp/split, nền xám, zoom/kéo/reset và preview ảnh trong HTML hiện có.
- Sửa preview cũ khi đổi phép chiếu để không hiển thị nhầm ảnh/split của thao tác trước.
- Kiểm chứng: shapes, finite coordinates, train-only sampling, cân bằng lớp và thao tác browser đạt.
- Giới hạn: RGB pooled features chưa là embedding CNN/ViT; không tạo tọa độ theo nhãn.

## 10/11/2026 — Tổ chức nhánh và tài liệu

- Nhánh cousework bỏ theo dõi Practice 2, giữ nguyên 82 file local đã đối chiếu SHA-256.
- Chia docs/define và docs/design; bổ sung model/chiến lược tuning, chưa đổi pipeline train.
- Chuyển progress/AI_usage vào docs/; thêm audit Markdown, cập nhật các liên kết và quy ước ghi chép.
- Kiểm chứng: liên kết nội bộ, UTF-8, giới hạn Markdown; không chạy lại EDA/train vì chỉ đổi tài liệu.
- Các thay đổi coursework hiện chưa commit/push; trạng thái phát hành xem progress.

## 10/11/2026 — Bàn giao chờ xác nhận phạm vi

- Đánh dấu phương án ViT/CIFAR-100 chờ xác nhận với giảng viên về hai dòng ViT/Anti Face Deepfake.
- Chỉ cập nhật tài liệu/trạng thái và phạm vi nhánh; không triển khai tiếp hoặc chạy thí nghiệm.
- Chuẩn bị commit/push cousework theo phạm vi được duyệt, không kèm lab/dataset/cache.

Khi gần 150 dòng, lưu lịch sử cũ vào docs/archive/changelog_<giai_doan>.md và giữ liên kết tại đây.
