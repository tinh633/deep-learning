# Tiến độ coursework

## 05/11/2026 — Thiết kế và khởi tạo

- Dataset: CIFAR-100; bốn chiến lược CNN/ViT × freeze/fine-tune.
- Model dự kiến: ResNet-50 và ViT-B/16, pretrained ImageNet-1K.
- Đã viết phân tích yêu cầu và kế hoạch 4 tuần.
- Đã tạo bộ khung src/, docs/, visualization/, data/ và outputs/.
- Tám file Python, HTML và requirements còn rỗng; notebook hợp lệ nhưng chưa có cell.
- Không tạo best/last checkpoint, metric hoặc log giả cho lần train chưa thực hiện.
- Chưa tải dataset/weights, cài dependency, chạy smoke hoặc huấn luyện.

## Bước tiếp theo

1. Triển khai config, data và EDA.
2. Triển khai model, kiểm tra freeze/fine-tune và tài nguyên GPU.
3. Smoke bốn cấu hình trước khi chốt hyperparameter.
4. Thực hiện pipeline theo [kế hoạch](Implementation_Plan.md).

## Kiểm tra bộ khung

- Kiểm tra UTF-8, giới hạn 150 dòng Markdown, liên kết nội bộ và notebook JSON.
- Chỉ stage thư mục coursework; giữ nguyên thay đổi của benchmark và chỉ mục chung.
- Không có kiểm thử chức năng vì mã thực thi chưa được triển khai.
