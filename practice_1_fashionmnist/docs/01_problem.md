# 1. Define problem

Phân loại ảnh FashionMNIST vào 10 nhãn bằng MLP, không dùng notebook.
Người dùng bổ sung một CNN cơ bản sau khi hoàn tất so sánh và phân tích lỗi MLP.
Input: ảnh grayscale 28 × 28. Output: 10 logits, lấy argmax làm nhãn dự đoán.

## Yêu cầu đã thống nhất

- Model class có __init__ và forward; mỗi dòng thể hiện một thao tác.
- Config tập trung; log tách riêng; code dùng thư viện có sẵn, tránh tách file vụn.
- Một HTML để xem dữ liệu, EDA, confusion matrix và ảnh sai nhãn.
- Làm từng phần, dừng để người dùng kiểm tra trước khi tiếp tục.
- Mục tiêu train >96% (hướng tới 97%), val/test >95%; chưa cam kết đạt được.
- Chọn cấu hình bằng validation, khóa lựa chọn trước khi test.

## Trạng thái

Hiện làm lại chuỗi so sánh: baseline lan_10, tăng rộng lan_11, tăng sâu lan_12 theo kế hoạch training.
Run lan_12 cũ không còn trong thư mục; báo cáo đã xuất trước đó là lịch sử, không phải kết quả chuỗi mới.
Chưa tự chốt model cuối; mốc >95% val/test chưa đạt và không phải ngưỡng pass đã được xác nhận trong đề.
Xem [đối chiếu yêu cầu lab và kết quả](05_evaluation.md).

Tiếp theo: [data preparation](02_data_prepare.md).
