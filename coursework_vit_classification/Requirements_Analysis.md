# Phân tích yêu cầu — Phân loại ảnh bằng Vision Transformer

Ngày lập: 05/11/2026. Trạng thái: thiết kế, chưa triển khai hoặc huấn luyện.

Kế hoạch chi tiết: [Implementation_Plan.md](Implementation_Plan.md).

## 1. Bài toán và mục đích

- Phân loại ảnh bằng Vision Transformer (ViT), đúng hướng đề được cung cấp.
- Dataset đã chọn: CIFAR-100; mỗi ảnh đầu vào cần được gán một trong 100 lớp.
- So sánh với CNN baseline và đánh giá tác động của freeze/fine-tune.
- Đây là đồ án vận dụng; ưu tiên quy trình rõ ràng, chạy được và giải thích được kết quả.
- Nhóm 6 thành viên, thời gian 4 tuần; một thành viên phụ trách tích hợp chính.

## 2. Đối chiếu yêu cầu đề

| Nội dung trong hướng dẫn | Cách đáp ứng |
|---|---|
| ViT image classification | Dùng ViT nguyên bản, có patch embedding và Transformer encoder |
| Tiny ImageNet hoặc CIFAR-100 | Chọn CIFAR-100 |
| Báo cáo vấn đề và phạm vi | Mô tả bài toán, dữ liệu, giới hạn và quy trình |
| Dữ liệu mẫu và mô tả | EDA, phân bố lớp, ảnh minh họa, nguồn và cách chia tập |
| Mã nguồn, hướng dẫn cài đặt/chạy | Pipeline Python đơn giản, cấu hình tập trung |
| Demo | Nạp checkpoint, dự đoán ảnh thuộc phạm vi CIFAR-100 |
| Kết quả so với baseline | CNN và ViT, mỗi model có freeze và fine-tune |
| Phân tích lỗi và giới hạn | Chỉ số theo lớp, nhầm lẫn, ảnh sai, chi phí tính toán |
| Mở rộng: CNN cùng số tham số | Cặp đề xuất hiện chưa đạt; phải công bố chênh lệch |
| Mở rộng: visualize attention | Thực hiện sau phần cốt lõi nếu còn thời gian |

Không có ngưỡng accuracy bắt buộc hoặc yêu cầu đạt kết quả cao nhất trong đề đã cung cấp.
Mục tiêu định lượng về accuracy sẽ ghi sau chạy thử, không đặt số liệu thiếu căn cứ.

## 3. Đầu vào, đầu ra và phạm vi

- Đầu vào huấn luyện: ảnh RGB và nhãn CIFAR-100, sử dụng nhãn chi tiết 100 lớp.
- Đầu ra model: 100 logits; khi trình bày dự đoán dùng softmax và top-k.
- Đầu ra thực nghiệm: cấu hình, log epoch, best/last checkpoint, metric và thời gian/VRAM.
- Sản phẩm trình bày: notebook đọc kết quả đã lưu và demo inference đơn giản.
- Không phát triển hệ thống phát hiện deepfake, detection hoặc segmentation.
- Không train từ đầu, không tìm kiếm hàng loạt kiến trúc, không phát triển dịch vụ trả phí.
- Dự đoán ảnh ngoài 100 lớp vẫn trả về một lớp; không xem đó là khả năng nhận diện mọi ảnh.

## 4. Dữ liệu và nguyên tắc đánh giá

CIFAR-100 gồm 60.000 ảnh RGB 32 × 32: 50.000 train và 10.000 test có nhãn.
Mỗi lớp có 500 ảnh train và 100 ảnh test; có thêm nhãn nhóm 20 lớp nhưng không dùng làm đích chính.

Phương án split dự kiến: 45.000 train, 5.000 validation, 10.000 official test.
Tách validation phân tầng theo lớp, seed 42; lưu chỉ số để bốn cấu hình dùng cùng split.
Không dùng test để chọn learning rate, epoch, checkpoint hoặc model thắng cuộc trong quá trình train.

EDA cần xem phân bố lớp, chất lượng ảnh, kích thước, thống kê kênh màu và ví dụ dễ nhầm.
Thống kê phục vụ thiết kế preprocessing và lựa chọn xử lý chỉ dựa trên train.
Augmentation chỉ áp dụng cho train; validation/test dùng biến đổi xác định.
Không commit dataset, cache trọng số hoặc dữ liệu pixel sinh ra từ ảnh.

Nguồn: [CIFAR chính thức](https://www.cs.toronto.edu/~kriz/cifar.html).
Ghi nguồn, thông tin sử dụng dữ liệu và kiểm tra toàn vẹn khi triển khai tải dữ liệu.

## 5. Các bước thực hiện

1. Hoàn thiện đề cương: tên đề tài, mục tiêu, phạm vi, sản phẩm và tiêu chí đánh giá.
2. Kiểm tra GPU, thư viện, quyền tải dữ liệu và pretrained weights.
3. Chuẩn bị CIFAR-100, lưu split và thực hiện EDA.
4. Nạp CNN/ViT pretrained, thay head 100 lớp, xác nhận tham số được train.
5. Smoke run cả bốn cấu hình để kiểm tra loss, gradient, VRAM, tốc độ và checkpoint.
6. Chốt thông số, thực hiện bốn lần train chính thức, lưu best và last.
7. Đánh giá checkpoint được chọn bằng validation trên test; lưu dự đoán theo ID.
8. So sánh kết quả, phân tích lỗi và giới hạn; thêm attention nếu còn thời gian.
9. Hoàn thiện notebook báo cáo, demo và hướng dẫn tái lập.
10. Kiểm tra chạy lại phần báo cáo/demo, rà soát tiêu chí và chuẩn bị trình bày.

## 6. Tiêu chí nghiệm thu

- [ ] Model chính đúng ViT; giải thích được patch, vị trí, CLS token và encoder.
- [ ] Cả bốn cấu hình dùng cùng dataset/split, khởi tạo độc lập từ pretrained.
- [ ] Freeze chỉ cập nhật head; fine-tune cập nhật backbone và head.
- [ ] Có log từng epoch, cấu hình đầy đủ và checkpoint có thể nạp lại.
- [ ] Có kết quả thực đo của bốn cấu hình; không thay số thiếu bằng số giả định.
- [ ] Test được giữ riêng; so sánh rõ checkpoint và chế độ đánh giá.
- [ ] Có baseline, phân tích theo lớp/ảnh sai và chi phí tài nguyên.
- [ ] Có mã nguồn, hướng dẫn chạy, notebook, demo và giới hạn của kết luận.
- [ ] Ghi rõ chưa đạt so sánh cùng số tham số nếu vẫn dùng cặp hiện tại.

## 7. Giới hạn cần công bố

Pretrained cùng ImageNet-1K không đồng nghĩa cùng recipe hoặc ngân sách pretraining.
Kết quả phản ánh transfer learning trên CIFAR-100 của các checkpoint cụ thể.
Một seed cho mỗi cấu hình chưa đủ kết luận về độ ổn định thống kê.
Ảnh resize lên 224 × 224 không tạo thêm chi tiết so với ảnh gốc 32 × 32.
Attention là thông tin hỗ trợ quan sát, không tự chứng minh nguyên nhân dự đoán.
