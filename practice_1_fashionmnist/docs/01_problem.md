# Bài 1 — Phân loại FashionMNIST bằng PyTorch và MLP

## Giới thiệu

Bài tập yêu cầu xây dựng quy trình phân loại ảnh thời trang bằng PyTorch. Với mỗi ảnh đầu vào, mô hình cần dự đoán một trong 10 loại sản phẩm của FashionMNIST.

Đây là bài toán **phân loại đa lớp có giám sát**: mỗi ảnh có một nhãn đúng để mô hình học mối liên hệ giữa nội dung ảnh và loại sản phẩm.

Theo lựa chọn của người thực hiện, mô hình sử dụng **MLP (Multilayer Perceptron)**. Ảnh được chuyển thành vector trước khi đưa qua các lớp kết nối đầy đủ. Kiến trúc và hyperparameters được ghi ở [03_model.md](03_model.md) và [05_experiments.md](05_experiments.md).

## Mục tiêu

- Hoàn thiện quy trình xử lý dữ liệu, xây dựng mô hình, huấn luyện và đánh giá.
- Sử dụng Tensor, Dataset/DataLoader, transforms, autograd và optimization trong PyTorch.
- Lưu mô hình và tải lại để dự đoán.
- Thực nghiệm thay đổi cấu trúc MLP hoặc hyperparameters và so sánh kết quả.
- Trực quan hóa loss và ảnh kèm nhãn dự đoán / nhãn thật.

## Dữ liệu

FashionMNIST gồm **60.000 ảnh train và 10.000 ảnh test**. Mỗi ảnh là grayscale 28 × 28 pixel, có giá trị gốc 0–255 (uint8) và một nhãn nguyên từ 0 đến 9.

| ID | Nhãn gốc | Diễn giải |
|---|---|---|
| 0 | T-shirt/top | Áo thun / áo top |
| 1 | Trouser | Quần dài |
| 2 | Pullover | Áo chui đầu |
| 3 | Dress | Váy liền |
| 4 | Coat | Áo khoác |
| 5 | Sandal | Dép sandal |
| 6 | Shirt | Áo sơ mi |
| 7 | Sneaker | Giày sneaker |
| 8 | Bag | Túi |
| 9 | Ankle boot | Giày boot cổ thấp |

Kiểm tra dữ liệu đã tải xác nhận mỗi lớp có 6.000 ảnh train và 1.000 ảnh test. Dữ liệu nằm tại `data/FashionMNIST_raw/`; link tải và checksum được lưu ở [data_source.md](data_source.md).

## Đầu vào và đầu ra

- **Đầu vào:** ảnh grayscale 28 × 28, chuyển thành vector 784 giá trị để đưa vào MLP. Cách biến đổi và chuẩn hóa sẽ được chốt trong thiết kế.
- **Đầu ra mô hình:** 10 điểm số tương ứng với 10 lớp; lớp có điểm số lớn nhất là nhãn dự đoán.
- **Đầu ra đánh giá:** loss, accuracy và các ảnh so sánh nhãn dự đoán với nhãn thật.

## Yêu cầu thực hiện và sản phẩm nộp

1. Tải FashionMNIST, áp dụng transforms và tạo Dataset/DataLoader.
2. Xây dựng MLP và vòng lặp forward → loss → backward → cập nhật tham số.
3. Đánh giá accuracy, lưu mô hình và tải lại để dự đoán.
4. Thực nghiệm với mạng hoặc hyperparameters; vẽ đồ thị loss và hiển thị ảnh dự đoán / nhãn thật.
5. Nộp code Python, báo cáo ngắn, đồ thị loss và ảnh kết quả; tham khảo tài liệu/tutorial PyTorch theo đề.

## Phạm vi và nguyên tắc đánh giá

- Triển khai bằng script `.py`, không dùng notebook; mô hình thuộc họ MLP.
- Tách validation từ train để chọn cấu hình; giữ test cho đánh giá cuối. Cách chia sẽ được thống nhất trước triển khai.
- Không dùng kết quả test để chọn mô hình hoặc điều chỉnh hyperparameters.
- Ghi nhận seed, cấu hình, cách chia dữ liệu và kết quả thực tế để tái lập thí nghiệm.
- Đề chưa chỉ định số epoch, số thí nghiệm hoặc ngưỡng accuracy. Không tự coi các giá trị chưa thống nhất là yêu cầu bắt buộc.
- Chỉ ghi nhận hoàn thành khi có bằng chứng chạy, đánh giá và tải lại mô hình thành công.

## Trạng thái hiện tại

Đã hoàn thành chuẩn bị dữ liệu, EDA, MLP 784→256→128→10, huấn luyện 10 lần, đánh giá validation/test và trực quan hóa. Mô hình bàn giao là `lan_10/best.pt` (validation 90,57%, test 89,41%).

Kết quả: [04_result.md](04_result.md). So sánh các lần train và đối chiếu đề bài: [05_experiments.md](05_experiments.md).
