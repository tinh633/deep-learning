# 1_eda.py — EDA và dữ liệu cho HTML

Source: [1_eda.py](../../src/1_eda.py).
Script chạy tuần tự ngay khi thực thi; không import để tái sử dụng.
Không train hoặc nạp model/checkpoint.

## Chạy và điều kiện

Từ gốc workspace: `python practice_2_pretrained_model/src/1_eda.py`.
Cần dataset local: load_datasets(download=False); split được đọc/tạo qua load_split.
Script đặt mặc định các luồng BLAS=1 trước import thư viện tính toán.

## Luồng xử lý

1. Lấy train gốc/test và split45k/5k/10k cố định.
2. Xuất RGB gzip từng chunk10.000 ảnh cho HTML: năm train, một test.
3. Duyệt batch1.000 ảnh để tính thống kê, đặc trưng và hash pixel.
4. Tạo PCA chung cho toàn dataset và ví dụ preprocessing.
5. Kiểm tra số mẫu/ID rồi ghi outputs/eda.json.

RGB được ghi theo bytes mỗi ảnh32×32×3, không phải PNG atlas hoặc pickle.
Mỗi chunk có đường dẫn/start/count/SHA của bytes giải nén; gzip mtime=0.
Các file này ở data/CIFAR10_web/, giữ local và được tạo lại khi chạy script.

## Các nhóm thống kê

| Nội dung | Phạm vi/cách tính |
|---|---|
| Số lớp, shape, dtype, pixel_range | Mỗi split |
| RGB mean/std/histogram | Chỉ45.000 train, thang[0,1], histogram256 mức |
| Brightness | Trung bình RGB/255 của ảnh |
| Contrast | STD dân số của gray là trung bình ba kênh |
| Saturation | Trung bình HSV S=(max-min)/max, max0 thì S0 |
| Edge | Trung bình độ lệch tuyệt đối pixel gray liền kề ngang/dọc |
| Histogram/phân vị | Mỗi đặc trưng, mỗi split; quantile0/1/25/50/75/99/100% |
| Mean shift | Chênh mean split so train, chia STD train |
| Theo lớp/tương quan | Chỉ train; Pearson giữa bốn đặc trưng |
| Extremes | Bốn thấp/bốn cao mỗi đặc trưng trong train, giữ ID/nhãn |
| Trùng ảnh | SHA256 RGB chính xác trên cả ba split, ghi nhóm và cross_split |

Edge là mô tả biến thiên pixel, không phải chứng cứ ảnh lỗi/mờ.
Ảnh cực trị hoặc ảnh gần nhau trên bản đồ không tự bị xóa/sửa nhãn.
Thống kê test mang tính mô tả; không dùng để chọn cấu hình.

## Bản đồ PCA toàn bộ60.000 ảnh

Mỗi ảnh được lấy trung bình block4×4 thành RGB8×8, flatten192 chiều.
PCA2 fit chỉ train, centered và không scale từng feature; seed42.
Transform train/val/test vào cùng hệ trục, làm tròn tọa độ5 chữ số.
Tọa độ được sắp theo gallery.ids tăng dần để HTML ghép đúng ảnh.
Đây là cấu trúc theo pixel thô, không phải embedding model hoặc t-SNE.
Hai chiều làm mất thông tin; không suy năng lực model chỉ từ mức tách cụm.

## Artifact và kiểm tra

eda.json chứa splits/gallery IDs/labels, pixel_sources, train_rgb, deep_eda,
dataset_projection, duplicates và preprocessing.
Preprocessing lấy một ảnh train: gốc/resize/flip cưỡng bức/khôi phục Normalize;
flip này minh họa phép biến đổi, không có nghĩa train luôn flip mọi ảnh.
Report chỉ lưu ID/cấu hình/thống kê preprocessing, không nhúng ảnh/base64.
HTML dựng preview từ RGB local; notebook dựng preview từ dataset local.
Browser resize chỉ minh họa, số liệu normalization lấy từ pipeline torchvision trong EDA.

Kiểm tra split45k/5k/10k, train/val không giao, tổng lớp khớp số mẫu,
histogram phủ hết mẫu và tọa độ PCA hữu hạn. JSON không cho NaN.
Chạy lại ghi đè artifact EDA/RGB, không đổi weights hay tự huấn luyện.
EDA JSON không chứa pixel ảnh; RGB local và outputs ảnh mẫu notebook không đưa lên Git.
Mở visualization/eda.html qua Live Server để đọc JSON và chunk RGB local.

Liên quan: [data](data.md), [README](../../README.md).

## Khả năng đọc biểu đồ HTML

Giữ PCA trên60.000 ảnh và tọa độ thật. HTML có thanh độ đậm điểm,
chế độ chỉ hiện lớp đang chọn và bộ lọc split; không dịch điểm để ép tách lớp.
Đã thử t-SNE sau PCA32 không nhãn trên60.000 ảnh,500 iteration,
nhưng không tách rõ lớp; không giữ phép tính này trong source/artifact hiện tại.
Điểm trộn phản ánh giới hạn mô tả RGB8×8, không chứng minh model phân loại kém.
