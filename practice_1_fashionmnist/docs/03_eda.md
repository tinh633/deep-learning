# 3. EDA sâu và visualization

Chạy từ gốc workspace: `python practice_1_fashionmnist/src/2_eda.py`.
Script chạy tuần tự, không có main hoặc if __name__; không import script này từ file khác.

## Đầu vào và đầu ra

- Đầu vào: IDX trong data/FashionMNIST_raw, cấu hình split trong config.py.
- outputs/eda.json: EDA cơ bản tương thích gallery, train gốc/test/all.
- outputs/deep_eda.json: phân tích train/validation theo đúng split của model.
- visualization/eda/: distributions.png, classes.png, mean_variation.png, pca.png.
- Giữ một HTML: visualization/explorer.html. Mở bằng Live Server từ workspace/project root.
- Cuộn đến **EDA sâu · hiểu dữ liệu trước khi train**; lọc train/validation và nhãn.
- Không cần server Python riêng khi dùng Live Server; không dùng file:// hoặc editor Preview.

## Nội dung và cách tính

| Phân tích | Dữ liệu sử dụng | Cách đọc |
|---|---|---|
| Nhãn/split | Toàn bộ 54.000 train, 6.000 validation | Cân bằng lớp, không giao nhau |
| Brightness | Trung bình 784 pixel | Thang 0–255, ảnh sáng/tối |
| Contrast | Độ lệch chuẩn pixel mỗi ảnh | Không phải độ lệch chuẩn giữa các ảnh |
| Foreground | Tỷ lệ pixel >30 | Ngưỡng cố định, không phải segmentation |
| Bounding box | Vùng bao pixel >30 | Rộng/cao theo pixel, ảnh trống có kích thước 0 |
| Tâm ảnh | Trung bình tọa độ có trọng số cường độ | x/y từ 0 đến 27 |
| Ảnh trung bình/biến thiên | Từng lớp trong train | Ảnh tổng hợp; STD pixel cao nghĩa là biến thiên lớn |
| PCA | Fit 5.000 train, chiếu 1.000 validation | Lấy 500/100 ảnh mỗi lớp, seed 42 |
| Ảnh cực trị | Toàn bộ từng split | 5 tối nhất, 5 sáng nhất, 5 tương phản thấp nhất |
| Cặp gần giống | 2.000 ảnh train trong mẫu PCA | Láng giềng theo pixel, giữ 8 cặp khoảng cách nhỏ nhất |

Các histogram train/val dùng mật độ, không so số ảnh thô giữa hai tập khác kích thước.
Boxplot theo lớp hiển thị median/Q1/Q3, ẩn điểm ngoài whisker trên hình nhưng không xóa dữ liệu.
PCA dùng ảnh /255, 32 thành phần; hai thành phần đầu giải thích khoảng 46,42% phương sai,
32 thành phần giải thích khoảng 82,58% trên mẫu train của lần chạy hiện tại.
Không fit preprocessing/PCA trên validation hoặc test. Không dùng test trong EDA sâu.

## Ảnh minh họa có thể truy vết

- Mọi ảnh cực trị/cặp gần giống có index trong 60.000 ảnh train gốc, nhãn và pixel thật.
- Bộ lọc nhãn chỉ lọc danh sách cực trị toàn split, không lấy lại top 5 riêng mỗi nhãn.
- Khoảng cách cặp gần giống là RMSE pixel [0,1]; nhỏ hơn nghĩa là gần hơn theo pixel.
- PCA index train/validation được lưu để kiểm tra nguồn mẫu và tái lập.
- Trùng chính xác toàn dữ liệu vẫn kiểm tra bằng SHA256 trong EDA cơ bản.

## Giới hạn và kiểm tra

- Hai chiều PCA mất thông tin; chồng lấn không chứng minh model không thể phân loại.
- Cặp gần giống chỉ tìm trong tập con, không phải so sánh mọi cặp trong dataset.
- Ảnh cực trị không đồng nghĩa lỗi/nhãn sai; không tự xóa ảnh hoặc sửa nhãn.
- Giữ nguyên preprocessing/model hiện có, không sửa train để ép kết quả theo EDA.
- Đã đối chiếu split với data.build_loaders: mỗi lớp 5.400/600, phủ đủ 60.000, không giao nhau.
- Đã kiểm tra quantile hữu hạn/thứ tự, ID/nhãn/pixel thật, RMSE và mẫu PCA chỉ thuộc split tương ứng.
- Edge qua HTTP cục bộ: bốn PNG tải được, lọc split/nhãn đúng, ảnh render, không có JS exception.
- Kiểm tra này xác nhận hành vi HTTP cho Live Server; không tự điều khiển extension Live Server.

Tiếp theo sau khi người dùng xem phần này: [training](04_training.md).
