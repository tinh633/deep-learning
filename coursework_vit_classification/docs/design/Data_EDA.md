# Workflow dữ liệu và EDA

## Vai trò file

| File | Trách nhiệm |
|---|---|
| src/config.py | Đường dẫn project/data/output, seed và kích thước validation |
| src/data.py | Tải CIFAR-100 qua torchvision, tạo hoặc đọc split phân tầng |
| src/1_eda.py | Thực hiện tải, thống kê và xuất dữ liệu cho viewer |
| visualization/explorer.html | Đọc JSON và RGB gzip local, trình bày EDA/gallery |

config/data không tự tải hoặc chạy EDA khi import. Chưa triển khai DataLoader train.

## Nguồn và toàn vẹn

- [Trang CIFAR chính thức](https://www.cs.toronto.edu/~kriz/cifar.html).
- Archive: https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz.
- torchvision kiểm tra MD5 archive và các batch khi tải/nạp; không bỏ kiểm tra integrity.
- MD5 archive công bố trong loader torchvision: eb9058c3a382ffc7106e4002c42a8d85.
- Lần tải 05/11/2026 dùng [mirror archive](https://huggingface.co/datasets/nakroy/cifar100-python/blob/main/cifar-100-python.tar.gz)
  do nguồn chính thức chậm; đã đối chiếu MD5 chính thức trước khi giải nén bằng torchvision.
- Script mặc định vẫn dùng URL chính thức, không tự thay nguồn hoặc bỏ kiểm tra checksum.
- Nhãn dùng fine labels 100 lớp; chưa đưa coarse labels 20 nhóm vào phân tích.

## Split

- Official train 50.000 ảnh: train 45.000 và validation 5.000, seed 42.
- Mỗi lớp có 450 train, 50 validation; official test có 100 ảnh/lớp.
- IDs trong cifar100_split.json tham chiếu official train; test dùng ID riêng 0–9999.
- Kiểm tra đủ IDs, không trùng, kích thước và cân bằng khi nạp split đã lưu.
- SHA-256 file split được ghi vào eda.json để đối chiếu các lần chạy.

## Nội dung khảo sát

- Phân bố số ảnh từng lớp theo split; gallery lọc đủ 100 lớp.
- Mean/std và histogram RGB tính trên train sau chia 255.
- Brightness và contrast tính từng ảnh; hiển thị min/Q1/median/Q3/max theo split.
- Train có thêm mean brightness theo lớp trong report.
- SHA-256 pixel phát hiện trùng chính xác; báo nhóm trùng trong/giữa split, không tự xóa.
- PCA: mean pooling ảnh xuống 8 × 8 RGB (192 chiều), fit PCA 32 chiều trên 45.000 train.
- Hiển thị 2 PC đầu cho đủ 60.000 ảnh; giữ cùng hệ trục cho train/validation/test.
- t-SNE: 5.000 ảnh chỉ từ train, 50 ảnh/lớp, seed 42; đầu vào là PCA 32 chiều.
- Cấu hình t-SNE: perplexity 30, 750 iterations, init PCA, learning_rate auto.
- Nhãn chỉ dùng lấy mẫu cân bằng/tô màu; không tham gia tính đặc trưng hoặc tọa độ.
- projection.json lưu tọa độ và thông số; PCA giữ khoảng 50,45% phương sai ở 2D trong lượt hiện tại.
- Không đánh giá near-duplicate, sai nhãn hoặc độ phù hợp pretrained tự động.
- Thống kê validation/test chỉ mô tả dữ liệu; không dùng để tối ưu preprocessing.

## HTML và dữ liệu local

HTML không dùng CDN. Chạy HTTP server từ workspace theo README, mở Chrome/Edge mới.
Report không nhúng pixel ảnh. Sáu RGB gzip chunk chứa đủ official train/test trong data/.
Gallery tải chunk cần dùng, cache theo phiên, phân trang và lọc lớp/tên/split.
Sắp xếp brightness/contrast giúp xem cực trị; ID nguồn được giữ nguyên.
Biểu đồ có đổi PCA/t-SNE, chọn nhiều lớp, lọc split PCA, giữ nền xám, zoom/kéo/reset và hover ảnh.
Chồng lấn màu là kết quả thực của đặc trưng RGB; không dịch điểm theo nhãn để làm tách lớp.
t-SNE không hỗ trợ transform test; chỉ mô tả mẫu train, không dùng khoảng cách giữa cụm làm metric.
Đây chưa phải không gian embedding của model; phân tích embedding CNN/ViT thuộc giai đoạn sau nếu cần.
Viewer không có predictions, TensorBoard hoặc ảnh lỗi test khi chưa huấn luyện.
Các giá trị thống kê không được tự động dùng thay ImageNet normalization.

## Tái lập và quản lý Git

Chạy lại 1_eda.py dùng dataset local đã kiểm tra, giữ split và tái sinh report/chunks.
Không commit data/ hoặc RGB gzip. outputs/ chỉ chứa thống kê và chỉ số, không có pixel.
Không tạo checkpoint hoặc run train giả trong giai đoạn EDA.
