# HTML trình bày Vision Transformer

## Mục tiêu và nguồn yêu cầu

- Đầu vào: `[012012710503] - Học sâu - CNS_CS1 - ViT.pdf`, 2 trang tại gốc coursework.
- Đầu ra: một trang [presentation.html](../../visualization/presentation.html), tiếng Việt, dùng để đọc/trình chiếu/in PDF.
- Bám đủ10 nội dung: đặt vấn đề, tổng quan, patch, embedding, vị trí/CLS, encoder, attention, head/train, CNN/ViT, demo.
- Có ví dụ bắt buộc224×224, P16, D768; nguồn kiến trúc, mã và dữ liệu được dẫn trong trang.
- Phần chưa có artifact thực được đánh dấu chờ bổ sung; không tạo dự đoán, heatmap hoặc kết quả D giả.

## Thiết kế và phạm vi

- HTML/CSS/JavaScript thuần, một file, không CDN, không backend hoặc dịch vụ hosting.
- Nền xanh đen, chữ tương phản, điểm nhấn xanh sáng; sidebar mục lục, responsive desktop/mobile.
- Có chế độ đọc, trình chiếu14 phần, phím điều hướng và in toàn bộ nội dung qua trình duyệt.
- Canvas cảnh hình học tự dựng minh họa patch và che vùng; không nhúng pixel dataset.
- Patch explorer cho P8/16/32 chỉ tính hình học; checkpoint thực vẫn ViT-B/16, không đổi model từ giao diện.
- Attention8×8 dùng số minh họa chuẩn hóa từng hàng, không lấy từ checkpoint hoặc giả lập inference.
- Top-5 trước/sau che,10–20 ảnh,≥3 phân tích và notebook demo còn thiếu; có vùng nội dung để bổ sung sau.
- Phân biệt demo ImageNet1K gốc với checkpoint CIFAR-100 về tập lớp và preprocessing.
- Không sửa pipeline train/evaluate hoặc mở test mới trong công việc trình bày.

## Kết quả và cập nhật nội dung

- Biểu đồ lấy history thực final A lan_14, B lan_23, C lan_19, mỗi run20 epoch.
- Nhúng snapshot ngày27/11/2026 để file mở độc lập vẫn có biểu đồ và nội dung đầy đủ.
- Nút đọc lại history tải JSON local qua HTTP, kiểm tra giá trị hữu hạn; khi lỗi giữ snapshot và báo rõ.
- Snapshot không tự cập nhật khi history thay đổi; khi giao bản mới cần đồng bộ dữ liệu nhúng.
- D chưa có final local; không gán điểm0 hoặc xếp hạng dữ liệu thiếu.
- Không dùng test nguồn C làm kết quả cuối chung; trạng thái khóa bốn checkpoint trước test chưa xác nhận.
- Sơ đồ là vector tự dựng theo paper/mã nguồn; không sao chép ảnh paper, không cần asset ngoài.

## Cách mở

Mở file HTML bằng Edge/Chrome hoặc dùng Live Server. Đọc lại history cần HTTP server:

```powershell
.venv/Scripts/python.exe -m http.server 8000 --bind 127.0.0.1
```

Từ workspace, mở `http://127.0.0.1:8000/coursework_vit_classification/visualization/presentation.html`.
Nhấn Trình chiếu; dùng Trước/Tiếp hoặc phím trái/phải; Escape để thoát.
Phím trái/phải không đổi slide khi focus trong input/select/canvas để giữ tương tác của chúng.
Nhấn In/lưu PDF rồi chọn Save as PDF; kiểm tra bản PDF trước khi dùng làm sản phẩm nộp.
HTML chưa thay thế notebook demo và các ảnh/kết quả phân tích đề bài yêu cầu.

## Kiểm chứng

- Đối chiếu nội dung PDF trích bằng pypdf với các phần HTML và trạng thái source/artifact local.
- Edge headless đạt ở desktop1440px/mobile390px, không tràn ngang hoặc lỗi JavaScript.
- Đạt: đổi patch, chọn bằng phím, flow kiến trúc, attention minh họa, chart, đọc lại history, mask/upload, trình chiếu.
- File mode dùng snapshot và HTTP mode tải JSON đạt; bản in thử15 trang A4 ngang, không trang trắng.
- Công cụ đọc PDF và kiểm tra trình duyệt được cài vào thư mục tạm, không thêm dependency runtime dự án.
- Screenshot/PDF kiểm tra ở thư mục tạm; không đưa pixel dataset hoặc file kiểm thử sinh ra vào Git.
