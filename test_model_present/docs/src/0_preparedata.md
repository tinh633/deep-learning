# 0_preparedata.py — Chuẩn bị bộ ảnh đánh giá

Source: [0_preparedata.py](../../src/0_preparedata.py).
Cấu hình: [config.md](config.md). Bước sau: [1_benchmark.md](1_benchmark.md).

## Mục đích và điều kiện chạy

Lấy 10 lớp × 50 ảnh validation ImageNet-1K có nhãn để các model đánh giá cùng dữ liệu.
Script không train, không tạo split mới và không thay bằng Imagenette/ImageNetV2.
Chỉ chạy khi đã có quyền truy cập dataset và đăng nhập Hugging Face CLI local.
Đăng nhập website đơn thuần không bảo đảm Python có token và quyền tải.

Từ thư mục workspace:

```powershell
.venv/Scripts/python.exe test_model_present/src/0_preparedata.py
```

Hướng dẫn cấp quyền và cài thư viện nằm trong [README](../../README.md).
Token không được lưu trong source, log hoặc gửi vào chat.

## Đầu vào

- Các biến trong config.py: nguồn, danh sách index/tên lớp, số ảnh/lớp và seed.
- Token local có quyền đọc ILSVRC/imagenet-1k.
- Revision hiện tại của dataset được tra qua HfApi và cố định cho lần streaming đó.

Script lấy revision mới khi chạy lại; muốn tái lập đúng dữ liệu cũ phải giữ manifest/metadata
và đối chiếu revision, không mặc định mọi lần tải đều dùng cùng revision.

## Luồng xử lý

1. Kiểm tra datasets/huggingface_hub và token; thiếu thì dừng với hướng dẫn.
2. Đối chiếu tên/index 10 lớp với categories của torchvision ResNet50 weights.
3. Tra revision nguồn, mở validation dạng streaming và tắt decode tự động.
4. Kiểm tra nguồn có 1.000 nhãn và tên các lớp khớp mapping torchvision.
5. Duyệt record, chỉ giữ lớp đã chọn; dùng bytes ảnh gốc thay vì encode lại.
6. PIL verify kiểm tra ảnh đọc được; SHA-256 kiểm tra bytes trùng trong mẫu đã chọn.
7. Ghi ảnh theo lớp, lưu ID gắn với vị trí record validation và nhãn ImageNet gốc.
8. Dừng khi đủ 50 ảnh của mỗi lớp; thiếu hoặc không đúng tổng 500 thì báo lỗi.
9. Xáo danh sách bằng seed 42; ghi manifest qua file tạm rồi replace.
10. Ghi metadata về nguồn, revision, phân bố, seed, cách chọn và hash manifest.

Streaming vẫn có thể phải đọc gần hết 50.000 record để tìm đủ các lớp.
Không đồng nghĩa chỉ truyền qua mạng đúng dung lượng của 500 ảnh được lưu.
Không chọn/lọc ảnh dựa trên kết quả dự đoán của model.

## Đầu ra

| Đường dẫn dưới data/imagenet1k/ | Nội dung |
|---|---|
| images/<label>/validation_<index>.JPEG | Bytes ảnh gốc, label là index ImageNet |
| manifest.csv | Danh sách 500 ảnh dùng chung |
| dataset_metadata.json | Thông tin truy xuất nguồn và cách lấy mẫu |

Manifest gồm:

| Cột | Ý nghĩa |
|---|---|
| id | imagenet1k/validation/<index>, ID record nguồn |
| path | Đường dẫn ảnh tương đối với thư mục manifest |
| label | Index lớp ImageNet trong khoảng 0–999 |
| split | validation |
| image_sha256 | Hash bytes ảnh để phát hiện thay đổi |

SHA-256 là dấu vân tay nội dung để đối chiếu file, không phải chỉ số chất lượng model.
Hash manifest CSV ở metadata khác hash manifest JSON của một run vì dạng lưu khác nhau.

## Hành vi khi lỗi và giới hạn

- Thiếu token/quyền, mapping sai, ảnh hỏng/trùng hoặc thiếu mẫu: dừng, không thay nguồn ngầm.
- Ảnh đã ghi trước lỗi có thể còn trên đĩa; benchmark chỉ dùng manifest hợp lệ đã hoàn tất.
- Chạy lại có thể ghi đè ảnh cùng đường dẫn và manifest; đây không phải công cụ resume download.
- Đây là validation ngoài train, nhưng có thể đã được dùng khi phát triển pretrained model.
- Mẫu 10 lớp không đại diện đầy đủ ImageNet-1K và không chứng minh ảnh hoàn toàn chưa từng được kiểm tra.

## Bằng chứng thực chạy

Lượt lan_1 đã dùng 500 ảnh duy nhất, đủ 50/lớp, revision được lưu trong metadata/run artifact.
Các kiểm tra manifest, bytes ảnh và mapping đã ghi ở [progress](../../progress.md).
Tài liệu này mô tả source hiện tại; không tải lại hoặc thay dataset khi bổ sung tài liệu.
