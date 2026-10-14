# Practice 1 — FashionMNIST

Phân loại ảnh FashionMNIST bằng MLP (PyTorch). Phạm vi hiện tại: chuẩn bị dữ liệu, trang xem dữ liệu bằng Live Server, huấn luyện, đánh giá validation/test và trực quan hóa.

**Mô hình bàn giao:** `outputs/runs/lan_10/best.pt` — MLP 784→256→128→10, epoch 12, validation 90,57%, test 89,41%. Chọn theo yêu cầu của người dùng; `lan_9` có validation loss thấp hơn. Xem [04_result.md](docs/04_result.md), [05_experiments.md](docs/05_experiments.md) (so sánh 10 lần train, đối chiếu đề bài) và [01_problem.md](docs/01_problem.md).

## Cài đặt

`pip install -r requirements.txt` (NumPy, PyTorch, torchvision, scikit-learn, matplotlib). Chạy mọi lệnh từ thư mục workspace (cha của `practice_1_fashionmnist`).

## Quy trình chạy

| Bước | Lệnh | Tài liệu |
|---|---|---|
| 1. Tải, kiểm MD5, giải nén, EDA | `python practice_1_fashionmnist/src/prepare_data.py` | [prepare_data.md](docs/scripts/prepare_data.md) |
| 2. Kiểm tra pipeline (54.000/6.000/10.000) | `python practice_1_fashionmnist/src/data_pipeline.py` | [data_pipeline.md](docs/scripts/data_pipeline.md) |
| 3. Kiểm tra forward của MLP | `python practice_1_fashionmnist/src/model.py` | [model.md](docs/scripts/model.md) |
| 4. Huấn luyện (tạo `outputs/runs/lan_N/`) | `python practice_1_fashionmnist/src/train.py` | [train.md](docs/scripts/train.md) |
| 5. Đánh giá validation | `python practice_1_fashionmnist/src/evaluate.py [lan_N] [checkpoint]` | [evaluate.md](docs/scripts/evaluate.md) |
| 6. Đánh giá validation + test | `python practice_1_fashionmnist/src/test.py [lan_N] [checkpoint]` | [test.md](docs/scripts/test.md) |
| 7. Báo cáo model cho HTML | `python practice_1_fashionmnist/src/model_report.py [lan_N]` | [model_report.md](docs/scripts/model_report.md) |
| 8. Biểu đồ đánh giá | `python practice_1_fashionmnist/src/visualize_metrics.py [lan_N]` | [visualize_metrics.md](docs/scripts/visualize_metrics.md) |
| 9. Ảnh dự đoán và nhãn thật | `python practice_1_fashionmnist/src/show_predictions.py [lan_N]` | [show_predictions.md](docs/scripts/show_predictions.md) |

Bước 5 đến 9 mặc định dùng `best.pt`; bước 6 đến 9 mặc định chạy `lan_10`. Mỗi lần train tạo thư mục `lan_N` mới và giữ nguyên run cũ. Test chỉ dùng để báo cáo sau khi đã chọn cấu hình bằng validation.

## Mở trang xem dữ liệu và đánh giá model

1. Mở workspace `D:\newtask\DL` trong VS Code, không chỉ thư mục visualization.
2. Mở `practice_1_fashionmnist/visualization/explorer.html`.
3. Chuột phải → **Open with Live Server**.
4. Đợi thông báo tải chuyển thành `70.000 ảnh phù hợp`.

URL mặc định: `http://127.0.0.1:5500/practice_1_fashionmnist/visualization/explorer.html`; dùng URL Live Server đã mở nếu cổng/thư mục gốc khác. Nhấn Ctrl+F5 nếu đang chạy. Cần HTTP và JavaScript; không hỗ trợ `file://` và Preview của editor.

- Lọc theo nhãn và train/test/all, số lượng cập nhật ngay; cuộn hoặc nhảy tới ảnh bất kỳ, bấm để xem chi tiết; EDA có bộ chọn tập riêng.
- Khoảng 55 MB dữ liệu được tải một lần mỗi lần mở trang; lọc chạy cục bộ và chỉ vẽ các hàng gần vùng nhìn.
- Phần **Đánh giá model** đọc `outputs/runs/lan_10/model_report.json`; thêm `?run=lan_N` vào URL để xem run khác (hiện có sẵn `lan_9` và `lan_10`; run khác cần chạy `model_report.py lan_N` trước). Xem [explorer.md](docs/scripts/explorer.md).
- Lỗi được hiển thị trong trang kèm hướng dẫn.

## Cấu trúc và kiểm chứng

- `src/`: chỉ các script Python ở bảng trên.
- `visualization/explorer.html`: một trang HTML (UI, CSS và JavaScript).
- `data/FashionMNIST_raw/`: file nén gốc và IDX giải nén (không đưa vào Git).
- `outputs/eda.json`, `outputs/browser_preview.png`: thống kê EDA và ảnh chụp kiểm thử trình duyệt.
- `outputs/runs/lan_N/`: log, checkpoint, `model_report.json` và `plots/` của từng lần train.
- Không cần Node hoặc file .cjs để dùng dự án.

Dữ liệu: 60.000 train và 10.000 test, mười lớp cân bằng; bốn MD5 khớp. Kiểm thử Edge qua HTTP cục bộ đạt phần ảnh/EDA, lọc nhãn/test và xem ảnh cuối mà không có lỗi JavaScript; chính extension Live Server chưa được tự động hóa. Các script PyTorch (`data_pipeline`, `model`, `evaluate`, `test`, `model_report`) đã chạy lại trên CPU và cho kết quả khớp báo cáo; trang đã được kiểm tra bằng jsdom với `?run=lan_10` và `?run=lan_9`.

Nguồn: [Zalando Research FashionMNIST](https://github.com/zalandoresearch/fashion-mnist). Link tải trực tiếp và checksum: [data_source.md](docs/data_source.md). Thiết kế: [design.md](docs/design.md), [02_data_preparation.md](docs/02_data_preparation.md), [03_model.md](docs/03_model.md).
