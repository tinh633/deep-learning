# Practice 1 — FashionMNIST MLP

Người tiếp tục dự án: đọc [tiến độ và yêu cầu bàn giao](progress.md) trước khi chỉnh sửa hoặc chạy thử nghiệm.

## Các bước

1. [Define problem](docs/01_problem.md)
2. [Data preparation](docs/02_data_prepare.md) — `python practice_1_fashionmnist/src/1_preparedata.py`
3. [EDA](docs/03_eda.md) — `python practice_1_fashionmnist/src/2_eda.py`
4. [Training](docs/04_training.md) — `python practice_1_fashionmnist/src/3_train.py`
5. [Evaluation](docs/05_evaluation.md) — chạy `4_evaluate.py`, sau đó `5_report.py` trong src.

## Cấu trúc

```text
src/
  config.py          tham số và đường dẫn
  model.py           class MLP và forward
  logger.py          tạo run, ghi log
  data.py            Dataset/DataLoader dùng chung
  metrics.py         đánh giá/đọc checkpoint dùng chung
  1_preparedata.py    tải và xác minh dữ liệu
  2_eda.py           EDA cơ bản và EDA sâu
  3_train.py         train và validation
  4_evaluate.py      đánh giá checkpoint
  5_report.py        JSON và biểu đồ model
docs/                5 bước làm bài
visualization/explorer.html
data/FashionMNIST_raw/
outputs/             EDA và kết quả chạy
```

Cài requirements.txt, chạy lệnh từ gốc workspace.
Mở HTML bằng Live Server từ workspace/project root, cuộn đến **EDA sâu** để xem phân tích mới.
Script 1–5 chạy tuần tự từ trên xuống, không main/if __name__; không import các script chạy.
data.py và metrics.py là hàm dùng chung, import không kích hoạt tải dữ liệu/train/đánh giá.
Run đang được chọn để báo cáo là lan_12 mới: train 96,19%, val 91,73%, test 90,17% (best epoch 48).
Xem [đánh giá 12 run và phân tích lỗi](docs/05_evaluation.md), kèm CSV/JSON và ảnh sai từng run.
Đọc [báo cáo thực nghiệm qua các run](docs/06_run_report.md): kiến trúc, tham số, kết quả và so sánh baseline/rộng/sâu.
HTML có bảng thí nghiệm, metrics theo lớp và ảnh đúng/sai; chưa tự chốt model cuối.
Chưa commit/push thay đổi này.

## CNN bổ sung

Mở [cnn_basic.ipynb](cnn_basic.ipynb) trong VS Code, chọn Python kernel có requirements của project,
rồi chạy các cell từ trên xuống. Đây là notebook bổ sung theo yêu cầu; pipeline MLP vẫn dùng src/*.py.
CNN có hai khối Conv/ReLU/Pool, backbone trả feature map 32×7×7, rồi head Linear 64 → 10.
Mặc định 50 epoch, cùng split/optimizer/scheduler với MLP 10–12; kết quả ở outputs/cnn_runs/lan_N,
không ghi đè MLP runs. Mỗi lần chạy lại cell train khởi tạo model và thư mục mới.
Notebook có bảng so sánh MLP, loss/accuracy, confusion, ảnh sai và feature maps.
Test mặc định tắt (RUN_TEST=False); chỉ bật sau khi xem validation và quyết định đánh giá checkpoint.
Đã kiểm tra toàn bộ code cell với 1 epoch trên 16 ảnh tổng hợp và dữ liệu thật tải đúng split;
chưa train CNN đầy đủ, chưa có kết luận cải thiện so với MLP.

## CNN new analysis

[cnn_new.ipynb](cnn_new.ipynb), sections 6-11: historical test interpretation, class metrics, ROC/PR and error analysis. Run the analysis cells separately to avoid retraining. Place the original cnn_new_best.pt in the Practice 1 folder; predictions are cached without image pixels in outputs/cnn_new_test.npz. Current checkout lacks that checkpoint, so real per-class results are not yet available. MLP remains presented on the existing HTML.
