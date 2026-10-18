# Practice 1 — FashionMNIST MLP

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
Run đang được chọn để báo cáo là lan_12: train 96,13%, val 91,40%, test 90,05%.
Xem [đối chiếu lab, repo tham khảo và báo cáo ngắn](docs/05_evaluation.md).
HTML có bảng thí nghiệm, metrics theo lớp và ảnh đúng/sai; chưa tự chốt model cuối.
Chưa commit/push thay đổi này.
