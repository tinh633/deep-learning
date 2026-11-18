# Exercise 2 · Fine-tuning trên IMDb

Fine-tune DistilBERT để phân loại review tiếng Anh thành `NEGATIVE` hoặc `POSITIVE`. Thực hiện từ thư mục của workflow bằng Python 3.11 hoặc 3.12 trong môi trường ảo. Lần đầu cần truy cập Hub để tải IMDb, tokenizer, checkpoint và metric accuracy của Evaluate.

```bash
python -m pip install -r requirements.txt
python src/1_preparedata.py
python src/2_eda.py
python src/3_train.py
python src/4_evaluate.py --run lan_1
python src/5_report.py --run lan_1
```

Dùng tên `lan_n` vừa được in khi train nếu đã có các lần chạy khác. Mask, padding, snapshot dữ liệu và cách chọn checkpoint được giải thích trong các tài liệu dưới đây. Khi nhóm đã chốt mô hình trên validation, chạy lệnh test trong [Training.md](Training.md).

| Muốn xem | Vị trí |
| --- | --- |
| Hướng dẫn lệnh, TensorBoard, lỗi thường gặp | [Training.md](Training.md) |
| Cấu hình baseline và cách so sánh | [Experiment_Plan.md](Experiment_Plan.md) |
| Dữ liệu, giới hạn độ dài, dynamic padding | [Dataset_and_Preprocessing.md](docs/Dataset_and_Preprocessing.md) |
| Chọn checkpoint, confusion matrix, cách đọc kết quả | [Evaluation_Protocol.md](docs/Evaluation_Protocol.md) |
| Hỏi đáp giảng viên | [Model_QA.md](docs/Model_QA.md) |
| Notebook đọc kết quả của run | [Exercise_2_Report.ipynb](Exercise_2_Report.ipynb) |
| Nhật ký AI | [Theo bước](AI_Usage_Step.md) · [Chi tiết](docs/AI_Usage_Full.md) |
