# Exercise 1 · Sentiment analysis

Chạy một review mẫu qua DistilBERT đã được fine-tune trên SST-2. Mục tiêu bài này là trình bày luồng **câu → token → mô hình → nhãn**.

Từ thư mục chứa README, dùng Python 3.11 hoặc 3.12 trong môi trường ảo. Nếu dùng Fish, kích hoạt bằng `source .venv/bin/activate.fish` tại thư mục chứa `.venv`.

```bash
python -m pip install -r requirements.txt
python src/1_sentiment.py
python visualization/plot_sentiment.py
```

Thử câu khác bằng `python src/1_sentiment.py --sentence "The acting was flat and the plot made no sense."`. Script lưu lần demo mới nhất vào `outputs/sentiment_result.json`. Sau khi đổi câu, chạy lại script vẽ để biểu đồ phản ánh kết quả mới. Lần đầu cần truy cập Hugging Face Hub.

- [Tokenization và luồng inference](docs/Tokenization_and_Inference.md)
- [Câu hỏi giảng viên có thể hỏi](docs/Model_QA.md)
- [Ghi nhận sử dụng AI theo bước](AI_Usage_Step.md)
- [Nhật ký AI chi tiết](docs/AI_Usage_Full.md)
- [Notebook báo cáo](Exercise_1_Report.ipynb)
