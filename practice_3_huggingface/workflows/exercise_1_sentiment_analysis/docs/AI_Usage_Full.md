# Nhật ký sử dụng AI · Exercise 1 — Sentiment Analysis

## 1. Chọn checkpoint và triển khai suy luận

- **AI hỗ trợ** chọn checkpoint `distilbert/distilbert-base-uncased-finetuned-sst-2-english`, dùng tokenizer cùng checkpoint và viết luồng suy luận trong `src/1_sentiment.py`. Script chuyển các ID đầu vào ngược thành token để hiển thị đúng chuỗi mà mô hình nhận.
- **Kiểm chứng** người thực hiện kích hoạt môi trường ảo, chạy script trên máy và nhận được log tải trọng số thành công. ID mô hình xuất trong JSON trùng với checkpoint đã chọn.
- **Bằng chứng** [`1_sentiment.py`](../src/1_sentiment.py) và `outputs/sentiment_result.json`.

## 2. Đọc token, logits và dự đoán của câu mẫu

- **AI hỗ trợ** giải thích vai trò của token đặc biệt, attention mask và softmax.
- **Kiểm chứng** đối chiếu trực tiếp output của câu “The film started slowly, but I absolutely loved its ending.” Chuỗi có 14 token, gồm `[CLS]` (ID 101) và `[SEP]` (ID 102). Attention mask có 14 giá trị `1`, phù hợp với đầu vào một câu không cần padding. Logit của `POSITIVE` (4,4213) cao hơn `NEGATIVE` (−4,1803), nhất quán với nhãn dự đoán **`POSITIVE`** và điểm softmax **99,98%**.
- **Bằng chứng** `outputs/sentiment_result.json` và log terminal của lần chạy. Điểm softmax mô tả độ tự tin của mô hình trên câu mẫu. Để đo accuracy cần đánh giá trên một tập có nhãn.

## 3. Chuẩn bị biểu đồ và nội dung trình bày

- **AI hỗ trợ** viết script đọc JSON để vẽ điểm của hai nhãn và soạn hướng dẫn đọc kết quả.
- **Kiểm chứng** mã tạo biểu đồ đã được kiểm tra với JSON mẫu trong quá trình phát triển. Kết quả thực tế của câu mẫu được xác nhận từ JSON in trên terminal.
- **Bằng chứng** [`plot_sentiment.py`](../visualization/plot_sentiment.py), [`Exercise_1_Report.ipynb`](../Exercise_1_Report.ipynb) và [giải thích tokenization và inference](Tokenization_and_Inference.md).
