# Đối chiếu đề Practice 3

| Yêu cầu | Vị trí triển khai trong workflow | Bằng chứng |
| --- | --- | --- |
| 1.1 Cài Transformers | Exercise 1 · `requirements.txt` | Lệnh cài đặt trong README |
| 1.2 Dùng sentiment model từ Hub | Exercise 1 · `src/1_sentiment.py` | Model ID trong `sentiment_result.json` |
| 1.3 Tokenize một câu | Exercise 1 · `src/1_sentiment.py` | Token, ID và attention mask từ cùng input tensor |
| 1.4 Phân tích cảm xúc | Exercise 1 · `src/1_sentiment.py` | Hai điểm softmax và nhãn dự đoán |
| 2.1 Cài Transformers, Datasets, Evaluate | Exercise 2 · `requirements.txt` | Hướng dẫn cài và phiên bản trong `config.json` của run |
| 2.2 Tải tập phân loại nhị phân | Exercise 2 · `src/1_preparedata.py` | IMDb và provenance trong `split_info.json` |
| 2.3 Load model và tokenizer pretrained | Exercise 2 · `src/1_preparedata.py`, `src/3_train.py` | Model ID và tokenizer ID |
| 2.4 Tiền xử lý | Exercise 2 · `src/1_preparedata.py`, `src/3_train.py` | Tokenize, truncate và dynamic padding |
| 2.5 Định nghĩa TrainingArguments | Exercise 2 · `src/3_train.py` | Đối số thực tế lưu trong `config.json` |
| 2.6 Tạo Trainer và fine-tune | Exercise 2 · `src/3_train.py` | Log, checkpoint và mô hình đã lưu |
| 2.7 Đánh giá mô hình | Exercise 2 · `src/4_evaluate.py` | Accuracy, precision, recall, F1 và confusion matrix |

## Tài liệu tham khảo

- [Hugging Face · Fine-tune a pretrained model](https://huggingface.co/docs/transformers/en/training#fine-tune-a-pretrained-model)
- [Hugging Face · Sentiment Analysis with Python](https://huggingface.co/blog/sentiment-analysis-python)
- [Hugging Face · Trainer](https://huggingface.co/docs/transformers/main_classes/trainer)
- [Hugging Face · Evaluate quick tour](https://huggingface.co/docs/evaluate/en/a_quick_tour)
- [Kaggle · Sentiment Analysis Using Hugging Face](https://www.kaggle.com/code/gauravduttakiit/sentiment-analysis-using-hugging-face)
- [Kaggle · Fine-tuning BERT for Text Classification](https://www.kaggle.com/code/neerajmohan/fine-tuning-bert-for-text-classification)
