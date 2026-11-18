# Nhật ký sử dụng AI · Exercise 2 — Fine-tuning DistilBERT trên IMDb

## 1. Chuẩn bị dữ liệu và kiểm tra EDA

- **AI hỗ trợ** xây dựng script lấy mẫu phân tầng từ IMDb, ghi metadata của snapshot, tokenize và giới hạn đầu vào ở 128 token. AI cũng hỗ trợ viết bước vẽ phân bố nhãn và độ dài review.
- **Kiểm chứng** biểu đồ thực tế có **1.000 review mỗi lớp ở train** và **200 mỗi lớp ở validation**. Biểu đồ độ dài cho thấy nhiều review vượt 128 token. Tỷ lệ bị cắt chính xác có thể tra trong trường `truncated_fraction` của `eda_summary.json`.
- **Bằng chứng** [`1_preparedata.py`](../src/1_preparedata.py), [`2_eda.py`](../src/2_eda.py), `label_distribution.png` và `token_lengths.png` trong `visualization/eda/`.

## 2. Fine-tune và chọn checkpoint

- **AI hỗ trợ** xây dựng pipeline `Trainer`, cấu hình padding theo batch, ghi lịch sử train/validation và chọn checkpoint theo validation F1.
- **Kiểm chứng** người thực hiện chạy `lan_1` đủ ba epoch. Biểu đồ đánh dấu checkpoint được chọn ở **epoch 1**. Từ epoch 1 đến epoch 3, train loss giảm khoảng **0,52 → 0,17**, validation loss tăng khoảng **0,37 → 0,49**, còn validation F1 gần như đi ngang. Checkpoint được chọn đạt **84,25% accuracy** và **85,25% F1** trên 400 review validation.
- **Bằng chứng** [`3_train.py`](../src/3_train.py), `outputs/runs/lan_1/validation_evaluation.json` và `visualization/lan_1/training_curves.png`.

## 3. Đánh giá độc lập trên test

- **AI hỗ trợ** xác định lệnh đánh giá test cho checkpoint đã lưu và đối chiếu số liệu JSON với confusion matrix.
- **Kiểm chứng** người thực hiện chạy đánh giá riêng trên **500 review test IMDb**. Tổng đường chéo trong confusion matrix là **197 + 212 = 409**, khớp với accuracy **81,80%**. Với lớp `POSITIVE`, precision là **80,00%**, recall **84,80%** và F1 **82,33%**. Mô hình đoán nhầm 53 review tiêu cực thành tích cực và 38 review tích cực thành tiêu cực.
- **Bằng chứng** [`4_evaluate.py`](../src/4_evaluate.py), `outputs/runs/lan_1/test_evaluation.json` và `visualization/lan_1/test_evaluation.png`.

## 4. Diễn giải và bàn giao

- **AI hỗ trợ** diễn giải chênh lệch validation–test và chuẩn bị câu trả lời về cách chọn checkpoint, cách đọc confusion matrix và ảnh hưởng có thể có của việc giới hạn đầu vào.
- **Kiểm chứng** accuracy test thấp hơn validation **2,45 điểm phần trăm**. Việc nhiều review vượt 128 token là một giới hạn của cấu hình. Các kết quả này chưa xác định phần nào trong chênh lệch xuất phát từ truncation.
- **Bằng chứng** [`Exercise_2_Report.ipynb`](../Exercise_2_Report.ipynb), [quy trình đánh giá](Evaluation_Protocol.md) và các JSON validation/test của `lan_1`.
