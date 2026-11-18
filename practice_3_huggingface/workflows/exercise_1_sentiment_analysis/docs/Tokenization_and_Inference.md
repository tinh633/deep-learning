# Tokenization và inference

## 1. Từ câu đến input tensor

DistilBERT dùng tokenizer tương ứng với checkpoint. Tokenizer chuẩn hóa câu theo cách tokenizer uncased được cấu hình, chia văn bản thành token hoặc subword, tra từ vựng để lấy ID rồi thêm token đặc biệt. Với câu ví dụ, script in cả `[CLS]` và `[SEP]` trong cùng danh sách với các token còn lại. So sánh token và ID tại **cùng vị trí** trong JSON.

| Trường | Cách đọc khi demo |
| --- | --- |
| `tokens` | Dạng dễ đọc của chuỗi mà mô hình nhận. Một từ có thể sinh nhiều token. |
| `token_ids` | Chỉ số trong vocabulary mà lớp embedding dùng để tra vector. |
| `attention_mask` | Giá trị 1 ở vị trí hợp lệ và 0 ở padding. Câu đơn ở đây thường toàn 1. |
| `max_length` | Giới hạn 512 token tính cả token đặc biệt. Câu dài hơn bị cắt. |

## 2. Từ tensor đến nhãn

`AutoModelForSequenceClassification` đọc checkpoint cùng head SST-2. Sau forward pass, `logits` là hai điểm thô. Softmax tạo hai điểm trong khoảng 0–1 có tổng xấp xỉ 1. Script lấy nhãn của điểm lớn nhất từ ánh xạ `id2label` của checkpoint. `eval()` đưa dropout về chế độ suy luận, còn `torch.inference_mode()` bỏ việc theo dõi gradient cho lần chạy này.

## 3. Cách trình bày

Mở notebook hoặc JSON, đọc câu mẫu trước. Chỉ một cặp token–ID và giải thích mask, sau đó mở biểu đồ hai điểm softmax. Với câu có cả “started slowly” và “loved its ending”, có thể giải thích kết quả là dự đoán chung của mô hình cho **toàn câu**. Hai điểm softmax chỉ mô tả độ tự tin nội bộ của classifier trên đầu vào đó. Đây là demo định tính, độ chính xác cần nhiều mẫu có nhãn.
