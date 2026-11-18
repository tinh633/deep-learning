# Dataset, chia tập và tokenization

## Nguồn và chia tập

[IMDb trên Hugging Face](https://huggingface.co/datasets/stanfordnlp/imdb) chứa review phim tiếng Anh có hai nhãn. Bài dùng `0 = NEGATIVE` và `1 = POSITIVE`. Script lấy mẫu phân tầng từ train chính thức rồi chia thành train/validation. Test lấy từ split test chính thức. Phân tầng giúp tỷ lệ hai nhãn trong các mẫu nhỏ gần với tập nguồn.

Mỗi review giữ `source_index`, là vị trí trong split gốc. Snapshot lưu số lượng, seed, phân bố nhãn, giới hạn token và SHA-256 theo thứ tự các dòng (chỉ số, nhãn, nội dung). Trước train, script kiểm tra train và validation theo hash. Khi evaluate, script so snapshot của run và kiểm tra hash của split đang đánh giá. Chuẩn hóa chữ thường và khoảng trắng để kiểm tra review trùng chính xác xuyên split. Biến thể diễn đạt gần giống vẫn có thể tồn tại.

## Từ review đến batch

1. Tokenizer từ checkpoint chia review thành subword và thêm token đặc biệt.
2. Tính độ dài đầy đủ để đo nguy cơ truncate. Sau đó tạo `input_ids` và `attention_mask` với giới hạn thiết lập, mặc định 128 token.
3. Collator pad tới độ dài lớn nhất **trong batch**. Ví dụ hai review dài 70 và 110 token sẽ được xếp thành hai hàng dài 110. Các vị trí pad mang mask 0.
4. Trainer lấy cột model cần, gồm nhãn, ID và mask. Text gốc, chỉ số và độ dài được giữ trong snapshot để làm EDA và phân tích lỗi.

## Đọc biểu đồ EDA

`label_distribution.png` cho thấy số lượng từng nhãn trong train/validation. `token_lengths.png` dùng độ dài trước khi truncate để thấy đuôi phân phối và mốc 128. `eda_summary.json` ghi median, p90, max và tỷ lệ bị cắt. Nếu nhiều review vượt giới hạn, có thể cần cân nhắc max length lớn hơn khi chạy thí nghiệm kế tiếp. EDA mặc định tập trung vào train/validation để các quyết định preprocessing dựa trên hai tập đó.
