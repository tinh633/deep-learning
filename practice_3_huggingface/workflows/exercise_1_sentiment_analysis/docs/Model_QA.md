# Hỏi đáp khi trình bày Exercise 1

**Vì sao tokenizer phải đi cùng model?**  
Embedding của model học theo đúng thứ tự ID trong vocabulary. Đổi tokenizer tùy ý có thể khiến ID của câu trỏ sang các vector khác với những gì model đã học.

**Mô hình đã học từ đâu?**  
Checkpoint dùng DistilBERT đã được fine-tune bằng nhãn SST-2. Lab chỉ thực hiện forward pass trên câu đưa vào. Mô hình và tokenizer công bố tại [model card](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english).

**Vì sao vẫn có thể sai khi confidence cao?**  
Softmax phân phối điểm tương đối giữa hai lớp. Một câu mỉa mai hoặc nằm ngoài miền dữ liệu có thể cho điểm rất lệch dù nhãn sai. Demo một câu chưa đo calibration.

**Nếu đề hỏi pipeline là gì?**  
`pipeline("sentiment-analysis")` gói các bước tokenize, forward và hậu xử lý. Trong bài này từng bước được gọi trực tiếp để in chính xác tensor đã đưa vào model.

**Khác Exercise 2 ở đâu?**  
Ở đây head đã học nhãn cảm xúc từ trước. Exercise 2 tải checkpoint ngôn ngữ cơ sở, tạo head mới rồi cập nhật trọng số bằng IMDb.
