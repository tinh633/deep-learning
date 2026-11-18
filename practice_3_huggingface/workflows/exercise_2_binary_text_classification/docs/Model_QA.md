# Hỏi đáp khi trình bày Exercise 2

**Pretrained nghĩa là gì ở Exercise 2?**  
DistilBERT đã học biểu diễn ngôn ngữ. Lab nạp trọng số encoder đó và khởi tạo head phù hợp với hai nhãn IMDb. Thông báo thiếu trọng số của classifier là kết quả dự kiến khi lấy checkpoint cơ sở. Fine-tuning cập nhật cả encoder và head trong baseline này.

**Head chỉ có hai nhãn thì loss chạy sao?**  
Model xuất hai logits trên mỗi review. Với nhãn nguyên 0/1, model dùng cross-entropy. Ví dụ nhãn đúng là 1 thì việc tăng điểm tương đối cho lớp 1 làm giảm loss. Accuracy dùng argmax, do đó hai mô hình có cùng accuracy vẫn có thể có loss khác nhau.

**Trainer làm gì?**  
Trainer đóng gói việc tạo batch, chạy forward, backward, optimizer, scheduler, logging và validation theo lịch đã cấu hình. Dataset cung cấp dữ liệu đã token hóa, collator pad theo batch, model tính loss, còn hàm metrics tính chỉ số từ logits và nhãn.

**Tại sao thử 3 epoch?**  
Ba epoch là baseline cho subset nhỏ và encoder đã pretrained. Cần xem validation loss và F1 theo epoch để biết có nên train lâu hơn. Chọn checkpoint tốt nhất cuối lượt chạy là model selection. Baseline chạy đủ số epoch được cấu hình.

**Tại sao dùng F1 để chọn?**  
F1 tóm tắt precision và recall của lớp tích cực. Khi trình bày, chỉ rõ lớp dương và đưa cả accuracy với confusion matrix để người nghe thấy lỗi của hai lớp.

**Kết nối với Lab 1 và Lab 2 như thế nào?**  
Lab 1 có vòng lặp forward, loss, backward và optimizer viết bằng PyTorch. Exercise 2 dùng Trainer để tổ chức các bước đó. Lab 2 khởi tạo backbone từ ImageNet rồi thay head cho CIFAR-10. Ở bài này, backbone là encoder ngôn ngữ pretrained và head phục vụ hai nhãn của IMDb.
