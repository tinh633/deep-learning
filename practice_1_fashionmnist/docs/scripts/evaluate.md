# evaluate.py

- Chạy sau train: `python practice_1_fashionmnist/src/evaluate.py [lan_N] [checkpoint]`.
- Chỉ đánh giá 6.000 mẫu validation, không dùng test và không cập nhật trọng số.
- Không truyền run: chọn lan_N mới nhất có checkpoint; checkpoint mặc định best.pt. Truyền `lan_N` và `model.pt` để đánh giá run cũ hoặc epoch cuối.
- Đọc checkpoint đã chọn, bật model.eval() và torch.no_grad(), tính CrossEntropyLoss trung bình theo mẫu và accuracy.
- Ghi outputs/runs/lan_N/evaluate.log bằng append; nhiều lần đánh giá giữ log cũ, mỗi lần có timestamp.
- Cần giữ model.py cùng kiến trúc và data_pipeline.py cùng cách chia/transform như lúc train. Nếu sửa kiến trúc, phải khôi phục kiến trúc tương ứng trước khi tải trọng số.
- Run cũ chỉ có train.log không thể evaluate vì chưa lưu trọng số; cần train lại. Không dùng model khởi tạo ngẫu nhiên để thay thế.
- Test chỉ dùng sau khi chốt kiến trúc/hyperparameters bằng validation; chưa triển khai bước test trong script này.
