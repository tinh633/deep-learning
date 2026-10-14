# test.py

- Chạy: `python practice_1_fashionmnist/src/test.py [lan_N] [checkpoint]`.
- Mặc định lan_10/best.pt (mô hình bàn giao); truyền tham số để chọn run/checkpoint khác phù hợp kiến trúc hiện tại.
- Tải state_dict bằng model.py hiện tại; kiến trúc và transform/split phải khớp lúc train.
- Dùng eval/no_grad, đánh giá cùng trọng số trên 6.000 validation và 10.000 test; không huấn luyện.
- CrossEntropyLoss trung bình theo mẫu, accuracy và chênh lệch Test − Validation (điểm phần trăm).
- Append test.log trong thư mục run, không ghi đè lịch sử.
- Người dùng yêu cầu đánh giá test ở bước này. Sau khi xem kết quả, không dùng test để tối ưu/chọn hyperparameters; nếu tiếp tục chỉnh model dựa trên test thì kết quả test không còn là đánh giá cuối độc lập.
- Script chỉ hỗ trợ checkpoint state_dict của train.py hiện tại, không dict checkpoint phiên bản cũ.
