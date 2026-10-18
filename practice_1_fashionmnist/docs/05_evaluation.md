# 5. Đánh giá các lần chạy và phân tích lỗi MLP

## Phạm vi và cách đo

Đối chiếu 12 run hiện có ngày 2026-10-18, dùng **best.pt**, không dùng trọng số epoch cuối thay thế.
Đo lại ở eval/no_grad trên train 54.000 và validation 6.000 ảnh, cùng split stratified seed 42.
Kiến trúc lấy từ kích thước trọng số: lan_5/6 có ba lớp ẩn dù config cũ thiếu hidden3.
Cả 12 validation accuracy khớp checkpoint; chênh loss do batch/tính toán dưới 0,000002.
Chỉ đo lại test 10.000 ảnh của lan_9/11/12 vì đã có test.log; không mở test cho các run còn lại.
Accuracy train dưới đây đo lại khi **tắt dropout**, khác accuracy online ghi trong train.log.
lan_3 dừng ở 11/30 và lan_5 ở 29/30 epoch: không coi là thí nghiệm hoàn tất.

## Các cấu hình thực sự đã chạy

Tất cả dùng ToTensor, ReLU, CrossEntropyLoss và Adam; batch 64, seed 42.
StepLR trong bảng là step_size; gamma luôn 0,3. Không có CNN trong các run này.

| Run | Lớp ẩn | Dropout | LR đầu | Weight decay | StepLR | Epoch chạy / dự kiến | Best epoch |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | 256–128 | 0 | 0,001 | 0 | 10 | 30/30 | 24 |
| 2 | 256–128 | 0,2 | 0,001 | 0 | 10 | 30/30 | 24 |
| 3 | 256–128 | 0,1 | 0,003 | 0 | 10 | 11/30 | 11 |
| 4 | 256–128 | 0 | 0,003 | 0 | 10 | 30/30 | 22 |
| 5 | 256–128–64 | 0 | 0,003 | 0 | 10 | 29/30 | 24 |
| 6 | 256–128–64 | 0 | 0,001 | 0 | 15 | 30/30 | 22 |
| 7 | 256–128–64 | 0 | 0,001 | 0,0001 | 15 | 30/30 | 26 |
| 8 | 256–128–64 | 0,1 | 0,001 | 0 | 15 | 30/30 | 22 |
| 9 | 256–128–64 | 0,2 | 0,001 | 0 | 25 | 50/50 | 41 |
| 10 | 256–128–64 | 0,2 | 0,001 | 0,00001 | 15 | 50/50 | 39 |
| 11 | 512–256–128 | 0,2 | 0,001 | 0,00001 | 15 | 50/50 | 50 |
| 12 | 512–256–128–64 | 0,2 | 0,001 | 0,00001 | 15 | 50/50 | 48 |

## Kết quả cùng checkpoint

Gap là train accuracy trừ validation accuracy, đơn vị **điểm phần trăm**.
Đúng validation = 6.000 trừ số sai; số đúng/sai train nằm trong CSV/JSON liên kết cuối trang.

| Run | Train acc % | Train loss | Val acc % | Val loss | Sai val | Gap |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 96,15 | 0,108480 | 91,18 | 0,295720 | 529 | 4,96 |
| 2 | 94,03 | 0,156939 | 91,00 | **0,264423** | 540 | 3,03 |
| 3 | 91,75 | 0,217550 | 89,75 | 0,280489 | 615 | 2,00 |
| 4 | 95,41 | 0,120175 | 90,62 | 0,340728 | 563 | 4,80 |
| 5 | 95,89 | 0,107565 | 90,72 | 0,378627 | 557 | 5,18 |
| 6 | 95,76 | 0,110802 | 91,03 | 0,312959 | 538 | 4,73 |
| 7 | 95,51 | 0,124792 | 90,85 | 0,300252 | 549 | 4,66 |
| 8 | 94,46 | 0,145072 | 90,97 | 0,278345 | 542 | 3,49 |
| 9 | 95,34 | 0,120329 | 91,05 | 0,287951 | 537 | 4,29 |
| 10 | 95,15 | 0,128022 | 91,33 | 0,277989 | 520 | 3,82 |
| 11 | **96,78** | **0,086345** | 91,48 | 0,308205 | 511 | 5,30 |
| 12 | 96,19 | 0,100915 | **91,73** | 0,299211 | **496** | 4,46 |

## Nhận xét từng lần chạy

- **1:** baseline hai lớp; học tốt train nhưng gap gần 5 điểm và 137 lỗi val có confidence ≥0,9.
- **2 so với 1:** dropout 0,2 giảm gap còn 3,03, loss val giảm 0,0313; accuracy giảm 11 ảnh.
- **3:** LR và dropout cùng đổi, lại dừng sớm; không thể quy accuracy thấp cho riêng một tham số.
- **4 so với 1:** LR tăng gấp ba làm val kém 34 ảnh, loss tăng 0,0450 trong lần thử này.
- **5 so với 4:** thêm lớp 64 chỉ giảm 6 lỗi val nhưng loss tăng 0,0379; run chưa hoàn tất.
- **6 so với 5:** val loss/accuracy tốt hơn, nhưng LR và lịch giảm LR cùng đổi; không tách được nguyên nhân.
- **7 so với 6:** weight decay 0,0001 giảm loss nhưng thêm 11 lỗi; gap chỉ giảm nhẹ.
- **8 so với 6:** dropout 0,1 giảm gap 1,23 điểm và loss 0,0346, nhưng thêm 4 lỗi val.
- **9 so với 8:** dropout, số epoch và lịch LR cùng đổi; accuracy hơn 5 ảnh nhưng loss cao hơn.
- **10 so với 9:** giảm 17 lỗi val, loss giảm 0,0100; weight decay và StepLR cùng đổi.
- **11 so với 10:** mở rộng giảm 9 lỗi val nhưng train tăng mạnh, gap 5,30 và val loss tăng 0,0302.
- **12 so với 11:** thêm lớp 64 giảm 15 lỗi val và loss 0,0090; gap thu hẹp nhưng vẫn còn rõ.

Lan_12 là ứng viên theo **val accuracy**, lan_2 có **val loss tại best.pt thấp nhất** trong 12 run.
Hai tiêu chí khác nhau: accuracy không phản ánh mức độ tự tin của các dự đoán sai.
Lan_12 có minimum val loss ở epoch 16 (0,269588), nhưng best accuracy ở epoch 48.
Không còn trọng số epoch 16 nếu không được lưu; không thể dùng log để khôi phục checkpoint đó.

## Lỗi validation của từng run

Mỗi lớp có 600 ảnh val. Mũi tên là **nhãn thật → nhãn dự đoán**; bảng ghi hai cặp sai nhiều nhất.
Recall Shirt là tỷ lệ Shirt được đoán đúng; confidence là softmax, không phải độ tin cậy đã hiệu chuẩn.

| Run | Cặp sai thứ nhất | Cặp sai thứ hai | Recall Shirt % | Sai với confidence ≥0,9 |
|---|---|---|---:|---:|
| 1 | Coat → Pullover: 67 | Shirt → T-shirt: 65 | 75,83 | 137 |
| 2 | Shirt → T-shirt: 62 | Coat → Pullover: 60 | 74,17 | 81 |
| 3 | T-shirt → Shirt: 77 | Coat → Pullover: 65 | 76,00 | 70 |
| 4 | Shirt → T-shirt: 62 | Pullover → Coat: 60 | 74,00 | 169 |
| 5 | Shirt → T-shirt: 60 | Coat → Pullover: 57 | 73,83 | 175 |
| 6 | Shirt → T-shirt: 76 | Pullover → Coat: 53 | 74,50 | 162 |
| 7 | Coat → Pullover: 61 | T-shirt → Shirt: 55 | 75,67 | 152 |
| 8 | Shirt → T-shirt: 85 | Pullover → Coat: 53 | 71,33 | 124 |
| 9 | Shirt → T-shirt: 72 | Pullover → Coat: 49 | 76,00 | 120 |
| 10 | Shirt → T-shirt: 61 | Pullover → Coat: 50 | 76,17 | 109 |
| 11 | Shirt → T-shirt: 67 | Coat → Pullover: 49 | 74,50 | 160 |
| 12 | Shirt → T-shirt: 60 | Coat → Pullover: 53 | 75,50 | 155 |

Có **217 ảnh val sai trong cả 12 run**; tổng hợp có 1.023 ảnh từng sai ở ít nhất một run.
Từ 10 → 11: sửa đúng 117 ảnh nhưng làm sai thêm 108 (giảm ròng 9); 403 ảnh cùng sai.
Từ 11 → 12: sửa đúng 114 nhưng làm sai thêm 99 (giảm ròng 15); 397 ảnh cùng sai.
Do đó tăng kích thước không cải thiện đồng đều trên mọi ảnh/lớp.
Quan sát ảnh minh họa: các nhóm áo có hình dáng/cổ/tay tương tự ở ảnh xám 28×28.
Đây là giải thích khả dĩ, không chứng minh nhãn gốc sai; chưa tự sửa nhãn hoặc loại ảnh.
MLP flatten vẫn nhận toàn bộ pixel nhưng thiếu thiên kiến không gian cục bộ của CNN.

## Test đã được xem: chỉ báo cáo, không dùng để chọn cấu hình

| Run | Test acc % | Test loss | Đúng / Sai | Macro F1 | Recall Shirt % |
|---|---:|---:|---|---:|---:|
| 9 | 89,97 | 0,343221 | 8.997 / 1.003 | 0,8995 | 71,70 |
| 11 | 90,05 | 0,360313 | 9.005 / 995 | 0,9004 | 71,50 |
| 12 | 90,17 | 0,357034 | 9.017 / 983 | 0,9013 | 70,60 |

Lan_12 hơn 11 chỉ 12 ảnh test; Shirt recall lại giảm 9 ảnh dù tổng accuracy tăng.
Lỗi test lan_12: Shirt → T-shirt 106, T-shirt → Shirt 95, Coat → Pullover 91,
Shirt → Pullover 83, Pullover → Coat 78. Shirt chỉ đúng 706/1.000 ảnh.
Ngược lại Trouser/Sandal/Bag có recall lần lượt 98,2%/97,7%/97,8%.
Val-test lan_12 chênh 1,56 điểm; train-val vẫn 4,46: chưa giải quyết triệt để tổng quát hóa.
Chưa có nhiều seed để kết luận chênh lệch nhỏ có ý nghĩa; test đã xem nhiều lần không còn là đánh giá mù.

## Bằng chứng và sử dụng

- [Bảng đầy đủ CSV](../outputs/run_review.csv), [JSON theo run/lớp/ID ảnh sai](../outputs/run_review.json).
- [12 confusion matrices validation](../visualization/run_confusion.png): hàng thật, cột dự đoán.
- [Ảnh sai từng run](../visualization/run_errors.png): mỗi cặp nhầm chọn ảnh sai tự tin nhất, không phải mẫu ngẫu nhiên.
- TP/TN/FP/FN one-vs-rest và precision/recall/F1 từng lớp có trong JSON, cả train/val và test đã đo.
- HTML explorer so sánh baseline 10 / mở rộng 11 / tăng sâu 12 bằng bảng và biểu đồ accuracy; chọn chi tiết cả 12 run để xem confusion và ảnh sai. Mặc định validation.
- Bảng so sánh lỗi có số ảnh sửa đúng/sai thêm; test chưa đo hiển thị —, không giả lập kết quả baseline.
- Chạy lại report hiện tại: `python practice_1_fashionmnist/src/5_report.py`; chọn run trong config.py.
- Tổng hợp 12 run là phân tích checkpoint một lần, không thêm pipeline hay thay đổi code train/model.
- Nguồn ý tưởng báo cáo: [repo LAB1 tham khảo](https://github.com/zombieTDV/Deeplearning_Course_UTH/tree/LAB1-FashionMNIST-Classification); số đo ở đây đều từ project này.

Chưa chốt model cuối. Bước tiếp theo có thể phân tích nhóm áo sâu hơn và đối chiếu CNN cơ bản như kế hoạch;
không chọn kiến trúc mới theo test. Chỉ commit/push khi được yêu cầu.
