# Báo cáo các lần thực nghiệm MLP trên FashionMNIST

## 1. Mục tiêu và thiết lập chung

Thực nghiệm nhằm đánh giá ảnh hưởng của learning rate, dropout, weight decay, độ rộng và độ sâu MLP.
Ảnh xám 28×28 được ToTensor đưa về [0,1], sau đó flatten thành vector 784 phần tử.
Các lớp ẩn dùng Linear → ReLU → Dropout; lớp cuối có 10 logits, dùng CrossEntropyLoss.
Tất cả run dùng Adam, batch size 64, seed 42 và StepLR với gamma 0,3.
Dữ liệu gồm train 54.000, validation 6.000 chia stratified và test chính thức 10.000 ảnh.

Mỗi run lưu best.pt theo validation accuracy cao nhất, hòa thì chọn validation loss thấp hơn.
Các kết quả dưới đây đo lại **cùng best.pt ở eval mode**, dropout tắt; không lấy train accuracy online.
Kiến trúc được xác định từ trọng số, vì config cũ của lần 5/6 thiếu tham số lớp ẩn thứ ba.
Lần 3 và 5 chưa chạy hết epoch; các run này chỉ được báo cáo kết quả hiện có.

## 2. Thiết kế các lần chạy

Lớp ẩn trong bảng chưa bao gồm đầu vào 784 và đầu ra 10. Step là số epoch giữa hai lần giảm LR.

| Run | Kiến trúc lớp ẩn | Dropout | LR đầu | Weight decay | Step | Epoch thực tế / dự kiến | Mục đích đối chiếu |
|---|---|---:|---:|---:|---:|---:|---|
| 1 | 256–128 | 0 | 0,001 | 0 | 10 | 30/30 | Mốc MLP hai lớp ẩn |
| 2 | 256–128 | 0,2 | 0,001 | 0 | 10 | 30/30 | Thêm dropout so với 1 |
| 3 | 256–128 | 0,1 | 0,003 | 0 | 10 | 11/30 | Thay dropout và LR; chưa hoàn tất |
| 4 | 256–128 | 0 | 0,003 | 0 | 10 | 30/30 | Tăng LR so với 1 |
| 5 | 256–128–64 | 0 | 0,003 | 0 | 10 | 29/30 | Thêm lớp ẩn so với 4; chưa hoàn tất |
| 6 | 256–128–64 | 0 | 0,001 | 0 | 15 | 30/30 | Giảm LR, đổi lịch giảm LR so với 5 |
| 7 | 256–128–64 | 0 | 0,001 | 0,0001 | 15 | 30/30 | Thêm weight decay so với 6 |
| 8 | 256–128–64 | 0,1 | 0,001 | 0 | 15 | 30/30 | Thêm dropout so với 6 |
| 9 | 256–128–64 | 0,2 | 0,001 | 0 | 25 | 50/50 | Tăng dropout/epoch, đổi lịch LR |
| **10** | **256–128–64** | **0,2** | **0,001** | **0,00001** | **15** | **50/50** | **Baseline cho so sánh rộng/sâu** |
| **11** | **512–256–128** | **0,2** | **0,001** | **0,00001** | **15** | **50/50** | **Mở rộng so với 10, giữ ba lớp ẩn** |
| **12** | **512–256–128–64** | **0,2** | **0,001** | **0,00001** | **15** | **50/50** | **Tăng sâu so với 11, thêm lớp 64** |

## 3. Kết quả thu được

Gap là train accuracy trừ val accuracy, tính bằng điểm phần trăm. Dấu — là chưa đo test.

| Run | Best epoch | Train acc % | Train loss | Val acc % | Val loss | Sai val / 6.000 | Gap | Test acc % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 24 | 96,15 | 0,108480 | 91,18 | 0,295720 | 529 | 4,96 | — |
| 2 | 24 | 94,03 | 0,156939 | 91,00 | **0,264423** | 540 | 3,03 | — |
| 3 | 11 | 91,75 | 0,217550 | 89,75 | 0,280489 | 615 | 2,00 | — |
| 4 | 22 | 95,41 | 0,120175 | 90,62 | 0,340728 | 563 | 4,80 | — |
| 5 | 24 | 95,89 | 0,107565 | 90,72 | 0,378627 | 557 | 5,18 | — |
| 6 | 22 | 95,76 | 0,110802 | 91,03 | 0,312959 | 538 | 4,73 | — |
| 7 | 26 | 95,51 | 0,124792 | 90,85 | 0,300252 | 549 | 4,66 | — |
| 8 | 22 | 94,46 | 0,145072 | 90,97 | 0,278345 | 542 | 3,49 | — |
| 9 | 41 | 95,34 | 0,120329 | 91,05 | 0,287951 | 537 | 4,29 | 89,97 |
| **10** | **39** | **95,15** | **0,128022** | **91,33** | **0,277989** | **520** | **3,82** | — |
| **11** | **50** | **96,78** | **0,086345** | **91,48** | **0,308205** | **511** | **5,30** | **90,05** |
| **12** | **48** | **96,19** | **0,100915** | **91,73** | **0,299211** | **496** | **4,46** | **90,17** |

## 4. Đánh giá quá trình thử tham số

**Lần 1–2:** dropout 0,2 làm train accuracy giảm nhưng gap và val loss giảm; val sai thêm 11 ảnh.
Điều này cho thấy regularization cải thiện loss nhưng chưa tăng accuracy trong phép thử này.

**Lần 3–5:** lần 3 chưa hoàn tất và đổi hai tham số nên chưa tách được tác động.
Lần 4 tăng LR lên 0,003 so với lần 1: val sai thêm 34 ảnh và loss tăng 0,0450.
Lần 5 thêm lớp 64 so với lần 4: giảm 6 lỗi nhưng loss cao hơn, lại chưa chạy đủ epoch.

**Lần 6–8:** lần 6 đổi cả LR và scheduler nên không quy cải thiện cho riêng một yếu tố.
So với 6, lần 7 thêm weight decay làm val sai thêm 11 ảnh dù loss giảm.
Lần 8 dùng dropout 0,1: gap giảm từ 4,73 xuống 3,49 điểm nhưng val vẫn sai thêm 4 ảnh.

**Lần 9–10:** lần 9 đổi dropout, số epoch và scheduler cùng lúc; không phải thử nghiệm một biến.
Lần 10 giảm 17 lỗi val và loss thấp hơn lần 9, nhưng cùng đổi weight decay và lịch LR.
Lần 10 được dùng làm baseline cho chuỗi kiểm tra độ rộng/độ sâu tiếp theo.

## 5. Baseline, mở rộng và tăng sâu

**Lần 11 — mở rộng:** tăng 256–128–64 thành 512–256–128, giữ số lớp ẩn và các tham số train.
So với baseline 10, train accuracy tăng 1,63 điểm nhưng val chỉ tăng 0,15 điểm (9 ảnh).
Val loss tăng 0,0302 và gap tăng từ 3,82 lên 5,30 điểm: học train tốt hơn chưa chuyển thành cải thiện tương xứng trên val.

**Lần 12 — tăng sâu:** giữ ba lớp của lần 11 và thêm lớp 64 trước đầu ra.
Val tăng 0,25 điểm so với 11 (15 ảnh), loss giảm 0,0090 và gap thu hẹp còn 4,46 điểm.
So với baseline 10, val tăng 0,40 điểm (24 ảnh) nhưng val loss vẫn cao hơn 0,0212.
Tăng sâu có lợi về accuracy trong chuỗi này; chưa có bằng chứng cải thiện đồng thời mọi chỉ tiêu.

Chuyển 10 → 11 sửa đúng 117 ảnh val nhưng làm sai thêm 108; chuyển 11 → 12 sửa 114 và sai thêm 99.
Vì vậy số lỗi giảm ròng không có nghĩa mọi ảnh hoặc mọi lớp đều tốt hơn.

## 6. Phân tích lỗi và kết quả test đã có

Lỗi lặp lại chủ yếu ở Shirt ↔ T-shirt/top và Coat ↔ Pullover; có 217 ảnh val sai ở cả 12 run.
Lần 12 còn nhầm Shirt → T-shirt 60 ảnh val và Coat → Pullover 53 ảnh val.
Ảnh xám 28×28 và hình dáng áo tương tự là nguyên nhân khả dĩ; không đủ căn cứ kết luận nhãn gốc sai.

Trên test, lần 9/11/12 đạt 89,97% / 90,05% / 90,17%; loss là 0,343221 / 0,360313 / 0,357034.
Lần 12 hơn 11 chỉ 12 ảnh test; recall Shirt lại giảm từ 71,5% xuống 70,6%.
Lần 10 chưa đo test nên không khẳng định 11/12 tốt hơn baseline trên test.
Chi tiết confusion, precision/recall/F1 và ảnh sai xem [phân tích lỗi](05_evaluation.md).

## 7. Nhận định hiện tại và giới hạn

Lần 12 có val accuracy cao nhất; lần 2 có val loss tại best.pt thấp nhất; lần 11 có train accuracy cao nhất.
Mục tiêu train >96% đã đạt ở một số run; mục tiêu val/test >95% chưa đạt.
Khoảng cách train–val và các lỗi nhóm áo vẫn tồn tại; chưa thể kết luận hết overfitting.
Mỗi cấu hình mới có một seed; chênh lệch nhỏ chưa chứng minh ưu thế ổn định.
Test đã được xem nhiều lần; không dùng các kết quả test này để tiếp tục chọn tham số.
Chưa chốt model cuối; CNN cơ bản là phần đối chiếu tiếp theo đã dự kiến, chưa có kết quả trong báo cáo này.

Notebook bổ sung đã tạo tại [cnn_basic.ipynb](../cnn_basic.ipynb): hai khối Conv/ReLU/Pool,
feature maps 28×28 → 14×14 → 7×7, head 64 → 10, tổng 105.866 tham số.
Cùng split và tham số train của 10–12; đầu ra riêng outputs/cnn_runs/lan_N. Chưa chạy đủ 50 epoch.
Đã kiểm tra luồng train/lưu/nạp/so sánh/biểu đồ bằng 16 ảnh tổng hợp; không dùng làm kết quả chất lượng.

Nguồn số liệu: [CSV](../outputs/run_review.csv), [JSON/checkpoint hash](../outputs/run_review.json), best.pt và train.log trong từng run.
Xem [HTML so sánh](../visualization/explorer.html), [confusion các run](../visualization/run_confusion.png) và [ảnh sai](../visualization/run_errors.png).
