# Bàn giao tiến độ — FashionMNIST: MLP và CNN

Cập nhật: 2026-10-26. Tài liệu này ghi yêu cầu đã thống nhất để người tiếp theo triển khai từng phần.
Đọc AGENTS.md ở gốc workspace, docs_for_agent và kiểm tra file/Git thực tế trước khi sửa.
Lần bàn giao này chỉ viết kế hoạch/tiến độ; không train, không đổi source và không chốt CNN.

## 1. Mục tiêu và nguyên tắc

- Bài MLP đã chạy xong chuỗi thực nghiệm; không tự chạy lại hoặc xóa các run.
- Dùng CNN cơ bản làm phần bổ sung, thực nghiệm có trình tự để đánh giá khác biệt với MLP.
- **Error analysis là trọng tâm**, phải có số liệu theo lớp, theo ảnh và bằng chứng trực quan.
- Kết quả phải hiện trên một HTML hiện có trong visualization/, có trục và mốc đọc rõ ràng.
- Code đơn giản, dễ đọc; làm một phần có thể review rồi chờ phản hồi trước phần kế tiếp.
- Chọn cấu hình bằng validation; không tinh chỉnh theo test hoặc hứa trước CNN chắc chắn tốt hơn.
- Giữ nguyên dữ liệu/run cũ; không commit/push nếu chưa được yêu cầu riêng.

## 2. Phân vai các run MLP — yêu cầu chính thức

| Nhóm | Run | Vai trò |
|---|---|---|
| Thử nghiệm ban đầu | 1–9 | Khảo sát model, LR, dropout, weight decay và lịch train |
| Baseline đã chốt để đối chiếu | **10** | MLP **256 → 128 → 64** |
| Mở rộng baseline | **11** | MLP **512 → 256 → 128**, vẫn ba lớp ẩn |
| Tăng sâu sau mở rộng | **12** | Giữ **512 → 256 → 128**, thêm lớp **64** |

Baseline chính thức của chuỗi rộng/sâu là **lần 10**, không dùng lần 1 hoặc lần 6 thay cho vai trò này.
Lần 3 dừng 11/30 và lần 5 dừng 29/30 epoch: vẫn báo cáo nhưng phải đánh dấu chưa hoàn tất.
Cấu hình thực tế lấy từ checkpoint/log; một số config cũ thiếu hidden3 dù trọng số có lớp này.

Ba run 10–12 cùng dropout 0,2, Adam lr=0,001, weight_decay=1e-5, 50 epoch,
StepLR(15; gamma=0,3), batch 64, seed 42. Split stratified train/val: 54.000/6.000;
test chính thức 10.000; ToTensor, không augmentation. Best theo val accuracy, hòa thì val loss.

| Run | Train acc % | Val acc % | Val loss | Sai val | Gap train–val | Test acc % |
|---|---:|---:|---:|---:|---:|---:|
| 10 — baseline | 95,15 | 91,33 | 0,277989 | 520 | 3,82 điểm | Chưa đo |
| 11 — rộng | 96,78 | 91,48 | 0,308205 | 511 | 5,30 điểm | 90,05 |
| 12 — sâu | 96,19 | 91,73 | 0,299211 | 496 | 4,46 điểm | 90,17 |

Train trong bảng đo lại best.pt ở eval mode, khác train accuracy online có dropout.
11 so 10: val +0,15 điểm, giảm 9 lỗi nhưng loss/gap tăng. 12 so 11: +0,25 điểm, giảm 15 lỗi.
12 so baseline: +0,40 điểm, giảm 24 lỗi nhưng val loss vẫn cao hơn. Không chỉ nhìn accuracy.
Run 12 có val accuracy tốt nhất trong MLP hiện có; không nhầm “baseline 10” với “ứng viên MLP tốt nhất”.

## 3. Những gì đã có

- Source MLP trong src/: 1_preparedata.py, 2_eda.py, 3_train.py, 4_evaluate.py, 5_report.py;
  config.py, model.py, data.py, metrics.py và logger.py là các module dùng chung.
- [Báo cáo thực nghiệm MLP](docs/06_run_report.md), [phân tích lỗi](docs/05_evaluation.md).
- [run_review.json](outputs/run_review.json) và [CSV](outputs/run_review.csv): 12 checkpoint,
  metrics, confusion, ID ảnh sai/confidence, SHA256 và các ảnh được sửa đúng/sai thêm.
- [HTML](visualization/explorer.html) đã chọn được 12 run, có so sánh 10–12 và gallery ảnh sai;
  biểu đồ còn cần cải thiện trục/mốc/khả năng đọc theo yêu cầu mới.
- [CNN notebook](cnn_basic.ipynb) **đã tồn tại**: mở rộng notebook này, không tạo bản trùng.
  Backbone: Conv(1,16) → ReLU → Pool → Conv(16,32) → ReLU → Pool;
  feature map 32×7×7, head Linear(1568,64) → ReLU → Dropout → Linear(64,10), 105.866 tham số.
- CNN outputs riêng tại outputs/cnn_runs/lan_N; không lẫn vào MLP outputs/runs/lan_N.
- CNN lần 1 mới có log tới epoch 17/50, best epoch 16 val 92,85%, loss 0,197239.
  Chưa có dòng Finished, chưa đủ cơ sở gọi là baseline CNN đã chốt hoặc kết quả cuối.
- Notebook hiện dùng trực tiếp `build_loaders`, đã bỏ wrapper worker/pinned memory và
  chuyển về cộng metrics bằng scalar. Output lưu trong notebook báo CPU, đến epoch 17/50.
- Ngày 2026-10-26: người dùng yêu cầu commit/push toàn bộ thay đổi hiện có; giữ notebook
  đúng trạng thái hiện tại, chưa chạy lại train hoặc xác nhận kết quả CNN hoàn tất.

## 4. Việc tiếp theo — thực hiện tuần tự

### A. Hoàn thiện đánh giá MLP

1. Đối chiếu nguồn số liệu rồi giải thích quá trình thử 1–9; đánh dấu các lần đổi nhiều biến.
2. So sánh riêng **10 baseline / 11 mở rộng / 12 tăng sâu**, cùng checkpoint và split.
3. Kiểm tra lỗi theo quy trình ở mục 5; hoàn thiện báo cáo trước khi chuyển sang CNN.

### B. Thực nghiệm CNN trong notebook

1. Kiểm tra notebook chạy ổn, GPU, data loader và run đang dở; không tự ghi đè hay coi là đã hoàn tất.
2. Chạy CNN cơ bản, lưu cấu hình, log mỗi epoch, history và best checkpoint theo validation.
3. Sau review, thử tuần tự: kiến trúc cơ bản → biến thể kiến trúc → optimizer/LR/scheduler/
   regularization → tinh chỉnh. Mỗi lần có mục tiêu, biến thay đổi và kết luận; không đổi tất cả cùng lúc.
4. Thống nhất số trial/epoch với người dùng trước đợt train dài; không tự chạy tìm kiếm không giới hạn.
5. Đặt tên rõ run CNN, ghi số tham số, epoch thực tế, thời gian train và seed; giữ mọi kết quả cũ.
6. Chốt cấu hình **baseline CNN** sau khi xem validation, phân biệt baseline với biến thể tốt nhất.
7. Chỉ test checkpoint đã chọn sau khi khóa cấu hình; ghi nhận test đã từng được xem trong project.

### C. So sánh MLP tốt nhất và CNN tốt nhất

Chọn ứng viên mỗi họ bằng cùng tiêu chí validation, không lấy test cao nhất để chọn.
Báo cáo train/val/test loss, accuracy, macro F1, gap, best epoch, tham số và thời gian.
Giữ split/preprocessing tương đương; nếu khác số epoch/batch/augmentation phải ghi rõ,
không quy mọi cải thiện cho riêng convolution. Test chưa đo phải để “chưa đo”.
Giải thích MLP flatten sớm, CNN học bộ lọc cục bộ dùng chung và feature maps còn hai chiều không gian;
pooling giảm độ phân giải, không giữ nguyên mọi chi tiết. Không coi hình feature map là bằng chứng đủ.

## 5. Error analysis — tiêu chí bắt buộc

- Confusion matrix số đếm và chuẩn hóa theo nhãn thật; hàng thật/cột dự đoán, đủ tên 10 lớp.
- Precision, recall, F1, support và TP/TN/FP/FN one-vs-rest từng lớp; tránh bị TN lớn che lấp lỗi.
- Phân tích Shirt ↔ T-shirt/top, Coat ↔ Pullover và các cặp nổi bật thực tế ở từng run.
- Ảnh sai phải có split, ID gốc, nhãn thật, dự đoán, confidence và tiêu chí chọn ảnh.
- Hiện cả ví dụ đúng/sai, lỗi tự tin cao, lớp yếu; không chỉ chọn ví dụ đẹp hoặc kết luận nhãn sai tùy tiện.
- So sánh **cùng ID ảnh**: MLP/CNN cùng sai, CNN sửa đúng, CNN làm sai thêm và cả hai cùng đúng.
- Kiểm tra accuracy tăng có đánh đổi recall lớp nào; phân biệt giả thuyết nguyên nhân với bằng chứng.
- Đối chiếu tổng confusion với số mẫu, tổng sai với danh sách ID, đảm bảo không trộn train/val/test.
- Hiện có 217 ảnh val sai ở cả 12 MLP; 10→11 sửa 117/sai thêm 108; 11→12 sửa 114/sai thêm 99.
  Dùng chúng làm điểm khởi đầu, rồi kiểm tra CNN có giải quyết được không; không dùng test để tune.

## 6. HTML cần hoàn thiện

- Dùng **một visualization/explorer.html**, thêm CNN và so sánh hai họ; không thêm template/viewer trùng.
- Có bộ chọn họ/run/split, bảng cấu hình, baseline/rộng/sâu, metrics và gallery lỗi liên kết confusion.
- Biểu đồ train/val loss và accuracy theo epoch có tên trục, tick số, grid, legend, tooltip nếu phù hợp;
  accuracy dùng rõ 0–1 với mốc 0,7/0,8/0,9 hoặc % với 70/80/90; epoch có mốc số cụ thể.
- Đánh dấu best epoch; phân biệt online train với eval checkpoint; trục cắt phải ghi rõ giới hạn.
- Test hiện có là điểm đo checkpoint: dùng điểm/bar so sánh, **không vẽ đường test mỗi epoch giả**.
- Nếu cần test loss theo epoch phải trao đổi lại protocol; hiện ưu tiên train/val curves + test checkpoint.
- Confusion có nhãn, thang màu và chuyển count/tỷ lệ; bố cục dễ đọc, responsive, không để đồ thị trống trục.
- Kiểm tra bằng trình duyệt/Live Server, số liệu khớp JSON; ghi rõ kiểm tra nào chưa làm.

## 7. Điều kiện bàn giao phần tiếp theo

Mỗi phần phải có artifact tái kiểm tra, báo cáo ngắn điều đã làm/kết quả/giới hạn và docs cập nhật.
Chưa có kết quả CNN hoàn tất hoặc so sánh cuối MLP–CNN; không trình bày kế hoạch như thành tích đã đạt.
Chỉ mở rộng phần tiếp theo sau review theo cách làm từng bước của người dùng.
