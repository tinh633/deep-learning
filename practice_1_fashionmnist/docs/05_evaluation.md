# 5. Evaluation và report

Script: src/4_evaluate.py và src/5_report.py.

Hai script chạy trực tiếp, không main/if __name__. Hàm dùng chung evaluate/load_checkpoint
nằm trong metrics.py; DataLoader trong data.py. Không import script 4/5 vì sẽ chạy tác vụ ngay.

- EVALUATE hiện chọn lan_12/best.pt, include_test=True theo cấu hình người dùng.
- Chạy `python practice_1_fashionmnist/src/4_evaluate.py` sau khi có checkpoint.
- Đánh giá train/validation cùng trọng số bằng eval/no_grad; loss trung bình theo ảnh.
- Test chỉ bật include_test=True sau khi khóa cấu hình bằng validation.
- Loader chỉ đọc checkpoint mới có config; không giữ nhánh tương thích run cũ đã xóa.
- Chạy `python practice_1_fashionmnist/src/5_report.py` để xuất JSON và biểu đồ.
- Report có kiến trúc, cấu hình/log, epoch, confusion matrix và ảnh sai kèm ID gốc/nhãn/confidence.
- TP/TN/FP/FN theo từng nhãn one-vs-rest; hàng matrix là thật, cột là dự đoán.
- Vẽ confusion, TP/TN/FP/FN, loss/accuracy theo epoch và scores cùng checkpoint.
- JSON nằm trong run và outputs/model_report.json cho HTML; chỉ có các tập đã đánh giá.
- Xuất classification.csv theo lớp, macro/weighted F1 và ví dụ đúng/sai trong plots/predictions.png.
- outputs/experiments.json tổng hợp cấu hình checkpoint, trạng thái hoàn tất, val accuracy/loss và minimum val loss các run.
- Khi chưa có model/report, HTML hiện thông báo thay vì kết quả lịch sử.

## Kết luận và giới hạn

So sánh train/val/test cùng checkpoint, kiểm tra các lớp dễ nhầm và khoảng cách tổng quát hóa.
Gap train-val 3 điểm phần trăm là ngưỡng cảnh báo đề xuất, không chứng minh hết overfitting.
Kiểm tra nhiều seed; không coi train accuracy cao là bằng chứng đủ về chất lượng model.
Test đã được xem trước reset; xóa file không khiến test trở lại hoàn toàn độc lập.
Chỉ báo số liệu thực chạy; không cam kết MLP sẽ đạt >95% val/test.

Commit và push chỉ khi người dùng cho phép riêng.

## Đối chiếu repo cùng đề tài và phần củng cố lab

Nguồn tham khảo: [zombieTDV / LAB1](https://github.com/zombieTDV/Deeplearning_Course_UTH/tree/LAB1-FashionMNIST-Classification).
Đối chiếu với đề Practice 1 đã cung cấp, không coi cấu trúc repo tham khảo là rubric chấm điểm.

| Nội dung đề | Project hiện tại | Phần củng cố |
|---|---|---|
| Dataset, transforms, DataLoader | Có, split stratified và nguồn/checksum | Giữ split chính thức test 10.000 |
| MLP, autograd, loss, optimizer | Có, code tường minh | Không cần chuyển CNN/notebook |
| Train và đánh giá accuracy | Có log, validation mỗi epoch | Tổng hợp các run để giải thích thử nghiệm |
| Lưu/nạp model | Có best.pt/model.pt và config | Báo cáo đúng checkpoint đã nạp |
| Loss graphs | Có script/HTML | Xuất lại cho run đang được chọn |
| Predicted vs actual | Có ảnh sai ở HTML | Bổ sung ví dụ đúng và sai vào PNG nộp bài |
| Brief report | Docs chủ yếu hướng dẫn, còn lệch run mới | Ghi kết quả, giới hạn và lỗi theo lớp |

Repo tham khảo có [metrics theo lớp](https://github.com/zombieTDV/Deeplearning_Course_UTH/blob/LAB1-FashionMNIST-Classification/src/eval_utils.py)
và [nhật ký thí nghiệm](https://github.com/zombieTDV/Deeplearning_Course_UTH/blob/LAB1-FashionMNIST-Classification/MLP_Experiment_Changelog.md).
Áp dụng ý tưởng trình bày: precision/recall/F1, các cặp nhầm và bảng tham số/kết quả;
không sao chép code, trọng số hoặc số đo của họ vào bài này.

Không áp dụng nguyên trạng [cách chia dữ liệu](https://github.com/zombieTDV/Deeplearning_Course_UTH/blob/LAB1-FashionMNIST-Classification/src/data_utils.py):
repo có hàm gộp train/test gốc rồi chia 50k/6k/14k; project này giữ 54k/6k và test chính thức 10k.
Vì split khác, không so accuracy ngang hàng hoặc kết luận project nào tốt hơn chỉ từ con số.
ROC/PR curves, CNN, Optuna trong repo là phần mở rộng; chưa thấy chúng là yêu cầu bắt buộc trong đề đã cung cấp.
Mốc >95% là mục tiêu người dùng, không phải ngưỡng pass được xác nhận từ đề.

Phạm vi bổ sung lần này: củng cố báo cáo/metrics/so sánh run từ checkpoint đã có,
không đổi kiến trúc, tham số hoặc train mới; không tự chốt model cuối.

## Báo cáo lịch sử — lan_12 cũ

Người dùng đang làm lại chuỗi rộng/sâu. Thư mục lan_12 cũ không còn;
số liệu dưới đây là kết quả lần trước, không mô tả bản sâu lan_12 mới sắp chạy.

Checkpoint: best.pt, epoch 42, tiêu chí val accuracy cao nhất; không phải minimum val loss.
MLP 784 → 512 → 216 → 128 → 10, dropout 0.2; ghi đúng 216 theo checkpoint, không đổi thành 256.
Adam lr=0.001, weight_decay=1e-5; StepLR(15,0.3), 50 epoch, batch 64, seed 42.
ToTensor, split 54.000/6.000/10.000; nạp lại checkpoint và đánh giá eval/no_grad.

| Tập | Accuracy | Loss | Macro F1 | Số lỗi |
|---|---:|---:|---:|---:|
| Train | 96,13% | 0,101472 | 0,9612 | 2.090 |
| Validation | 91,40% | 0,295117 | 0,9138 | 516 |
| Test | 90,05% | 0,350778 | 0,9003 | 995 |

Train-val cách 4,73 điểm phần trăm, val-test cách 1,35; chưa thể kết luận hết overfitting.
Shirt có test recall 71,5%, Coat 82,6%, Pullover 83,6%; lỗi tập trung ở nhóm áo.
Các cặp nổi bật: Shirt → T-shirt/top 120 ảnh, chiều ngược lại 99; Coat → Pullover 83,
Pullover → Coat 72. Đây là kết quả của project này, không lấy từ repo tham khảo.
Val loss thấp nhất của run tại epoch 22 là 0,263777, khác checkpoint epoch 42 đang báo cáo.
Không khẳng định có thể khôi phục trọng số epoch 22 từ log; chỉ best.pt/model.pt đã lưu mới sử dụng được.

### Artifacts cần xuất lại sau chuỗi mới

- Báo cáo JSON và classification.csv theo lớp của checkpoint mới.
- Loss/accuracy, confusion matrix, ảnh đúng/sai, scores và TP/TN/FP/FN.
- [Lịch sử thí nghiệm](../outputs/experiments.json), bảng tương tác tại visualization/explorer.html.

Chưa có nhiều seed để chứng minh chênh lệch nhỏ giữa cấu hình có ý nghĩa.
Một số run đổi nhiều tham số cùng lúc; bảng tổng hợp không biến chúng thành thí nghiệm một biến.
Không có rubric chấm điểm chi tiết ngoài đề Practice 1 đã cung cấp; đây là kiểm tra độ đầy đủ,
không phải cam kết được điểm/pass. Người dùng vẫn cần kiểm tra yêu cầu riêng của giảng viên.
