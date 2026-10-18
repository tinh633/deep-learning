# 4. Model và training

File: config.py, model.py, 3_train.py và logger.py trong src.
Chạy `python practice_1_fashionmnist/src/3_train.py` khi người dùng yêu cầu train.

- MODEL hiện tại: hidden1=512, hidden2=216, hidden3=128, dropout=0.2.
- FashionMLP khai báo fc1/fc2/fc3/fc4; forward tách từng thao tác, trả logits.
- CrossEntropyLoss, Adam và StepLR gọi trực tiếp, không optimizer factory.
- TRAIN chứa seed/epochs/lr/weight_decay và decay; validation sau mỗi epoch.
- logger.py tạo lan_1, lan_2, ...; cấu hình ghi một lần, metrics một dòng/epoch.
- File train chia khối bằng khoảng trắng: khởi tạo, train từng batch, validation, lưu checkpoint.
- Format log epoch nằm trong logger.log_epoch; train chỉ gọi một dòng, không tạo thêm file helper.
- metrics.py cung cấp hàm tính validation dùng chung; data.py tạo DataLoader.
- 3_train.py chạy trực tiếp, không main/if __name__; không import file này từ script khác.
- Run có train.log, metrics.json, best.pt và model.pt; không sao chép source.
- Best theo val accuracy cao nhất, hòa thì val loss thấp hơn.
- Checkpoint lưu model/data/train config, epoch và state_dict; chưa hỗ trợ resume optimizer.
- Train metrics theo epoch là online; accuracy mục tiêu cần đo lại ở eval mode.
- Run mới nhất lan_12 đã được người dùng train/test; báo cáo hiện tại theo [05_evaluation.md](05_evaluation.md).
- Cấu hình run cũ lấy từ checkpoint, kích thước lớp từ tensor; không dùng config hiện tại để mô tả lịch sử.

## Chuỗi so sánh mới do người dùng thực hiện

| Vai trò | Run | Lớp ẩn | Biến thay đổi |
|---|---|---|---|
| Baseline | lan_10 | 256 → 128 → 64 | Mốc đối chiếu |
| MLP rộng | lan_11 | 512 → 256 → 128 | Tăng width, giữ 3 lớp ẩn |
| MLP sâu — kế hoạch | lan_12 | 512 → 256 → 128 → 64 | Thêm lớp thứ tư so với bản rộng |

Giữ dropout 0.2, Adam lr=0.001, weight_decay=1e-5, 50 epoch, StepLR(15,0.3),
batch 64, seed/split 42 và cùng tiêu chí checkpoint để đối chiếu.
Đây là kế hoạch độ sâu mới; chưa sửa model hoặc chạy lan_12 trong bước ghi nhận này.
lan_11 đang có log train; không sửa code/khởi chạy thêm khi người dùng đang train.

Thứ tự tiếp theo: hoàn tất hai MLP → so validation/chọn MLP → phân tích lỗi MLP
→ thêm một CNN cơ bản như thí nghiệm bổ sung → so sánh với MLP trên cùng dữ liệu.
CNN là mở rộng đã được người dùng yêu cầu, không thay thế phần MLP của bài.
Kiến trúc CNN đề xuất để thảo luận sau: Conv(1,16) → ReLU → Pool → Conv(16,32)
→ ReLU → Pool → Flatten → Linear(32×7×7,64) → ReLU → Linear(64,10), kernel=3/padding=1.
Chưa triển khai CNN, chưa chọn cấu hình theo test; các mô hình đều chọn bằng validation trước.

Run lan_12 cũ đã bị người dùng loại khỏi thư mục runs; các số liệu lan_12 ở báo cáo cũ
không phải kết quả của bản sâu mới. Sau khi chạy lại cần xuất lại report/HTML đúng checkpoint.

## Ngân sách đã duyệt, chưa thực hiện

Tối đa 6 cấu hình, 60 epoch/run, early stopping patience 10;
top 2 cấu hình kiểm tra với 3 seed tổng cộng. Không chạy tổ hợp mọi tham số.
Kiến trúc MLP rộng/sâu và regularization thử tuần tự; không đạt aim thì tổng kết giới hạn.
Model đã được người dùng thêm lớp ẩn thứ ba; early stopping chưa triển khai.
Sau reset người dùng đã chạy lại, hiện có baseline lan_6 để đối chiếu.

## Thử regularization sau lan_6

- Giữ MLP [256,128,64], dropout=0, Adam lr=0.001, 30 epoch.
- Giữ split/seed 42, batch 64, StepLR(15,0.3).
- Chỉ thay weight_decay từ 0 thành 0.0001; hidden3=64 ghi rõ trong config để checkpoint đầy đủ.
- Chọn best theo validation accuracy, hòa thì loss; so với lan_6, không đánh giá test.
- Đây là một phép thử đơn lẻ; không kết luận cải thiện cho đến khi có số đo.

### Kết quả lan_7 — đã chạy đủ 30 epoch

Đo lại best.pt của từng run trên train và validation ở eval mode:

| Run | Weight decay | Best epoch | Train acc | Val acc | Val loss |
|---|---:|---:|---:|---:|---:|
| lan_6 | 0 | 22 | 95,76% | 91,03% | 0,312959 |
| lan_7 | 0,0001 | 26 | 95,51% | 90,85% | 0,300252 |

Val accuracy giảm 0,18 điểm phần trăm (11 ảnh/6.000); val loss thấp hơn khoảng 0,0127.
Gap train-val chỉ giảm từ 4,73 xuống 4,66 điểm phần trăm: chưa giải quyết đáng kể overfitting.
Một seed không đủ kết luận weight decay luôn tốt/xấu; lần thử này không cải thiện accuracy.
Giữ lan_6 làm mốc accuracy tốt hơn trong cặp so sánh; chưa chốt mô hình cuối, chưa chạy test.
EVALUATE.run vẫn do người dùng chọn; config được cập nhật cho từng thử nghiệm mới.

Bằng chứng: [so sánh](../outputs/runs/lan_7/regularization_comparison.json),
[train log](../outputs/runs/lan_7/train.log), [eval log](../outputs/runs/lan_7/evaluate.log).

### Thử tiếp dropout 0.1

Giữ baseline lan_6: MLP [256,128,64], Adam lr=0.001, weight_decay=0,
seed 42, batch 64, StepLR(15,0.3), 30 epoch. Chỉ bật dropout=0.1 ở ba lớp ẩn.
Không kết hợp với weight decay để đối chiếu riêng từng thay đổi; chưa đánh giá test.
Train accuracy online có dropout không so trực tiếp với baseline không dropout;
cần đo lại train/val của cùng best checkpoint ở eval mode sau khi hoàn tất.

### Kết quả dropout — lan_8

Đã chạy đủ 30 epoch; best theo val accuracy tại epoch 22.
Bảng dưới đo cùng checkpoint ở eval mode, dropout tắt; số baseline lấy từ lần đánh giá đã xác minh trước đó.

| Run | Regularization | Train acc | Val acc | Val loss | Gap train-val |
|---|---|---:|---:|---:|---:|
| lan_6 | Không | 95,76% | 91,03% | 0,312959 | 4,73 điểm % |
| lan_7 | Weight decay 0,0001 | 95,51% | 90,85% | 0,300252 | 4,66 điểm % |
| lan_8 | Dropout 0,1 | 94,46% | 90,97% | 0,278345 | 3,49 điểm % |

Dropout giảm gap khoảng 1,23 điểm phần trăm và giảm val loss khoảng 0,0346 so với baseline.
Val accuracy thấp hơn 0,067 điểm phần trăm (4 ảnh/6.000), chưa có bằng chứng cải thiện accuracy.
Gap nhỏ hơn phù hợp với regularization nhưng không chứng minh hết overfitting;
train accuracy cũng giảm và còn thấp hơn mục tiêu >96%.
Một seed không đủ kết luận cấu hình nào vượt trội; chưa tăng rộng/sâu hoặc thử seed khác ở bước này.

Config giữ dropout=0.1, weight_decay=0 để người dùng xem; chưa chốt model, chưa chạy test.
Bằng chứng: [train](../outputs/runs/lan_8/train.log), [evaluate](../outputs/runs/lan_8/evaluate.log),
[so sánh](../outputs/runs/lan_8/regularization_comparison.json).

Tiếp theo: [evaluation](05_evaluation.md).

## Phong cách tham khảo đã đọc

Tài liệu người dùng cung cấp: **Chapter 3 - TRAINING DEEP NEURAL NETWORKS (1).pdf**, 25 trang.

- Trang 5: dùng datasets/transforms/DataLoader trực tiếp; không tự viết lại thư viện.
- Trang 6: model class khai báo từng lớp trong __init__, forward từng thao tác riêng.
- Trang 7–9: loss/optimizer rõ ràng; vòng epoch/batch hiện đủ zero_grad, forward, loss, backward, step.
- Trang 10–12: validation dùng eval/no_grad, trả lại train mode; lưu và nạp state_dict.
- Giữ log một lần mỗi epoch theo yêu cầu người dùng, dù ví dụ slide in từng batch.
- Không dùng test làm validation như ví dụ giản lược trang 11.
- Không sao chép mean/std của MNIST sang FashionMNIST ở trang 5.
- TensorBoard, transfer learning và CNN trong phần sau không thuộc phạm vi hiện tại.

Bước đọc tài liệu chỉ cập nhật hướng dẫn; chưa thay code, thêm dependency hoặc train.
