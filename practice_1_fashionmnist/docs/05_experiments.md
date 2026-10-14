# 05 — So sánh các lần train và đối chiếu đề bài

Số liệu lấy từ `train.log`, `evaluate.log`, `test.log` trong `outputs/runs/`. Mọi run dùng cùng split
(54.000 train / 6.000 validation, seed 42), `ToTensor()`, CrossEntropyLoss, batch 64 và seed 42.
Mỗi cấu hình chỉ chạy một seed.

## 1. Các lần train

| Run | Kiến trúc (số tham số) | Optimizer, lr | Weight decay | LR decay | Epoch | Train loss cuối | Validation |
|---|---|---|---|---|---|---|---|
| lan_1 | 784-128-10 (101.770) | SGD momentum 0,9; 0,01 | 0 | không | 20 | 0,2381 | không có trọng số |
| lan_2 | 784-128-10 (101.770) | Adam; 0,01 | 0 | không | 20 | 0,3030 | không có trọng số |
| lan_3 | 784-128-64-10 (109.386) | Adam; 0,01 | 0 | không | 1, chỉ 128 ảnh | 2,2508 | 2,2595 / 19,93% (kiểm tra pipeline) |
| lan_4 | 784-128-64-10 (109.386) | Adam; 0,01 | 0 | không | 20 | 0,3044 | 0,4035 / 87,47% |
| lan_5 | 784-512-256-128-10, Dropout 0,2 (567.434) | AdamW; 0,001 | 1e-4 | không | 50 | 0,1535 | 0,3156 / 90,82% |
| lan_6 | 784-256-128-10 (235.146) | Adam; 0,001 | 0 | không | 20 | 0,1588 | 0,3187 / 90,05% |
| lan_7 | như lan_6 | Adam; 0,001 | 0 | StepLR(10; 0,3) | 20 | 0,1380 | 0,2938 / 90,75% |
| lan_8 | như lan_6 | Adam; 0,001 | 1e-4 | StepLR(10; 0,3) | 20 | 0,1643 | 0,2729 / 90,67% |
| lan_9 | như lan_8 | Adam; 0,001 | 1e-4 | StepLR(10; 0,3) | 20 | 0,1643 | best.pt epoch 15: 0,2646 / 90,67% |
| lan_10 | như lan_6 | Adam; 0,001 | 0 | StepLR(10; 0,3) | 23/30, dừng sớm | 0,1148 (epoch 23) | best.pt epoch 12: 0,2751 / 90,57% |

Validation của lan_3 đến lan_8 là trọng số cuối epoch (`model.pt`); lan_9 và lan_10 là `best.pt`.
lan_1 và lan_2 chỉ có `train.log`, không lưu trọng số nên không đánh giá được validation.
Test chỉ được đánh giá cho lan_9 (0,3099 / 89,21%) rồi lan_10 (0,3201 / 89,41%).

## 2. Quan sát (một seed, không phải kết luận thống kê)

- lan_8 và lan_9 có cùng cấu hình và cho cùng kết quả ở epoch 20 (train loss 0,1643; validation 90,67%), cho thấy lần chạy tái lập được trên cùng máy.
- lan_4 (lr 0,01) đạt 87,47%, thấp hơn các run lr 0,001 (90,05–90,82%); kiến trúc cũng khác nên chưa tách riêng ảnh hưởng của learning rate.
- Từ lan_6 → lan_7 → lan_8, thêm StepLR rồi weight decay đi cùng validation loss giảm 0,3187 → 0,2938 → 0,2729.
- lan_5 có validation accuracy cao nhất (90,82%) nhưng validation loss (0,3156) không thấp nhất.
- Các run lan_5 đến lan_10 chênh nhau dưới 1 điểm phần trăm validation accuracy; chưa đo biến thiên giữa các seed nên không xếp hạng chắc chắn.
- lan_10 được cấu hình sau khi đã xem kết quả test của lan_9, nên lựa chọn này có thể chịu ảnh hưởng của test.

## 3. So sánh lan_9 và lan_10 (cùng quy trình đánh giá)

| | lan_9 `best.pt` (epoch 15) | lan_10 `best.pt` (epoch 12) |
|---|---|---|
| Validation loss / accuracy | 0,2646 / 90,67% | 0,2751 / 90,57% |
| Test loss / accuracy | 0,3099 / 89,21% | 0,3201 / 89,41% |
| Train accuracy (eval mode) | 93,59% | 93,50% |
| Số ảnh sai train / validation / test | 3.459 / 560 / 1.079 | 3.509 / 566 / 1.059 |

lan_9 tốt hơn trên validation; lan_10 tốt hơn trên test 20 ảnh (0,2 điểm phần trăm). Chênh lệch nhỏ và chỉ có một seed
mỗi run. Báo cáo, biểu đồ của cả hai nằm trong `outputs/runs/lan_9/` và `lan_10/`; trang xem model: `explorer.html?run=lan_9`.

## 4. Đối chiếu đề bài

| Yêu cầu | Sản phẩm | Ghi chú |
|---|---|---|
| Learn | — | Tự học theo tài liệu PyTorch; repo không có sản phẩm kiểm chứng |
| Nạp FashionMNIST | `prepare_data.py`, `data_pipeline.py` | MD5 khớp; 54.000 / 6.000 / 10.000 |
| Transforms | `ToTensor()` chia 255 | Không Normalize, không augmentation |
| Xây mạng | `model.py` | MLP 784→256→128→10, 235.146 tham số |
| Vòng lặp train | `train.py` | forward → loss → backward → optimizer.step |
| Đánh giá accuracy | `evaluate.py`, `test.py` | Validation 90,57%; test 89,41% |
| Lưu và nạp mô hình | `best.pt` (state_dict) | Nạp lại ở evaluate, test và model_report |
| Thử nghiệm mạng/hyperparameters | Bảng mục 1 | 10 lần train, một seed |
| Biểu đồ loss | `plots/epoch_curves.png`, `explorer.html` | |
| Ảnh dự đoán và nhãn thật | `plots/predictions.png`, `plots/confident_mistakes.png`, `explorer.html` | |
| Báo cáo ngắn | [04_result.md](04_result.md), tài liệu này | |
| Tài liệu PyTorch | — | Do người thực hiện tham khảo |
