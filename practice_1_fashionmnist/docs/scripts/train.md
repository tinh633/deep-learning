# train.py

- Chạy: `python practice_1_fashionmnist/src/train.py`.
- Cấu hình hiện tại trong train.py: MLP 784→256→128→10, 30 epoch, Adam lr=0.001, không weight decay, StepLR(10,0.3), seed 42, batch 64. Cấu hình các run cũ nằm trong train.log của từng run và bảng ở 05_experiments.md.
- Train: forward → CrossEntropyLoss → backward → optimizer.step. Validation: eval/no_grad sau mỗi epoch; không dùng test trong train.
- Một dòng mỗi epoch: train_loss, val_loss, val_accuracy, best_epoch và lr đã dùng. Scheduler cập nhật sau epoch.
- best.pt cập nhật khi validation loss giảm; model.pt lưu epoch cuối. Cả hai chứa state_dict, không có optimizer state/resume.
- Mỗi lần chạy tạo outputs/runs/lan_N/ chưa tồn tại, giữ nguyên run cũ. Không sao chép code, không tạo CSV/config riêng.
- train.log ghi một lần cấu hình thực tế, loss/optimizer/options, model, seed, dữ liệu và device, rồi kết quả từng epoch.
- evaluate.py mặc định dùng best.pt; test.py dùng run/checkpoint được chọn thủ công. best.pt chỉ tốt nhất theo validation loss trong run đó, không đồng nghĩa là mô hình bàn giao; mô hình bàn giao là lan_10/best.pt (xem 04_result.md).
- Giữ kiến trúc và data split/transform tương ứng khi tải lại trọng số.
