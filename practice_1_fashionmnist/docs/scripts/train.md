# train.py

- Chạy: `python practice_1_fashionmnist/src/train.py`.
- Cấu hình hiện tại: MLP 784→256→128→10, 20 epoch, Adam lr=0.001, weight_decay=0.0001, StepLR(10,0.3), seed 42, batch 64.
- Train: forward → CrossEntropyLoss → backward → optimizer.step. Validation: eval/no_grad sau mỗi epoch; không dùng test trong train.
- Một dòng mỗi epoch: train_loss, val_loss, val_accuracy, best_epoch và lr đã dùng. Scheduler cập nhật sau epoch.
- best.pt cập nhật khi validation loss giảm; model.pt lưu epoch cuối. Cả hai chứa state_dict, không có optimizer state/resume.
- Mỗi lần chạy tạo outputs/runs/lan_N/ chưa tồn tại, giữ nguyên run cũ. Không sao chép code, không tạo CSV/config riêng.
- train.log ghi một lần cấu hình thực tế, loss/optimizer/options, model, seed, dữ liệu và device, rồi kết quả từng epoch.
- evaluate.py mặc định dùng best.pt; test.py dùng run/checkpoint được chọn thủ công. best.pt chỉ tốt nhất theo validation loss trong run đó, không đồng nghĩa đã chốt mô hình cuối.
- Giữ kiến trúc và data split/transform tương ứng khi tải lại trọng số.
