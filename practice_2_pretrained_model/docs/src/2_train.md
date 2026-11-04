# 2_train.py — Huấn luyện một cấu hình

Source: [2_train.py](../../src/2_train.py).
Chạy từ gốc: `python practice_2_pretrained_model/src/2_train.py`.
Có main guard để Windows workers không thực thi lại vòng train khi import.

## Đầu vào

config chọn MODEL_NAME, EXPERIMENT, PRETRAINED, EPOCHS và RESUME_RUN.
Dataset phải có sẵn, download=False. Không tự chạy cả sáu tổ hợp.
Seed random/NumPy/PyTorch, tắt cuDNN benchmark và bật cuDNN deterministic;
không gọi torch.use_deterministic_algorithms(True).
CUDA được chọn nếu có, nếu không dùng CPU; `.to(device)` áp dụng cho model và batch.

## Khởi tạo

RESUME_RUN=None tạo lan_N mới; có tên run thì dùng thư mục đó.
Mở train.log và SummaryWriter(tensorboard/, flush_secs=2).
Tạo loaders → PretrainedCNN → CrossEntropyLoss → Adam.
Adam chỉ nhận tham số requires_grad=True; không có scheduler, AMP hay early stopping.
Lượt mới dùng weights pretrained nếu PRETRAINED=True;
resume khởi tạo không tải pretrained rồi nạp last.pt.

## Vòng lặp hiển thị trực tiếp

1. model.train(): bật chế độ học theo chính sách freeze/BatchNorm của model.py.
2. Từng batch: lên device, zero_grad, forward, tính loss, kiểm tra hữu hạn,
   backward và optimizer.step.
3. Cộng loss×batch_size và số đúng để tính trung bình toàn tập train.
4. model.eval()/no_grad để đo validation, không augmentation và không update weights.
5. Tạo row gồm epoch, train/val loss/accuracy, LR, seconds.
6. save_epoch lưu best/last/history; log_epoch ghi một dòng và TensorBoard scalar.

Không dùng test trong vòng train dù build_loaders có trả test_loader.
Train online có dropout/augmentation và weights thay đổi theo batch;
khác train_metrics.json đo lại cùng best checkpoint ở eval mode.
seconds của row đo phần train/validation trước thao tác lưu checkpoint/log.
Summary thời gian gồm cả setup của phiên; GPU_peak_bytes là peak allocated PyTorch.

## Resume và ngắt

resume_training kiểm tra settings và khôi phục weights/Adam/RNG/generator/history.
Epoch bắt đầu len(history)+1, dừng ở config.EPOCHS.
Không tự mở lớp theo epoch: B/C fine-tune từ đầu.

KeyboardInterrupt ghi Interrupted và ném lại; lỗi khác ghi traceback Failed.
finally đóng SummaryWriter/logger và ghi Summary.
Không tạo checkpoint ngay trong nhánh ngắt: chỉ epoch đã save xong được bảo toàn.
Giữa epoch thì resume chạy lại phần epoch chưa hoàn tất từ last.pt trước đó.
Đổi cấu hình khi resume có thể bị từ chối; xem [logger](logger.md).

## Đầu ra

outputs/runs/lan_N: config.json, train.log, history.json,
best.pt, last.pt, tensorboard/events.out.tfevents.*.
Best theo val accuracy, hòa val loss rồi epoch sớm nhất.
Xem bài không cần chạy train lại; [evaluate](3_evaluate.md) đọc best.pt riêng.
