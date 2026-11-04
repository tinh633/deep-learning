# 3_evaluate.py — Đánh giá checkpoint và phân tích lỗi

Source: [3_evaluate.py](../../src/3_evaluate.py).
Từ gốc: `python practice_2_pretrained_model/src/3_evaluate.py`.
Có main guard cho Windows workers; không train hoặc cập nhật weights.

## Chọn run và khôi phục

EVAL_RUN trong config chọn outputs/runs/lan_N/best.pt, không dùng last.pt.
RUN_TEST=False chỉ đo train/validation; True thêm test.
MODEL_NAME/EXPERIMENT hiện tại không thay cấu trúc checkpoint được đánh giá.

Đọc saved.settings, classes, epoch và weights; dựng PretrainedCNN(pretrained=False),
nạp model_state strict=True rồi chuyển lên CUDA/CPU và model.eval().
Checkpoint chứa NumPy RNG nên torch.load dùng weights_only=False;
chỉ nạp checkpoint đã xác minh nguồn gốc.

So SHA256 của split local với saved.settings.split_sha256; khác thì dừng.
SHA nhận diện bytes chính xác, không phải chỉ kiểm tra nội dung JSON tương đương.
build_loaders dùng settings.data đã lưu, training=False;
train/val/test đều không augmentation/shuffle. Thứ tự lớp phải khớp checkpoint.
IMAGENET_MEAN/STD vẫn lấy hằng hiện tại của data/config, không lưu riêng trong settings.data.

## Suy luận và chỉ số

no_grad bao vòng batch; forward → CrossEntropyLoss → softmax.
Thu targets, predicted, confidence và trung bình loss có trọng số số mẫu.
Confusion đủ10 lớp, hàng thật/cột dự đoán; chuẩn hóa theo support của hàng.

- TP là đường chéo; FP=tổng cột−TP; FN=tổng hàng−TP; TN=N−TP−FP−FN.
- Precision/recall/F1/support tính bằng sklearn, zero_division=0.
- Macro F1 là trung bình F1 của10 lớp.
- TPR=TP/(TP+FN), FPR=FP/(FP+TN), specificity=TN/(FP+TN).
- Với tỷ lệ mẫu số0, các trường tpr/fpr/specificity trả None.

Mỗi prediction có id/true/predicted/confidence.
Train/val ID tham chiếu train gốc50.000; test ID0–9.999.
wrong_ids lưu toàn bộ lỗi; confidence max softmax chưa phải độ tin cậy đã hiệu chuẩn.

## Dữ liệu ROC/PR

Chỉ test lưu probabilities Nx10 cùng probability_class_order;
thứ tự hàng trùng predictions. Kiểm tra shape, hữu hạn và tổng xác suất≈1.
Script chưa vẽ ROC/AUC/PR; notebook đọc xác suất này để tính và trình bày.
Không dùng argmax/confidence duy nhất để thay xác suất từng lớp khi tính ROC.

## Artifact và log

| Đường dẫn | Nội dung |
|---|---|
| outputs/runs/lan_N/evaluate.log | Config/môi trường/kiến trúc, tổng điểm và chỉ số từng lớp |
| train_metrics.json | Train đo lại bằng best checkpoint |
| validation_metrics.json | Validation cùng checkpoint |
| test_metrics.json | Chỉ được cập nhật khi RUN_TEST=True |
| visualization/runs/lan_N/*_errors.png | Tối đa24 lỗi tự tin cao, hòa theo ID |

Các JSON metrics ở trong thư mục run; có checkpoint SHA, settings, epoch,
environment, thời điểm UTC, thời gian, confusion, per_class và predictions.
PNG lấy ảnh RGB gốc32×32 phóng96×96 nearest, không lấy tensor đã normalize.
Không có lỗi thì không tạo PNG mới; file cũ có thể vẫn tồn tại từ lần trước.
RUN_TEST=False cũng không xóa test_metrics.json cũ.

Kiểm tra số prediction/confusion khớp dataset và loss/confidence hữu hạn.
JSON dùng file tạm rồi replace; log ghi nối.
Ngắt/lỗi giữ báo cáo split đã ghi; các split có thể thuộc các lần đánh giá khác nhau,
vì vậy đọc checkpoint_sha256/evaluated_at để đối chiếu trước khi ghép báo cáo.

Liên quan: [data](data.md), [model](model.md), [logger](logger.md).
