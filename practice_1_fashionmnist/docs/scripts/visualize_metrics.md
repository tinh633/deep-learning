# visualize_metrics.py

- Chạy: `python practice_1_fashionmnist/src/visualize_metrics.py [lan_N]` (mặc định lan_10).
- Input: model_report.json của run và eda.json; chạy model_report.py nếu thiếu báo cáo.
- Phụ thuộc: NumPy, matplotlib. Không train/evaluate lại model.
- Output trong outputs/runs/lan_N/plots/: confusion_matrices.png, tp_tn_fp_fn.png, split_loss_accuracy.png, epoch_curves.png và tp_tn_fp_fn.json.
- Confusion matrix: hàng nhãn thật, cột dự đoán; mỗi tập dùng số lượng ảnh riêng.
- Với lớp i: TP=M[i,i], FN=tổng hàng−TP, FP=tổng cột−TP, TN=tổng mẫu−TP−FN−FP. Đây là one-vs-rest, không phải một ma trận nhị phân chung cho bài 10 lớp.
- Biểu đồ loss/accuracy ba tập dùng cùng checkpoint best.pt trong eval mode.
- Đường epoch: train loss online, validation loss/accuracy. Log không có train accuracy hoặc test từng epoch; không suy diễn các đường này từ một checkpoint.
- Chạy lại cập nhật biểu đồ của cùng báo cáo; các log train/evaluate/test và checkpoint không bị sửa.
