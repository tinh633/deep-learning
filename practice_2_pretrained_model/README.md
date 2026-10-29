# Practice 2 — Pretrained CNN trên CIFAR-10

**Kế hoạch mới: VGG16 + DenseNet121, mỗi model 3 cấu hình A/B/C, tổng 6 lượt.**
Python thực hiện train/evaluate; notebook tổng hợp phương pháp, kết quả và nhận xét.

## Bắt đầu từ đâu?

1. Đọc [yêu cầu lab](Practice_2_Pretrained_CNN_Requirements.md) và [kế hoạch](Experiment_Plan.md).
2. Mở [Practice_2_Report.ipynb](Practice_2_Report.ipynb) trong VS Code/Jupyter, chọn kernel .venv.
3. Run All: đọc EDA/artifact và biểu đồ hiện có; không tự train, tải weights hoặc chạy test.
4. Xem [progress.md](progress.md) trước khi tiếp tục; train đang tạm dừng.

## Mục tiêu

So sánh transfer learning/fine-tuning của VGG16 và DenseNet121.

Đáp ứng checklist và phân tích có kiểm soát với thời gian vừa phải.

Cùng split/seed/batch/epoch, so A/B về freeze và B/C về LR.

## Cấu trúc

```text
practice_2_pretrained_model/
├── Practice_2_Report.ipynb    # Bài trình bày, đọc kết quả thật
├── src/                      # 7 file Python, một pipeline train/evaluate
├── data/                     # Dataset/RGB viewer, giữ local
├── outputs/                  # Split, EDA, logs/history, best/last, TensorBoard
├── visualization/eda.html     # Duyệt dữ liệu đầy đủ, nhúng TensorBoard nếu server chạy
├── Experiment_Plan.md         # Thiết kế 6 lượt và checklist
├── Training.md                # Lệnh chạy, artifact và resume
├── Model_QA.md                # Giải thích câu hỏi trong quá trình làm
├── progress.md                # Trạng thái và việc tiếp theo
├── AI_Usage_Step.md           # Nhật ký theo giai đoạn
└── docs/AI_Usage_Full.md      # Nhật ký kỹ thuật có hỗ trợ AI
```

| File | Vai trò |
|---|---|
| config.py | Model, cấu hình A/B/C, hằng số dùng chung, run train/evaluate |
| data.py | CIFAR-10, split phân tầng và DataLoader; evaluation không augmentation/shuffle |
| model.py | VGG16/DenseNet121, head Linear 10 lớp và freeze/unfreeze |
| logger.py | Môi trường/log/TensorBoard, lưu và khôi phục trạng thái epoch |
| 1_eda.py | Thống kê train, phân bố lớp, duplicates, preprocessing, dữ liệu HTML |
| 2_train.py | Vòng lặp train/validation trực tiếp, ghi kết quả cuối epoch |
| 3_evaluate.py | Đọc checkpoint thực, metrics/confusion/per-class/ảnh sai, test có khóa |

## Dữ liệu và EDA

- CIFAR-10: 45.000 train / 5.000 validation / 10.000 test, phân tầng seed 42.
- EDA đã tạo outputs/eda.json, RGB train/histogram, kiểm tra trùng và preprocessing.
- Chuẩn hóa ImageNet; train có random flip, evaluation không augmentation.
- [Nguồn CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html).
- URL archive: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
- MD5 archive: c58f30108f718f92721af3b95e74349a; torchvision kiểm tra khi tải.
- Dataset, weights, event logs và pixel ảnh phát sinh giữ local, không commit lên Git.

## Sử dụng

Chạy từ gốc workspace bằng .venv; xem lệnh cụ thể và lưu ý GPU trong [Training.md](Training.md).

- EDA: src/1_eda.py, sau đó xem notebook hoặc mở visualization/eda.html bằng Live Server.
- Train: chọn MODEL_NAME/EXPERIMENT/RESUME_RUN trong config rồi chạy src/2_train.py.
- A: freeze H1 LR .001; B: fine-tune H1 LR .001; C: fine-tune H1 LR .0001; batch 64, tối đa 20 epoch.
- Resume chỉ cùng run/cấu hình; best dùng validation, last dùng tiếp tục epoch.
- Evaluate: chọn EVAL_RUN; giữ RUN_TEST=False đến khi khóa checkpoint bằng validation.
- TensorBoard: mở server theo lệnh trong HTML hoặc Training; notebook không tự bật server.

## Nội dung notebook

Notebook đi theo: mục tiêu → dữ liệu/EDA → preprocessing → kiến trúc/freeze → ma trận → cách train → kết quả → đánh giá → kết luận/checklist.
Mỗi số đo kèm model, cấu hình, split, epoch và artifact; dữ liệu thiếu được ghi rõ chưa đo.
Phần sau test gồm TP/TN/FP/FN, TPR/FPR/specificity, ROC OvR và macro/micro AUC, đối chiếu theo lớp và cùng tập test.
PR curve/AP bổ sung phân tích positive; bảng và ảnh lỗi ghép cùng ID cho biết model nào sửa được lỗi của model còn lại.
Kết quả run dở dang được phân biệt với lượt hoàn tất; validation và test được báo cáo riêng.
