# AI Usage Step — Ghi chú giai đoạn hiện tại

## 2026-10-29 — Phân tích chỉ số sau test

- Bổ sung PR curve/AP theo lớp, macro/micro AP và đối chiếu bốn nhóm đúng/sai cùng ID, thống kê theo lớp và ảnh test local.
- Cặp checkpoint xác định rõ, không tự chọn bằng điểm test; AP khác trapezoidal PR-AUC. Kiểm tra giả perfect/chance AP, ID alignment, nhãn sai/ID trùng, gallery và missing-report qua.
- Không chạy test thật, không thay source train/evaluate, không lưu pixel outputs. Nội dung đã tổng hợp Full.

- Tham khảo module metrics LAB2 và tài liệu scikit-learn; thêm TPR/FPR/specificity, ROC OvR per-class/macro/micro và bảng so sánh vào notebook.
- Evaluate lưu probabilities Nx10 và thứ tự lớp chỉ ở test report; giữ test tắt mặc định, không thêm lượt train.
- Kiểm tra giả qua: AUC perfect=1/chance=.5, rates, lớp vắng, thiếu/sai score, plots, khác split bị chặn; smoke evaluate score đúng thứ tự và rates đúng.
- Không chạy test CIFAR thật; số liệu giả chỉ trong bộ nhớ/thư mục tạm, notebook không lưu outputs. Đã tổng hợp Full.

## 2026-10-29 — Source và cấu hình

- Dọn source theo ma trận sáu lượt: bỏ ResNet18/H2, head selector và HEAD/TRAIN không còn dùng; giữ Linear 10 lớp và tên state_dict.
- Resume chỉ đọc last.pt full-state, không fallback best/reset Adam; train/evaluate/notebook dùng API hai backbone.
- CPU kiểm tra shape/gradient/BatchNorm hai model ở hai chế độ, logits khớp H1 trước dọn; bản sao checkpoint thật resume/save qua, không sửa artifact gốc.
- Smoke tạm train/save/resume/evaluate qua API mới, validation ID/confusion đúng và test tắt; không train chính thức.

- Hỗ trợ AI tập trung vào source, cấu hình, notebook, tài liệu và kiểm tra; số liệu lấy từ artifact thực.
- config.py dùng hằng số dễ đọc; model nhận keyword; get_settings() lưu metadata tương thích checkpoint.
- Train/validation giữ vòng lặp trực tiếp; logger hỗ trợ cấu hình máy, epoch metrics, resume và best/last.
- Evaluate đọc full-state checkpoint có NumPy RNG, dùng preprocessing không augmentation/shuffle.
- Kiểm tra CPU: continuous/ngắt-resume khớp weights, best/last/history qua; đổi cấu hình/split bị từ chối.
- Metrics khớp sklearn/ID/PNG; test tắt; DataLoader CIFAR thật giữ split 45k/5k/10k.

## 2026-10-29 — Git và lượt VGG16 A

- Commit 0a4eb77 đã push lên Thái; outputs/runs trước đó trống, không còn checkpoint lịch sử local.
- Lượt mới lan_1: VGG16 A, ImageNet-1K V1/H1/freeze, Adam .001, batch64/worker0, ngân sách20 epoch.
- Bắt đầu 19:53:28; CUDA được xác nhận, GPU quan sát3336 MiB/100%.
- Epoch1: train accuracy73,6333%, loss .988279; val accuracy79,62%, loss .671030; thời gian420,84 giây.
- Lượt train đã tạm dừng; best/last đều epoch1 có Adam/RNG, config trỏ resume lan_1 từ epoch2.
- Chưa có kết quả test hoặc sáu lượt hoàn tất; không xếp hạng hai model từ số đo hiện có.

## 2026-10-29 — Notebook và phạm vi thí nghiệm

- Ma trận mới: 2 model VGG16/DenseNet121 × A freeze .001, B fine-tune .001, C fine-tune .0001.
- H1/batch64/20 epoch, không scheduler/ensemble/TTA; B/C không dùng chung ý nghĩa với mã cũ.
- Notebook trình bày dữ liệu, EDA, preprocessing, kiến trúc, thí nghiệm, kết quả và giới hạn; đọc artifact local.
- Toàn bộ code cell với dữ liệu thật và nhánh thiếu artifact qua; không train/download/test; outputs ảnh không được lưu.
- best/last VGG16 A vẫn tương thích cấu hình mới; sáu tổ hợp config, compile/diff check và giới hạn Markdown qua.
- 5W1H ở đầu các file src; notebook/tài liệu không dùng template này.
- Markdown chuyển sang ghi chú kỹ thuật, nhật ký công việc; giữ bằng chứng và lịch sử đo được.

## Trạng thái và phần còn lại

- Train đang dừng; checkpoint cuối epoch1. Đợt bàn giao Git gồm notebook, tài liệu, source đồng bộ sáu lượt và log/config/history; weights/events/data giữ local.
- Notebook UI trong IDE chưa được kiểm tra; kiểm tra hiện có thực thi code cell trên CPU.
- Tiếp theo: hoàn thiện notebook, tiếp tục A của hai model, sau đó B/C; test chỉ sau khóa checkpoint bằng validation.
- Các mục đã tổng hợp vào [AI_Usage_Full.md](docs/AI_Usage_Full.md).
- Trạng thái chi tiết: [progress.md](progress.md). Thiết kế: [Experiment_Plan.md](Experiment_Plan.md).
