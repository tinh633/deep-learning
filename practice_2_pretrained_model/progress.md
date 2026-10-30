# Tiến độ và bàn giao — Practice 2

Cập nhật 2026-10-29: **đã dọn lan_1 cũ; sẵn sàng train mới theo kế hoạch sáu lượt.**
Config RESUME_RUN=None, VGG16 A, bắt đầu epoch1; không còn checkpoint local của lượt cũ.

## 1. Giai đoạn hiện tại

Dừng train, giảm còn 2 model × 3 cấu hình, tạo notebook báo cáo.

Phạm vi sáu lượt giảm thời gian thí nghiệm và giữ đủ đối chiếu trong checklist.

Notebook đọc artifact của các lượt chạy mới. Giải thích 5W1H chỉ đặt ở đầu file Python trong src, không áp thành mẫu cho tài liệu/notebook.

## 2. Việc đã làm và bằng chứng

- CIFAR-10 local còn đủ; split seed 42: 45.000/5.000/10.000, phân tầng và không giao ID.
- EDA report/RGB viewer: phân bố lớp, RGB train/histogram, preprocessing, 0 nhóm trùng pixel chính xác.
- HTML duyệt toàn bộ 60.000 ảnh, bộ lọc/preview; dữ liệu và weights không đưa lên Git.
- Pipeline 7 Python: config/data/model/logger/EDA/train/evaluate, vòng lặp học trực tiếp.
- Log/TensorBoard theo epoch, không live.jsonl/progress.json hoặc batch tracking.
- Resume giữ model/Adam/RNG/DataLoader, có best/last atomic, kiểm tra cấu hình và split.
- Trước tái cấu trúc: CPU train liên tục/ngắt-resume khớp weights; evaluate khớp sklearn/ID/PNG, test tắt và split sai bị từ chối.
- Mốc trước lượt VGG16 mới: 0a4eb77 đã push Thái. Đợt bàn giao tiếp theo gồm notebook báo cáo, ma trận sáu lượt, source đã dọn và artifact epoch1; mã commit đối chiếu Git history.
- Đã dừng launcher PID 2728 và Python PID 18928; GPU 0 MiB/0% tại thời điểm kiểm tra.
- Lượt VGG16 A cũ từng hoàn tất 1/20 epoch, val79,62%, loss .671030; chưa có kết quả test. Thư mục lan_1 đã xóa khi dọn bản bàn giao.
- Không còn best/last cũ để resume. Lượt mới khởi tạo từ pretrained và tạo lan_1 mới; giữ dataset/split/EDA local.
- Notebook mới gồm mục tiêu, EDA, preprocessing, model, ma trận, luồng train, artifact/curves, evaluate và kết luận/checklist.
- Đã thực thi toàn bộ code cell trên CPU với artifact local: qua; không train/download/evaluate test, không lưu outputs ảnh vào notebook.
- Kiểm tra nhánh thiếu dataset/EDA/runs/reports qua; A/B/C có đúng freeze/LR/batch cho sáu tổ hợp; compile src/diff check qua và Markdown tối đa 150 dòng. Chưa kiểm tra giao diện notebook trong IDE.

## 3. Kế hoạch mới

| Mã | Backbone | LR | Head/batch/epoch | Đối chiếu |
|---|---|---:|---|---|
| A | Freeze | .001 | H1 / 64 / 20 | Baseline |
| B | Mở nhóm cuối | .001 | H1 / 64 / 20 | A/B: fine-tune |
| C | Mở nhóm cuối | .0001 | H1 / 64 / 20 | B/C: LR |

Chạy mỗi hàng với VGG16 và DenseNet121, tổng 6 lượt. Không H2/ResNet trong lịch mới, không scheduler/ensemble/TTA.
model.py chỉ còn VGG16/DenseNet121 và Linear 10 lớp; state_dict VGG16 A được giữ nguyên.
- Đã bỏ tham số chọn head, thuộc tính head_name/HEAD/TRAIN dư thừa và fallback resume best/reset Adam. Train/evaluate/notebook đồng bộ API; metadata H1 vẫn giữ để nạp checkpoint hiện có.
- CPU kiểm tra hai backbone × freeze/fine-tune qua shape/gradient/BatchNorm; logits H1 khớp code trước khi dọn. Bản sao last.pt VGG16 A resume/lưu epoch tiếp theo qua; artifact gốc không đổi.
- Smoke tạm train/save/resume/evaluate theo API mới qua, validation ID/confusion đúng, test tắt; train chính thức vẫn dừng.
- Notebook có phân tích sau test: tỷ lệ theo lớp, ROC OvR, macro/micro AUC, bảng/curves so sánh; evaluate chỉ lưu xác suất đầy đủ cho test, không đổi RUN_TEST=False.
- Kiểm tra giả qua: perfect/chance AUC, rates, lớp vắng, thiếu/score sai, plots và từ chối khác split; smoke evaluate xuất Nx10 đúng thứ tự. Chưa chạy test CIFAR-10 chính thức.
- Toàn bộ code cell notebook mới chạy qua với artifact hiện tại chưa có test report; RUN_TEST=False, outputs trống, compile/diff và giới hạn Markdown qua.
- Đã thêm PR/per-class AP/macro-micro AP và paired error analysis theo ID: bảng bốn nhóm, theo lớp, ảnh local. Kiểm tra giả AP perfect/chance, ghép ID, nhãn sai/ID trùng, gallery và thiếu report qua; không chạy test thật.
- Notebook cell 7.0 đã có chạy evaluate qua script hiện có, chọn danh sách run và bật test riêng; mặc định chỉ đọc. Báo cáo được dò lại trực tiếp từ thư mục run, không phụ thuộc history/config.json.
- Smoke checkpoint-only tạm qua: đánh giá train/val và test có xác suất Nx10, config khôi phục khi thành công/lỗi, weights không đổi. Không evaluate checkpoint chính thức.
Mã B/C cũ khác B/C mới; phân loại artifact bằng model_config thực, không chỉ theo chữ A/B/C.

## 4. Việc tiếp theo, theo thứ tự

1. Kiểm tra nội dung notebook và cấu hình ma trận sáu lượt.
2. Train mới VGG16 A từ epoch1 với RESUME_RUN=None; lượt đầu tạo lan_1.
3. Evaluate train/validation của A, test vẫn tắt; chạy DenseNet121 A, rồi B và C của cả hai model.
4. Cập nhật notebook từ artifact thật; phân biệt run dở/chưa chạy/hoàn tất.
5. So sánh validation và khóa hai checkpoint trước khi bật test.
6. Hoàn thiện per-class/confusion/ảnh sai, chi phí train và nhận xét giới hạn một seed.
7. Cập nhật nhật ký, trạng thái Git và kết quả kiểm tra.

## 5. Lưu ý bàn giao

- Lượt ResNet18 A cũ từng đạt best val 88,54%, nhưng outputs/runs cũ đã trống trước lượt VGG16 hiện tại; chỉ là lịch sử, không ghép vào ma trận mới.
- Đừng nhầm lan_1 mới với lan_1 ResNet cũ. Nguồn nhận diện là config.json và checkpoint thực.
- Notebook không tự chạy train/test; cell kiến trúc tạo model ngẫu nhiên CPU chỉ để xem cấu trúc.
- Đã bỏ bố cục 5W1H khỏi tài liệu/notebook, thêm docstring ngắn ở đầu 7 file src; không đổi logic hay code cell. Compile qua, train vẫn dừng.
- Notebook trình bày mục tiêu/phương pháp/thí nghiệm/kết quả/giới hạn, tài liệu tham khảo ở cuối. Logic phân tích giữ nguyên; train vẫn dừng.
- Markdown đã chuẩn hóa thành ghi chú kỹ thuật; Full theo lịch sử ngày, Step theo giai đoạn, không kể hội thoại. Giữ thông tin đo được, lỗi, kiểm tra và bước tiếp theo.
- Xem Experiment_Plan.md, Training.md, Practice_2_Report.ipynb, AI_Usage_Step.md và docs/AI_Usage_Full.md.

<!-- VGG16_ABC_STATUS_20261030 -->
## 2026-10-30 — VGG16 A/B/C hoàn thành

| Run | Experiment | Epoch | Best epoch | Best validation accuracy |
|---|---|---:|---:|---:|
| lan_1 | VGG16 A | 20/20 | 20 | 85.62% |
| lan_2 | VGG16 B | 20/20 | 20 | 91.96% |
| lan_3 | VGG16 C | 20/20 | 19 | 93.24% |

### Kiểm tra artifact
- Đã khôi phục source project.
- A/B/C có đủ best.pt, last.pt, history.json và config.json.
- SHA256 checkpoint và file liên quan đã đối chiếu, khớp backup.
- Split SHA256 khớp giữa ba thí nghiệm.
- VGG16 C best validation loss: 0.32306 tại epoch 19.
- Test set chưa được đánh giá.
- DenseNet121 do thành viên khác phụ trách, chưa xác minh trong lần bàn giao này.

### Công việc tiếp theo
1. Hoàn thiện Practice_2_Report.ipynb.
2. Trực quan hóa train/validation loss và accuracy.
3. Phân tích overfitting, confusion matrix và per-class metrics.
4. Khóa checkpoint theo validation trước khi đánh giá test.
5. Tích hợp kết quả DenseNet121 khi có artifact.
6. Kiểm tra source và chuẩn bị GitHub.
