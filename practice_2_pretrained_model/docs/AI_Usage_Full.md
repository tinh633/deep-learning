# AI Usage Full — Nhật ký kỹ thuật Practice 2

Lịch sử triển khai, kiểm tra và kết quả các hạng mục có hỗ trợ AI.
Công cụ sử dụng: Python/PyTorch, Git, kiểm tra tiến trình/GPU, Node và Edge headless.
Trạng thái hiện tại nằm trong [progress.md](../progress.md); ghi chú giai đoạn tại [Step](../AI_Usage_Step.md).
Các lượt thử, lượt lỗi và kết quả lịch sử không thay thế kết quả của ma trận thí nghiệm hiện hành.

## 2026-10-26 — Thiết kế và cấu trúc ban đầu

- Chọn CIFAR-10, VGG16/ResNet18/DenseNet121; thiết kế 18 lượt = 3 model × 6 cấu hình, điều kiện so sánh đồng bộ.
- Tạo Experiment_Plan.md và README.md; mô tả EDA, head H1/H2, freeze/fine-tune, LR/batch và giới hạn so sánh.
- Tổ chức src/data/outputs/visualization; pipeline 7 Python, model/config/logger riêng, train/evaluate trực tiếp.
- Khởi tạo config/data và mô tả các bước còn lại; import module không tự tải dữ liệu hoặc chạy tác vụ.
- Kiểm tra cú pháp, split lặp lại/coverage/không giao ID/phân bố lớp, từ chối cấu hình không tương thích: qua.
- Kiểm tra liên kết Markdown, whitespace và giới hạn 150 dòng/file; chưa có kết quả model ở giai đoạn thiết kế.

## 2026-10-26 — Dataset và EDA

- Tải CIFAR-10 bằng torchvision; một lần tải dừng vì DNS ở 119.767.040 byte, sau đó curl resume/retry hoàn tất.
- Archive được kiểm tra và giải nén; đọc đủ 50.000 train và 10.000 test. Phiên tải khôi phục kết thúc exit 0.
- Tạo outputs/cifar10_split.json: 45.000 train/5.000 validation, phân tầng seed 42, không giao ID.
- Triển khai 1_eda.py: counts, shape/dtype/pixel range, RGB train mean/std/histogram, preprocessing và duplicates SHA-256.
- EDA thật: lớp cân bằng, 0 nhóm trùng pixel chính xác; đối chiếu 360 thumbnail/nhãn, histogram và RGB độc lập qua.
- DataLoader cả ba split cho batch 64×3×224×224 hữu hạn, nhãn đúng phạm vi; không dùng smoke làm kết quả model.
- Viewer HTML ban đầu có lọc split/lớp, biểu đồ, HTTP fetch và file picker; fixture giả/missing-report, Node syntax và Edge DOM qua.
- Kiểm tra Edge ngoài sandbox sau lỗi GPU/access trong sandbox; HTTP với report thật render bảng/gallery/histogram qua.

## 2026-10-26–27 — Hoàn thiện viewer

- Mở rộng gallery đủ 60.000 ảnh thành 236 atlas PNG lossless; JSON giữ ID/nhãn theo split.
- Bổ sung filter tất cả split/lớp, ID/khoảng ID, sắp xếp, 50/100/200 ảnh mỗi trang và preview pixel/mượt.
- Đối chiếu toàn bộ pixel/nhãn với CIFAR-10; Edge kiểm tra filter, query không hợp lệ, trang cuối, preview, desktop/mobile qua.
- Đồng bộ bảng màu teal, header/menu ngang, chiều rộng và thẻ/bảng với viewer Practice 1; bỏ sidebar.
- Sau đó thay atlas bằng 6 RGB gzip HWC uint8 và canvas; giữ archive/Python batches, xóa 236 atlas đã xác minh đúng đường dẫn.
- Kiểm tra lại đủ 60.000 pixel/nhãn/hashes, chunk boundaries, local-file offline, phân trang và mobile width 390: qua.
- Không còn cần xuất hàng chục nghìn PNG riêng; dữ liệu viewer được tái tạo local bằng 1_eda.py.

## 2026-10-26–27 — Model, logging và evaluate

- PretrainedCNN có backbone/head rõ ràng, H1 Linear hoặc H2 MLP; freeze trước, mở nhóm cuối nếu fine-tune.
- Override train() giữ BatchNorm running statistics; kiểm tra 12 tổ hợp model/head/freeze bằng weights ngẫu nhiên qua.
- Logits, gradient hữu hạn đúng nhóm, optimizer update, frozen weights/BN và eval mode được kiểm tra; SGD một step chỉ là kiểm tra cơ chế.
- Tách ghi chú H1/H2, số tham số, hidden 256/Dropout .2 và giới hạn so sánh vào Model_QA.md.
- Triển khai logger/train: môi trường, kiến trúc/config/loss/optimizer, validation, best checkpoint và TensorBoard 2.21.0.
- A/E và H2 smoke CPU trên 8 train/4 validation CIFAR thật, 2 epoch; checkpoint reload/event scalars qua.
- Kiểm tra lỗi/interrupt khi setup và giữa epoch 2: không báo Finished giả, giữ epoch 1; artifact tạm được dọn.
- Triển khai evaluate theo config checkpoint, strict state load, split SHA-256; train evaluation bỏ augmentation.
- Báo cáo loss/accuracy/macro-F1, per-class/TP/TN/FP/FN, confusion, prediction ID/confidence và tối đa 24 ảnh sai tự tin nhất.
- Evaluate smoke khớp checkpoint/sklearn/zero-support/ID/PNG; test tắt không iterate, test giả 5 ảnh qua khi bật.
- Split hash sai bị từ chối, giữ báo cáo cũ; chưa có test chính thức từ các smoke này.

## 2026-10-27 — Môi trường CUDA và bàn giao

- Giữ .venv chính, thay torch CPU bằng torch 2.11.0+cu126 và torchvision 0.26.0+cu126; pip check qua.
- Dừng lượt CPU trước khi đổi DLL, giữ trạng thái gián đoạn; thêm main guard train/evaluate cho Windows worker 2.
- Smoke worker ban đầu lỗi OpenBLAS thiếu RAM; giới hạn BLAS/OMP/MKL threads rồi kiểm tra lại.
- GPU pretrained ResNet18 A, batch64/worker2, CIFAR 128 train/64 validation qua; peak CUDA allocated 488,8 MiB.
- Chưa kiểm tra GPU cho mọi tổ hợp tại thời điểm này; kết quả smoke không phải lượt train đầy đủ.
- Dọn .venv-gpu tạo dở; giữ .venv CUDA. Tạo progress.md riêng, shared progress chuyển thành chỉ mục.
- Bàn giao source/tài liệu trên nhánh Thái; lượt CPU dở không được ghi nhận như kết quả GPU chính thức.

## 2026-10-28 — ResNet18 A và lỗi worker

- Lượt ResNet18 A đầu: H1/freeze, Adam .001, batch64/worker2/20 epoch, 45k train/5k validation, CUDA.
- Log xác nhận 5.130 tham số trainable; quan sát sau khởi động GPU 58%, 786 MiB; chưa phải GPU peak của cả run.
- Hoàn tất 6 epoch rồi lỗi shared mapping Windows 1455 giữa epoch 7; exit 1, best epoch5 val88,20%, loss .339812.
- Đổi worker2 → 0, chạy lại từ pretrained tại lan_2, giữ LR/head/batch; không resume lượt lỗi.
- Lượt mới đủ 20 epoch, best epoch18 val88,54%, loss .337606; xóa lượt lỗi và đổi tên lượt thành công lan_2 → lan_1.
- Giữ log lịch sử kèm ghi nhận đổi tên. Evaluate train89,6533%, val88,54%, val macro-F1 .885577; counts/IDs/lỗi khớp, test tắt.
- Kết quả ResNet này là lịch sử; artifact đã không còn trong workspace trước lượt VGG16 mới ngày 29/10.

## 2026-10-28 — Weights, quản lý dữ liệu và nhật ký

- Tải VGG16 chính thức bằng HTTP ranges/CDN download-r2.pytorch.org do đường tải đơn chậm; dừng các phép tải trùng.
- Hoàn tất cache và kiểm tra SHA-256; pipeline cũ lỗi quyền đọc cache, chạy lại với quyền đọc phù hợp.
- GPU VGG16 preflight batch64 qua, peak 2445,9 MiB; lan_2 bắt đầu 16:00:25, pretrained ImageNet-1K V1/CUDA/worker0.
- Định danh ImageNet-1K: 1.000 lớp; V1 là phiên bản weights. Đối chiếu DEFAULT và metadata torchvision local.
- Chuẩn hóa tên nhật ký: AI_Usage_Step.md theo giai đoạn, docs/AI_Usage_Full.md giữ lịch sử; đồng bộ liên kết.
- Gỡ 6 RGB gzip đã theo dõi bằng Git --cached, giữ local; bỏ ngoại lệ data ignore và không rewrite lịch sử.
- Commit 4bdb86b đã push Thái; data, weights, events và ảnh lỗi dẫn xuất được giữ local.

## 2026-10-28 — Resume và theo dõi train

- VGG16 lan_2 cũ dừng sau epoch2; best cũng epoch2 nhưng thiếu Adam/RNG. Tiếp tục thử từ epoch3 với Adam/random reset có cảnh báo.
- Bổ sung best/last full-state và resume cùng run; smoke CPU save/restore/legacy qua. Không tái tạo last epoch20 của ResNet từ best epoch18.
- Từng bổ sung scalar/snapshot theo batch; sau đó trở lại log/TensorBoard cuối epoch, bỏ live.jsonl/progress.json.
- Dừng/xóa lượt VGG16 đang dở và train lại từ đầu; giữ ResNet tại giai đoạn đó. Lượt VGG16 mới dừng sau epoch1, best/last có Adam/RNG.
- HTML nhúng TensorBoard, bỏ dashboard/JS train riêng. Sửa ký tự tiếng Việt và thêm Live Server ignore outputs/data/events.
- TensorBoard HTTP/API scalars từng kiểm tra 200 và nhận hai run; iframe/Live Server reload chưa kiểm tra trực tiếp trong IDE.

## 2026-10-29 — Làm gọn source và chạy mới

- Làm rõ config bằng hằng số, tham số model theo keyword, get_settings() giữ format checkpoint; vòng train/validation trực tiếp.
- Gom JSON/hash/đóng log; bỏ registry không dùng; chuyển resume/save epoch sang helper, tách vẽ ảnh sai khỏi evaluate.
- Evaluate đọc full-state checkpoint chứa NumPy RNG; DataLoader training=False không augmentation/shuffle.
- CPU continuous và ngắt/resume khớp weights; best/last/history qua; cấu hình/split khác bị từ chối.
- Metrics sklearn/ID/PNG/test tắt qua; CIFAR thật giữ 45k/5k/10k và preprocessing đúng chế độ.
- Compile và diff check qua. Commit a7a6af8 rồi 0a4eb77 đã push Thái; weights/events/pixel data không đi kèm.
- Trước push 0a4eb77, outputs/runs đã trống, không còn checkpoint để resume; Git ghi nhận xóa các báo cáo cũ.
- Khởi chạy mới VGG16 A tại lan_1 lúc 19:53:28, pretrained/H1/freeze, Adam .001, batch64/worker0, ngân sách20 epoch.
- Log/config xác nhận CUDA; quan sát GPU3336 MiB/100%. Epoch1: train73,6333%, loss .988279; val79,62%, loss .671030, 420,84 giây.
- Dừng PID2728/18928 giữa lượt train; best/last đều epoch1, có Adam/RNG. GPU về 0 MiB/0%; resume tiếp từ epoch2.

## 2026-10-29 — Phạm vi sáu lượt và notebook báo cáo

- Thu gọn 18 → 6 lượt: VGG16/DenseNet121 × A freeze .001, B fine-tune .001, C fine-tune .0001; H1/batch64/20 epoch.
- B/C mới khác mã cũ; phân loại artifact bằng model_config thực. Giữ nhánh ResNet/H2 lịch sử, không đổi state_dict VGG16 A.
- Tạo Practice_2_Report.ipynb: mục tiêu, EDA/ảnh train, preprocessing, kiến trúc CPU, thiết kế, history/curves, metrics và giới hạn.
- Thực thi toàn bộ code cell local và nhánh thiếu artifact qua; không train/download/test. Notebook không lưu outputs ảnh dataset.
- Kiểm tra best/last epoch1 tương thích A mới; sáu tổ hợp có đúng freeze/LR/batch; compile/diff và Markdown ≤150 dòng qua.
- Biên tập notebook theo văn phong báo cáo, giữ tham khảo ở cuối; chưa kiểm tra UI notebook trong IDE.
- 5W1H chỉ ở docstring đầu 7 file src; Markdown/notebook trình bày theo nội dung, không áp template.
- Chuẩn hóa Markdown thành ghi chú kỹ thuật và nhật ký công việc; giữ kết quả/lỗi/giới hạn, bỏ phần kể hội thoại.
- Trạng thái giai đoạn: train dừng, config resume lan_1; các thay đổi sau 0a4eb77 được gom vào đợt bàn giao notebook/source/tài liệu tiếp theo, mã commit theo Git history.

## 2026-10-29 — Loại nhánh source không còn sử dụng

- Bỏ ResNet18/H2 và tham số head; chỉ VGG16/DenseNet121 với Linear 10 lớp, weights ImageNet-1K V1 khai báo rõ.
- Bỏ head_name/HEAD/TRAIN dư thừa; metadata H1 giữ format checkpoint. Train/evaluate/notebook đồng bộ lời gọi.
- Resume bắt buộc last.pt với Adam/RNG, bỏ fallback best/reset optimizer. Không xóa checkpoint hoặc dữ liệu.
- CPU: shape/gradient/BatchNorm hai model × freeze/fine-tune qua; logits khớp H1 trước dọn; bản sao checkpoint thật resume/lưu epoch tiếp theo qua.
- Smoke tạm train/save/resume/evaluate API mới qua; validation ID/confusion khớp và test tắt. Không khởi chạy run chính thức.

## 2026-10-29 — Chỉ số sau test và ROC-AUC

- Đối chiếu LAB2 metrics và scikit-learn; notebook bổ sung TP/TN/FP/FN, TPR/FPR/specificity, per-class OvR ROC và macro/micro AUC.
- Phân biệt tỷ lệ argmax với ROC quét ngưỡng; so sánh yêu cầu cùng split/ID/nhãn/lớp, không tuning bằng test. Thiếu score hoặc lớp vắng không tạo AUC giả.
- Evaluate thêm per-class rates và probabilities Nx10/thứ tự cột riêng test; không đổi RUN_TEST=False hoặc chạy test thật.
- Kiểm tra giả perfect/chance AUC, rates, missing/invalid score, lớp vắng, plots và split mismatch qua; smoke export score/rates qua. Artifact tạm, notebook output-free.

## 2026-10-29 — PR/AP và đối chiếu lỗi cùng ID

- Notebook có PR OvR, per-class/macro/micro AP, bảng bốn nhóm đúng/sai giữa hai checkpoint, thống kê theo lớp và gallery ảnh test local.
- Ghép theo ID/nhãn/split/lớp; không tự chọn checkpoint theo test, trường hợp nhiều report cần PAIR_LABELS. AP dùng average_precision_score, không nhầm diện tích hình thang.
- Kiểm tra giả perfect/chance AP, ID alignment, nhãn sai/ID trùng, gallery và thiếu report qua; chưa chạy test chính thức, không lưu outputs ảnh.

## 2026-10-29 — Cell thực thi đánh giá checkpoint

- Notebook 7.0 chọn danh sách run, bật evaluate và test riêng (mặc định False), thực thi 3_evaluate.py; cấu hình chỉ thay tạm và được khôi phục.
- Checkpoint đầy đủ model_state/settings/classes/weights/epoch cùng dataset/split local đủ để đánh giá; reader không cần history/config.json để đọc report mới.
- Smoke tạm no-op/train-val/test Nx10/config restore/checkpoint unchanged qua; không đánh giá test thật hoặc train lại model.

## 2026-10-29 — Bản bàn giao train mới
- Xóa outputs/runs/lan_1 sau khi xác minh không có tiến trình train; bỏ cả weights/events và log/config/history cũ, giữ dataset/split/EDA.
- Đặt RESUME_RUN=None cho VGG16 A từ epoch1; notebook cell đánh giá giữ hai cờ mặc định False. Chuẩn bị commit bản bàn giao, không khởi chạy train.

<!-- VGG16_ABC_FULL_20261030 -->
## 2026-10-30 — Kết quả và xác minh VGG16 A/B/C

### Cấu hình và kết quả

| Experiment | Fine-tune | LR | Batch | Epoch | Best epoch | Best val accuracy |
|---|---|---:|---:|---:|---:|---:|
| A | False | 0.001 | 64 | 20/20 | 20 | 85.62% |
| B | True | 0.001 | 64 | 20/20 | 20 | 91.96% |
| C | True | 0.0001 | 64 | 20/20 | 19 | 93.24% |

### Khôi phục artifact

- A lấy từ backup date30-10.
- B lấy từ practice2-vgg16-b-backup.
- C lấy từ practice2-vgg16-c-backup.
- Đã khôi phục source và ba run vào project.
- best.pt, last.pt, history.json và config.json được đối chiếu SHA256.
- Split SHA256: fa5a9fcd6b5d9d717eb35b10c728300f1a01c666c4e524a6faa3de47fc7ce5c3.
- C đạt best validation accuracy 93.24% tại epoch 19, validation loss 0.32306.
- Epoch 20 của C đạt validation accuracy 92.68%.

### Nhận xét và giới hạn

- C cao hơn A 7.62 điểm phần trăm và cao hơn B 1.28 điểm phần trăm về validation accuracy.
- Train accuracy cuối C là 99.44%, cao hơn validation accuracy; cần xem đường cong để đánh giá overfitting.
- Chưa có kết quả test chính thức.
- Thí nghiệm sử dụng một seed, chưa đánh giá biến thiên giữa các seed.
- DenseNet121 do thành viên khác phụ trách, chưa có artifact để tổng hợp.

### Công việc tiếp theo

- Hoàn thiện notebook báo cáo từ artifact thật.
- Vẽ learning curves và so sánh A/B/C.
- Hoàn thiện confusion matrix, per-class metrics và phân tích ảnh sai.
- Khóa checkpoint bằng validation trước khi đánh giá test.
- Tích hợp DenseNet121 và hoàn thiện so sánh hai kiến trúc.
- Chuẩn bị source và tài liệu để push GitHub.
