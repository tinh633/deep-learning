# Tiến độ và bàn giao — Practice 2

Cập nhật2026-11-04: rà soát trước Git; đủ sáu lượt train,12 checkpoint và báo cáo ba split.
Đợt hoàn thiện: chuẩn hóa tên weights, đối chiếu artifact, bổ sung trực quan hóa và báo cáo.
Nhánh chốt: `practice2-pretrained-model`; đã được duyệt commit/push kèm đủ 12 checkpoint qua Git LFS.

## Trạng thái thực nghiệm

| Run | Model/cấu hình | Best / last epoch | Test |
|---|---|---|---:|
| lan_1 | DenseNet121 A | 10 / 20 | 90,01% |
| lan_2 | DenseNet121 B | 7 / 20 | 93,06% |
| lan_3 | DenseNet121 C | 16 / 20 | 93,48% |
| lan_4 | VGG16 A | 20 / 20 | 84,48% |
| lan_5 | VGG16 B | 20 / 20 | 91,91% |
| lan_6 | VGG16 C | 19 / 20 | 93,10% |

- Tất cả 20/20 epoch. A freeze; B fine-tune .001; C fine-tune .0001; H1/batch64/seed42.
- DenseNet nguồn feature/thuan; VGG nguồn vgg16-results-2026-10-30 lan_1/2/3 → local lan_4/5/6.
- VGG best (1)/(2), last (1)/(2) đã đổi về best.pt/last.pt; hash trước/sau đổi tên không đổi.
- Sáu best/last đúng model/cấu hình/epoch/split; sáu best nạp strict vào model hiện tại qua.
- Các báo cáo ba split khớp SHA256 best; không cần test lại VGG.
- DenseNet A/B/C đã đánh giá local. C nguồn93,49%, local93,48%; không sửa trọng số để khớp điểm.

## Đã có

- Dataset local, split45k/5k/10k phân tầng, không giao ID; SHA ổn định qua .gitattributes LF.
- EDA sâu: split/RGB/duplicates, bốn đặc trưng ảnh, phân vị/lớp/tương quan và train extremes.
- HTML một trang: EDA, đặc trưng, dataset, preprocessing/quality, TensorBoard, lỗi test theo A/B/C.
- Bản đồ dataset duy nhất: đủ60.000 ảnh trên một canvas, PCA RGB8×8 fit train và transform cả ba split; nằm trong 1_eda.py, không có script4 hoặc model inference.
- Notebook: bảng sáu run/cost/params, train-val riêng, test bars, confusion ba split, ROC/PR và paired errors.
- Bảng paired errors khớp accuracy: DenseNet hơn VGG A5,53 / B1,15 / C0,38 điểm phần trăm.
- TensorBoard local có events lan_2/lan_3. Thiếu events gốc ở các run khác; history vẫn đầy đủ.
- Quy tắc lưu: dataset/derived pixel giữ local; best/last sáu run lưu qua Git LFS, events gốc hiện có được đưa lên nhánh bàn giao.

## Kiểm tra và lưu ý

- Đã kiểm tra numerical reports, split/checkpoint SHA, confusion/counts/scores và strict model load.
- Node DOM harness: gallery filter/sort/page/race, RGB decoding, EDA tables/charts, projection hover/filter qua.
- Model, checkpoint, split và vòng train không bị thay đổi bởi các bước trình bày.
- Notebook mặc định không train/evaluate/download. RUN_EVALUATION/EVALUATE_TEST giữ False.
- Chưa kiểm tra trực tiếp bố cục HTML trong trình duyệt hoặc UI notebook trong IDE.

## Trạng thái bàn giao hiện tại

Code/tài liệu/artifact đã qua rà soát2026-11-04; chưa commit/push thay đổi mới.
Chỉ đưa Practice2 và .gitignore liên quan; không git add toàn workspace vì có benchmark riêng.
UI HTML/notebook trong IDE chưa kiểm tra trực tiếp; đã có kiểm tra DOM và thực thi notebook.

Chi tiết kiểm tra cuối được cập nhật bên dưới. Nhật ký đầy đủ: docs/AI_Usage_Full.md.

- Final map check: 60000 points; PCA2 retains 48.89% train RGB8x8 variance. Numeric and DOM/canvas tests passed; model-projection script/output removed.

- Final notebook rerun passed all17 code cells after map correction; saved numeric/metric plot outputs, omitted dataset pixels. No training/download/inference rerun. Remaining work is visual review and explicitly authorized Git publication.

- 2026-10-30: Corrected branch scope: practice2-pretrained-model publishes Practice 2 only. Untrack Practice 1 on this branch while preserving all local files and other branches; root README/index updated. No model or result changes.

## 2026-11-01 — Tài liệu source

- Đối chiếu bảy file Python và viết bảy hướng dẫn tại docs/src/: config, data, model, logger, EDA, train, evaluate.
- Mô tả tham số, luồng xử lý, artifact, freeze/BatchNorm, resume cuối epoch và điều kiện kiểm tra dữ liệu/checkpoint.
- Thêm liên kết từ README; chỉ sửa tài liệu, không train/evaluate hoặc thay weights/code.

## 2026-11-02 — Không gian ảnh và TensorBoard

- Thử t-SNE không nhãn trên60.000 ảnh,500 iteration, KL4,1496; quan sát vẫn trộn lớp.
  Giữ PCA hiện có, không đưa t-SNE vào pipeline; thêm độ đậm/chỉ hiện một lớp, không đổi tọa độ.
- Bỏ iframe TensorBoard tự tải trong HTML, giữ link; thêm TensorBoard.ipynb riêng.
- Notebook đọc events thật, vẽ loss/accuracy không cần server, khởi động server có health check,
  link tab riêng và dashboard. Tránh metadata server cũ; hiện có events lan_2/lan_3.
- Node DOM kiểm tra60.000 điểm, lọc lớp6.000/train4.500, hover/click qua.
- Bốn cell notebook đã chạy qua; HTTP200 và scalar tags đúng2 run. Chưa kiểm tra UI trong IDE/browser.
- Không train/evaluate model hoặc đổi weights/split. Không commit/push.

## 2026-11-04 — Rà soát trước Git

- Sửa Training.md về12 weights GitLFS, resume và TensorBoard riêng; bổ sung dependencies notebook.
- Loại pixel/base64 khỏi eda.json và EDA generator. Preview HTML/notebook dựng từ dataset local;
  source report chỉ giữ ID/cấu hình/thống kê. Chỉ xóa outputs ảnh ở cell5/7/31 trước lưu Git.
- Ignore tensorboard_server.log; giữ dataset/RGB local. Không sửa checkpoint hoặc kết quả train/test.
- Kiểm tra12 checkpoint khớp LFS SHA/size, best/last epoch, sáu strict model loads và18 reports;
  confusion/counts/accuracy/per-class/probabilities khớp. Split SHA khớp toàn bộ checkpoint.
- Chạy17 cell report +4 cell TensorBoard thành công; flags evaluate/test giữFalse.
- DOM/JS kiểm tra60k điểm/lọc/hover, preprocessing4 canvas và thiếu pixel qua.
- Giới hạn đã ghi: batch64 cố định, events chỉ2/3, một seed/nhiều máy và test đã quan sát.

- 2026-11-04: Publication authorized: Practice2 and its .gitignore change to practice2-pretrained-model; ImageNet benchmark excluded.
