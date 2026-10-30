# Tiến độ và bàn giao — Practice 2

Cập nhật 2026-10-30: đủ sáu lượt train, 12 checkpoint và báo cáo train/validation/test.
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

## Việc còn lại để bàn giao

1. Kiểm tra bản đồ60.000 ảnh và chạy notebook sau các chỉnh sửa trình bày cuối.
2. Mở notebook/HTML để review bố cục; không chạy lại train hoặc chọn lại cấu hình theo test.
3. Review thay đổi Git và quyết định commit/push riêng khi có yêu cầu.

Chi tiết kiểm tra cuối được cập nhật bên dưới. Nhật ký đầy đủ: docs/AI_Usage_Full.md.

- Final map check: 60000 points; PCA2 retains 48.89% train RGB8x8 variance. Numeric and DOM/canvas tests passed; model-projection script/output removed.

- Final notebook rerun passed all17 code cells after map correction; saved numeric/metric plot outputs, omitted dataset pixels. No training/download/inference rerun. Remaining work is visual review and explicitly authorized Git publication.
