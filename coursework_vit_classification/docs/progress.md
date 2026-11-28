# Tiến độ coursework

Cập nhật 28/11/2026: **A/B/C hoàn tất pilot/final; D chưa có kết quả tích hợp.**
B/C do các thành viên khác train; local nhập/kiểm tra bản sao tên chuẩn, giữ nguyên gói nguồn bàn giao.
Trạng thái và giới hạn hiện tại: [Experiment_Results](Experiment_Results.md), [runs](../outputs/runs/README.md).
Lịch sử: [progress_before_A_2026-11-18.md](progress_before_A_2026-11-18.md).

## Phạm vi đã chốt

- CIFAR-100, ConvNeXt-Base A/B và ViT-B16 C/D, pretrained ImageNet-1K V1; không kết hợp deepfake.
- A chạy máy hiện tại; ba thành viên khác nhận B/C/D, mỗi người 2 pilot + 1 final.
- A: 2 pilot × 5 epoch (LR 1e-3, 3e-4), chọn bằng validation; final 20 epoch khởi tạo mới.
- A/B/C: batch16, AdamW, WD=0, không scheduler, seed42, AMP, workers0; nhập gói không chạy test mới. Gói C có test nguồn.
- Giữ split 45k/5k/10k và công bố 12 nhóm ảnh trùng xuyên split; không đổi dữ liệu giữa các lượt.
- Quy trình: [Training_Strategy](design/Training_Strategy.md), [Work_Allocation](design/Work_Allocation.md).

## Đã hoàn thành

- Code data/model/smoke/train/resume/evaluate/demo; EDA HTML; notebook theo dõi log epoch.
- 3 kiểm thử cấu trúc meta + integration CPU đạt trước khi đổi config train; có hash theo phiên bản kiểm tra.
- Xóa riêng lan_9 ResNet cũ theo yêu cầu; bằng chứng cũ ở Git 6bf7dcc, không phải ConvNeXt.
- Lan_9 mới: smoke lỗi tải mạng, 0 batch. Tải weights chính thức, checksum 6075fbad thành công.
- Lan_10: smoke dừng bảo vệ sau 1 batch vì Windows commit 2942,83 MiB (<3072).
- Được phép đóng ứng dụng khác: đã đóng Zalo/Discord; Chrome không đủ quyền, giữ VS Code/hệ thống.
- Lan_11: smoke A batch 16, 32 train/16 val, PASS đủ 12 checklist, save/load/replay/final guard.
- Peak train/workflow 511,22/607,22 MiB; commit cuối 3716,92 MiB. Không phải chất lượng train đầy đủ.

## Kết quả Exp A và trạng thái hiện tại

- Lan_12 LR1=1e-3 và lan_13 LR2=3e-4 đều đủ 5 epoch; best val 80,64% / 80,78%, chọn LR2.
- Lan_14 final đủ 20 epoch; best val 81,46% epoch 9, last val 80,94% / train 92,77%.
- Guard dừng giữa epoch 20 vì commit 3061,45 MiB; đã giải phóng bộ nhớ và resume từ last epoch 19.
- Lần dừng được lưu interruptions.json; history đủ 1–20, optimizer last đủ 56.242 update, tính cả 18 AMP skip.
- [Kiểm chứng thực](../outputs/validation/experiment_A_check.json): best/last, frozen tensors và RNG/history đạt.
- [experiment_A.json](../outputs/experiment_A.json) trạng thái completed; process train/monitor đã thoát, GPU về 0 MiB.
- [Kết quả A](A_Results.md); Training_Monitor.ipynb đọc được lan_12/13/14. Không mở test.
- Config giữ final/resume lan_14 nhưng TRAINING_APPROVED=False để tránh chạy lặp; EVAL_RUN=lan_14, validation.
- A xuất bản tại68641b3; hiện manifest21 checkpoint +1 replay reference, 11.975.178.650 byte. B lan_20–23, C lan_15–19; còn chờ D.

## Bàn giao khi bị ngắt

1. Kiểm tra experiment_A.json, PID thật, summary/history trước khi chạy mới; không chạy song song.
2. Run bị ngắt có last.pt: dùng RESUME_RUN cùng cấu hình/budget; không đổi LR hoặc train lại từ đầu.
3. Không chạy lại controller từ đầu khi đã có pilot; tiếp tục đúng bước còn thiếu.
4. Guard thiếu bộ nhớ: dừng, không hạ ngưỡng hoặc tự thay batch đã chốt.
5. A/B/C đã đủ budget; không chạy lại, tiếp theo nhận D và khóa protocol đánh giá; không dùng test nguồn C để tuning.

Audit: [Audit_Log](Audit_Log.md). Thay đổi pipeline: [CHANGELOG](CHANGELOG.md).

## 21/11/2026 — Nhập Exp C từ C_ban_giao

- Fast-forward cousework tới da0d199, giữ thay đổi tài liệu local; chuyển gói C vào outputs/runs, không train/test mới.
- Smoke nguồn lan_1/2 → lan_15/16; pilot nguồn lan_3/4 → lan_17/18; final nguồn lan_5 → lan_19.
- Hai pilot đủ 5 epoch, best val 80,56% / 81,84%; chọn LR2=3e-4. Final đủ 20 epoch, best val82,08% epoch9.
- Best/last final đã tải LFS, kiểm tra SHA/size và header epoch9/20, head100×768, split khớp local.
- [experiment_C.json](../outputs/experiment_C.json) dùng tên local; [import_C_manifest.json](../outputs/import_C_manifest.json) lưu nguồn và ánh xạ.
- Raw config/checkpoint/log/evaluation giữ nguyên byte, selected_from=lan_4 trong raw là pilot nguồn, tương ứng local lan_18.
- Không resolve tên raw này trực tiếp vào local lan_4; pilot C không kèm checkpoint, cần nguồn đầy đủ nếu muốn resume theo code hiện tại.
- Gói nguồn có test và resume Colab→Kaggle; chưa xác nhận khóa đủ bốn model trước test, chưa audit sâu frozen tensors.
- Kiểm chứng mọi file nhập và 14 run cũ không đổi; manifest hiện14 checkpoint. Chưa commit/push chuẩn hóa tên.

## 21/11/2026 — Rà soát lịch sử run

- Đối chiếu config/summary của lan_1–14, log lỗi, history final và Audit_Log/A_Results để giải thích vai trò từng lượt.
- Phân biệt smoke, pilot, final và lan_9 ResNet lịch sử đã xóa với lan_9 ConvNeXt hiện tại; không train hoặc đánh giá test mới.
- Bổ sung ý nghĩa các check PASS trong Audit_Log cho smoke lịch sử và lan_11, giữ giới hạn bộ nhớ của source cũ.
- Bổ sung bảng riêng lan_1–14 trong Audit_Log: nội dung đạt, lỗi/NOT_RUN và giới hạn mỗi run; ghi rõ lan_6 gán nhãn gradient FAIL nhưng traceback là memory guard.

## 25/11/2026 — Rà soát overfitting A/C

- Đối chiếu Define/Design, data/model/train và history/config/summary final A lan_14, C lan_19.
- A: best val 81,46% epoch9; epoch20 train92,77%/val80,94%, val loss0,6138→0,6474 từ best.
- C: best val82,08% epoch9; epoch20 train95,55%/val81,30%; min val loss0,6157 epoch7→0,7043 epoch20.
- Dấu hiệu overfitting cuối run, rõ hơn ở C; gap online train/val chỉ mô tả, chưa phải đánh giá cùng checkpoint/transform.
- Đề xuất giữ best epoch9 A/C, chờ kết quả B/D để hoàn tất ma trận; chưa phê duyệt train lại hoặc đổi protocol.
- B/D mở toàn backbone từ pretrained độc lập, không nối tiếp A/C; không mặc định fine-tune sẽ giảm overfitting.
- Gói C có test nguồn, trạng thái khóa đủ bốn checkpoint trước test chưa xác nhận; không dùng test để định hướng tuning.
- Chỉ rà soát artifact/source, không chạy train/evaluate/test mới; không sửa cấu hình hoặc commit.

## 27/11/2026 — Nhập B và đối chiếu lại C trên remote

- Fetch origin/cousework tới b754c2f; remote đã force-update, không merge thay đổi ngoài gói bàn giao vào workspace đang sửa.
- C: xác minh 33 file nhập trước và blob nguồn remote; không có phần mới, giữ lan_15–19.
- B_ban_giao: smoke → lan_20, LR1 → lan_21, LR2 → lan_22, final → lan_23; raw artifact giữ nguyên.
- Config/model B mới là ConvNeXt-Base fine-tune toàn bộ 87.668.964 tham số, không lấy tên ResNet trong commit cũ làm cấu hình.
- Pilot 5/5 epoch: LR1 best val87,84%; LR2 best val89,42%, chọn LR2=3e-5; final20/20 best val89,70% epoch13.
- LFS fetch không nhận dữ liệu; tải qua GitHub media theo commit và xác minh SHA-256/size cả8 file theo LFS pointer.
- import_B_manifest.json lưu mapping, hash/file và kiểm tra header; experiment_B.json dùng tên local, checkpoints.json gồm21 checkpoint +1 replay reference.
- Raw B selected_from=lan_25 tương ứng LR2 nguồn/local lan_22; không resolve thành run local cùng tên trước khi áp dụng mapping.
- Kiểm tra metadata checkpoint/head100×1024/split/epoch; 19 run cũ không thay size/mtime, C kiểm tra thêm SHA-256.
- Không chạy train/inference/test; cập nhật README/Checkpoints/AI usage; chưa commit/push, local HEAD giữ nguyên.

## 27/11/2026 — Chỉ mục phân loại run

- Thêm outputs/runs/README.md: đủ23 run, tách3 final,6 pilot,14 smoke; ghi LR/validation/best epoch và trạng thái thực.
- Ghi chú ResNet lịch sử, lan_9 tái sử dụng tên, resume A/C, mapping bàn giao và giới hạn test C.
- Đối chiếu config/history/summary, kiểm tra liên kết local và giới hạn150 dòng; chỉ đổi tài liệu, không train/test mới.

## 27/11/2026 — HTML trình bày theo PDF ViT

- Đọc PDF [012012710503] gồm2 trang; soạn visualization/presentation.html theo đủ10 nội dung, ví dụ tensor và checklist sản phẩm nộp.
- HTML một file không CDN: mục lục, đọc/trình chiếu14 phần, phím điều hướng, in PDF, patch/encoder/attention minh họa.
- Nhúng snapshot60 epoch thật từ final A/B/C, có nút đọc lại JSON qua HTTP; không thực thi train hoặc inference.
- Để rõ phần thiếu: D, bộ10–20 ảnh/Top-5 thật, che vùng nhiều lần,≥3 phân tích và notebook demo; không tạo dự đoán/heatmap giả.
- Giao diện che ảnh hỗ trợ ảnh local, ghi rõ chỉ preview; canvas tự vẽ không nhúng pixel dataset.
- Phân biệt weights ImageNet gốc và checkpoint CIFAR100 về preprocessing/tập lớp; có nguồn paper/mã/data và PDF yêu cầu.
- Dùng hướng dẫn Sites cho thiết kế/local preview; không đăng ký hosting, commit hoặc publish.
- Edge headless PASS desktop1440/mobile390, không tràn ngang/lỗi JS; flow, patch, attention, chart, refresh, upload/mask và trình chiếu đạt.
- Kiểm tra file mode snapshot, HTTP JSON, phím khi focus nút, liên kết/IDs và snapshot khớp history thực đạt.
- Bản in thử chỉnh từ25 còn15 trang A4 ngang, đủ14 phần, không trang trắng; PDF/screenshot QA ở thư mục tạm.
- Thêm docs/design/Presentation.md và liên kết README; Markdown≤150 dòng, git diff --check đạt (có cảnh báo CRLF).
- Công cụ PDF/browser cài thư mục tạm; không thêm dependency runtime, không sửa code train/test. Chưa commit/push.

## 28/11/2026 — Rà soát docs và đồng bộ nhánh

- Xác nhận B/C do các thành viên khác train; bổ sung provenance trong docs, wrapper/manifest và HTML, không đổi raw artifacts.
- Đồng bộ README gốc/project, Define/Design/source docs và mô tả notebook; thêm Experiment_Results với số liệu/giới hạn hiện tại.
- Coursework_Report còn0 cell; D, bộ10–20 ảnh/Top-5/che vùng/≥3 phân tích và notebook demo vẫn chưa đủ.
- Kiểm tra22 checkpoint/reference khớp size/SHA256, toàn bộ raw B/C khớp manifest; AST source/notebook và final history đạt.
- Giữ B_ban_giao/C_ban_giao nguyên trạng remote, phục hồi bản local; chuẩn bị commit nối tiếp cả remote b754c2f và lịch sử local, không force.
- Chuẩn hóa ignore/LFS, giữ data/cache ngoài Git; kiểm tra links/giới hạn150 dòng và staged tree trước push.
- Được chỉ định commit/push trong phiên này; trạng thái thực sau push xem Git log và remote, không coi ghi chép chuẩn bị là đã phát hành.
