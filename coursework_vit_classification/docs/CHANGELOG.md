# Changelog — Thay đổi đáng kể về code và pipeline

Ghi vấn đề/nhu cầu, thay đổi, lý do và cách kiểm chứng; không ghi từng chỉnh sửa nhỏ.
Trạng thái hiện tại: [progress](progress.md). Checklist smoke: [Audit_Log](Audit_Log.md).
Audit chỉ ghi kết quả kiểm tra; lịch sử hỗ trợ AI nằm ở [AI_usage](AI_usage.md).

## 28/11/2026 ? T?ch h?p artifact v? trang tr?nh b?y

- B/C do th?nh vi?n kh?c train; b?n sao t?n chu?n ? lan_15?23, g?i ngu?n gi? nguy?n tr?n remote.
- Th?m manifest/provenance, ch? m?c23 run v? k?t qu? A/B/C; s?a m? t? hi?n tr?ng l?i th?i trong docs/notebook.
- Th?m HTML tr?nh b?y theo PDF, t??ng t?c minh h?a v? snapshot th?c; demo/D c?n thi?u, kh?ng s?a code train/evaluate.
- Chu?n h?a Git LFS/ignore cho checkpoint nh?p v? B replay, gi? data/cache ngo?i Git; ki?m ch?ng ? progress.

## 18/11/2026 — Khởi chạy A được duyệt

- Xóa lan_9 ResNet, giữ lịch sử trong Git; tên lan_9 mới là lỗi tải mạng, không dùng như bằng chứng cũ.
- Tải weights với quyền mạng ngoài sandbox; đóng Zalo/Discord theo chỉ dẫn, không đóng Code/hệ thống.
- A smoke batch 16 đạt sau giải phóng commit; bật config A pilot theo protocol đã duyệt.
- Controller tạm local chạy 2 pilot và final tuần tự, tự dừng khi child lỗi hoặc GPU/disk vượt ngưỡng.
- Sửa kiểm tra disk của controller dùng str(path) theo psutil Windows; lỗi xảy ra trước khi run train được tạo.
- Tách lịch sử progress sang file liên kết để giữ status hiện tại ngắn, không vượt 150 dòng.
- Guard dừng giữa final epoch20; giữ checkpoint19, đóng lại ứng dụng theo quyền đã có và resume cùng run đủ epoch20.
- Không sửa loop hoặc giảm guard; giữ interruptions.json và log; cấu hình sau hoàn tất chặn train lặp.

## 18/11/2026 — Thay CNN baseline bằng ConvNeXt-Base

- A/B chuyển sang convnext_base; classifier[2] là Linear(1024,100), chỉ mở Linear ở A.
- Giữ LayerNorm pretrained; freeze tắt stochastic depth, fine-tune bật theo torchvision.
- Smoke kiểm tra cả LayerNorm cuối bất biến; final từ chối pilot khác model/strategy/weights.
- Đồng bộ Define/Design và source docs, thay ResNet50.md bằng ConvNeXt_Base.md.
- Không sửa log/checkpoint ResNet cũ hoặc coi chúng là bằng chứng ConvNeXt; cần GPU smoke mới.
- Kiểm chứng: [report cặp mới](../outputs/validation/convnext_pipeline_check.json); meta và synthetic CPU, không phải train thật.

## 05/11/2026 — Chuẩn bị dữ liệu và viewer

- Nhu cầu: dữ liệu/split nhất quán và quan sát toàn CIFAR-100 trước train.
- Thay đổi: config/data/EDA, split phân tầng seed 42 và HTML đọc report/RGB gzip local.
- Sửa stdout Windows sang UTF-8 để script không lỗi khi in tiếng Việt.
- Kiểm chứng: checksum archive, IDs/cân bằng split, toàn bộ RGB bytes, labels và thống kê train đạt.
- Browser: gallery, bộ lọc, phân trang và ảnh nguồn đã kiểm tra; không chạy model/train.

## 05/11/2026 — Phép chiếu dữ liệu

- Nhu cầu: quan sát phân bố ảnh và mức chồng lấn lớp trong không gian 2D.
- Thay đổi: PCA fit train/transform đủ 60.000 ảnh; t-SNE 5.000 train, 50 ảnh/lớp.
- Thêm chọn lớp/split, nền xám, zoom/kéo/reset và preview ảnh trong HTML hiện có.
- Sửa preview cũ khi đổi phép chiếu để không hiển thị nhầm ảnh/split của thao tác trước.
- Kiểm chứng: shapes, finite coordinates, train-only sampling, cân bằng lớp và thao tác browser đạt.
- Giới hạn: RGB pooled features chưa là embedding CNN/ViT; không tạo tọa độ theo nhãn.

## 10/11/2026 — Tổ chức nhánh và tài liệu

- Nhánh cousework bỏ theo dõi Practice 2, giữ nguyên 82 file local đã đối chiếu SHA-256.
- Chia docs/define và docs/design; bổ sung model/chiến lược tuning, chưa đổi pipeline train.
- Chuyển progress/AI_usage vào docs/; thêm audit Markdown, cập nhật các liên kết và quy ước ghi chép.
- Kiểm chứng: liên kết nội bộ, UTF-8, giới hạn Markdown; không chạy lại EDA/train vì chỉ đổi tài liệu.
- Các thay đổi coursework hiện chưa commit/push; trạng thái phát hành xem progress.

## 10/11/2026 — Bàn giao chờ xác nhận phạm vi

- Đánh dấu phương án ViT/CIFAR-100 chờ xác nhận với giảng viên về hai dòng ViT/Anti Face Deepfake.
- Chỉ cập nhật tài liệu/trạng thái và phạm vi nhánh; không triển khai tiếp hoặc chạy thí nghiệm.
- Chuẩn bị commit/push cousework theo phạm vi được duyệt, không kèm lab/dataset/cache.

## 14/11/2026 — Khôi phục sau ngắt và giảm RAM smoke

- Phát hiện phiên trước dừng lúc loading ViT, kèm lỗi Windows 800705af/thiếu commit trong công cụ terminal.
- Không kết luận đó là nguyên nhân duy nhất của máy đơ; chưa có báo cáo bộ nhớ tại thời điểm ngắt.
- Nạp pretrained bằng meta + mmap/assign để tránh bản khởi tạo CPU lớn; giữ đúng weights/checksum.
- Train/val dùng chung pixel array; fingerprint tensor thay clone toàn backbone trong kiểm tra freeze.
- Checkpoint/replay dùng mmap và file tham chiếu; giải phóng optimizer/grad cũ trước restore.
- Guard giữ tối thiểu 2 GiB RAM/3 GiB commit; thử thu hồi cache trước khi từ chối, không đổi cấu hình model.
- Phân loại lỗi guard thành resources thay vì gradients; giữ nguyên raw report lan_6, recheck lan_7.
- Kiểm chứng A lan_4/B lan_8/C lan_5 PASS; D lan_7 dừng an toàn bởi guard, chưa đạt toàn checklist.
- Thêm docs/src.md, cập nhật xác nhận phạm vi và mô tả script đã có; không chạy tuning/full train/test.
- Test guard áp lực bộ nhớ bằng mock, không cố gây OOM thực; kiểm tra invalid model/strategy và loader dùng chung pixel đạt.
- Review ghi nhận guard chưa bao phủ cuối save/load/replay; đây là việc còn mở, chưa sửa trong bản xuất này.

## 16/11/2026 — Hoàn thiện source và tài liệu theo module

- Tách smoke từ 2_train.py sang 0_smoke.py; 2_train.py giữ loop pilot/final trực tiếp, không tạo Trainer.
- Bổ sung guard sau save/load/replay và cuối smoke; giữ nguyên các bằng chứng GPU cũ.
- Train có config gate, val chọn best, last/full state, resume config validation và history từ checkpoint.
- Evaluate/demo dùng chung preprocessing/model, test cần opt-in và checkpoint final.
- Tổ chức docs/src/ cùng cấp define/design, mỗi Python có Markdown 5W1H và luồng input/output riêng.
- Regression CPU kiểm tra pipeline tổng hợp, không đọc CIFAR test hoặc tải pretrained; kết quả xem progress.

## 16/11/2026 — Resource probe A

- Thêm RUN_PURPOSE để phân biệt lượt đo khả năng chạy với pilot chọn cấu hình chính thức.
- Final từ chối chọn LR từ checkpoint purpose=resource_probe; resume cũng đối chiếu purpose.
- Giữ default config chưa cho train dài; probe dùng runtime overrides, tham số thực được lưu trong run.

## 16/11/2026 — Chia sẻ checkpoint bằng Git LFS

- Cho phép track đúng best.pt/last.pt của coursework, không thay file hoặc migrate lịch sử Git.
- Giữ ignore dataset/cache/replay_reference; tạo manifest dung lượng/hash và hướng dẫn tải/sử dụng.
- Kiểm tra 7 checkpoint có đủ model/optimizer/scaler/RNG, 100 class labels và head phù hợp.
- Smoke/resource probe được đánh dấu riêng, chưa có final hoặc official test.

Khi gần 150 dòng, lưu lịch sử cũ vào docs/archive/changelog_<giai_doan>.md và giữ liên kết tại đây.
