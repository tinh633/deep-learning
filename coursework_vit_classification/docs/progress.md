# Tiến độ coursework
Lịch sử 16/11: code pipeline đã có; A resource_probe hoàn tất 1 epoch thật, chưa tuning/final/test chính thức. D GPU cần recheck.
Giảng viên đã xác nhận ViT classification trên CIFAR-100, không kết hợp Anti Face Deepfake.
Nhật ký thay đổi: [CHANGELOG](CHANGELOG.md); checklist smoke: [Audit_Log](Audit_Log.md).
## 05/11/2026 — Thiết kế và khởi tạo
- Dataset: CIFAR-100; bốn chiến lược CNN/ViT × freeze/fine-tune.
- Model dự kiến: ResNet-50 và ViT-B/16, pretrained ImageNet-1K.
- Đã viết phân tích yêu cầu và kế hoạch 4 tuần.
- Đã tạo bộ khung src/, docs/, visualization/, data/ và outputs/.
- Tám file Python, HTML và requirements còn rỗng; notebook hợp lệ nhưng chưa có cell.
- Không tạo best/last checkpoint, metric hoặc log giả cho lần train chưa thực hiện.
- Chưa tải dataset/weights, cài dependency, chạy smoke hoặc huấn luyện.
## Bước tiếp theo
1. Guard cuối workflow đã có và được test tổng hợp; cần đủ RAM/commit để recheck D trên GPU thật.
2. Chốt duplicate policy và training budget khi ma trận smoke đạt.
3. Chạy pilot/final/evaluate/demo theo cấu hình đã chốt; không dùng kết quả tổng hợp để chọn model.
4. Thực hiện pipeline theo [kế hoạch](design/Implementation_Plan.md).
## Kiểm tra bộ khung

- Kiểm tra UTF-8, giới hạn 150 dòng Markdown, liên kết nội bộ và notebook JSON.
- Chỉ stage thư mục coursework; giữ nguyên thay đổi của benchmark và chỉ mục chung.
- Không có kiểm thử chức năng vì mã thực thi chưa được triển khai.

## 05/11/2026 — Triển khai dữ liệu và HTML

- config.py: đường dẫn, seed 42 và validation 5.000 ảnh.
- data.py: torchvision CIFAR100, split phân tầng, kiểm tra IDs/kích thước/cân bằng.
- 1_eda.py: tải/integrity, thống kê RGB train, brightness/contrast, trùng pixel và RGB gzip local.
- explorer.html: EDA và gallery đủ dataset; lọc split/lớp/tên, phân trang, sắp xếp ảnh cực trị.
- requirements và README có hướng dẫn chạy; docs/design/Data_EDA.md mô tả workflow.
- Đã sửa lỗi stdout cp1252 trên Windows; script chủ động xuất UTF-8.
- Nguồn chính thức chậm, tải range không cải thiện; mirror Hugging Face đã tải đủ và khớp MD5 chính thức.
- EDA hoàn tất đủ 60.000 ảnh: train/validation/test 45.000/5.000/10.000, 450/50/100 ảnh mỗi lớp.
- Kiểm chứng toàn bộ RGB gzip khớp byte dataset, labels/counts/RGB mean/std đúng dữ liệu train.
- Phát hiện 26 nhóm trùng pixel chính xác, 12 nhóm xuyên split; giữ nguyên và ghi chú để rà soát.
- Python AST, JavaScript syntax, UTF-8 và giới hạn Markdown đã kiểm tra.
- Edge headless đã xem tổng quan và gallery; test lọc train/val/test, tên lớp, sắp xếp, trang 2 đều đạt.
- HTTP server local tại 127.0.0.1:8012; URL viewer theo hướng dẫn README, thay port 8000 thành 8012.
- Chưa triển khai model/train/evaluate/demo; không commit hoặc push thay đổi giai đoạn này.

## 05/11/2026 — Biểu đồ không gian ảnh

- 1_eda.py xuất projection.json: PCA đủ 60.000 ảnh, fit 45.000 train; t-SNE 5.000 train, 50 ảnh/lớp.
- Đặc trưng RGB 8 × 8; PCA 32 chiều trước t-SNE, seed 42, perplexity 30, 750 iterations.
- Lượt thực tế: PCA 2D giữ 50,45% phương sai; t-SNE KL divergence 2,281.
- HTML thêm chọn phép chiếu, nhiều lớp, split PCA, nền xám, độ đậm, zoom/kéo/reset và hover ảnh.
- Không tạo cụm theo nhãn, không dùng pretrained embedding hoặc thêm src mới.
- Kiểm tra tọa độ hữu hạn/đúng số lượng, mẫu t-SNE chỉ từ train và cân bằng 100 lớp đạt.
- Edge: đổi phép chiếu, lọc lớp/split, nền xám, preview ảnh, zoom/reset đạt; đã xem ảnh giao diện thật.

## 10/11/2026 — Tách phạm vi nhánh cousework

- Bỏ theo dõi Practice 2 khỏi Git trên nhánh cousework, giữ file/model/kết quả local.
- Ignore các lab và benchmark ngoài coursework để tránh thêm nhầm vào nhánh này.
- README gốc, gitattributes và chỉ mục chung chuyển sang coursework.
- Nhánh practice2-pretrained-model và Thái không bị sửa; không viết lại lịch sử Git.
- Thay đổi chưa commit/push; các phần triển khai EDA còn đang ở working tree.

## 10/11/2026 — Phân nhóm tài liệu

- docs/define/: yêu cầu, phạm vi và nghiệm thu trong Requirements_Analysis.md.
- docs/design/: Implementation_Plan.md và Data_EDA.md mô tả cách thực hiện.
- README, progress và AI_usage giữ ở gốc dự án để tra cứu và bàn giao.
- Cập nhật cây thư mục, liên kết từ README và giữa tài liệu; không đổi code/data.

## 10/11/2026 — Hoàn thiện Define/Design trước model/train

- Define bao gồm tên, bài toán, người dùng, input/output, tính năng, ràng buộc và tiêu chí đánh giá.
- Design bao gồm kiến trúc, module, folders, data flow, API nội bộ dự kiến, milestone và risks.
- Bổ sung ResNet50.md và ViT_B16.md: kiến trúc, tensor, weights, freeze/fine-tune và kiểm thử dự kiến.
- Phân biệt code đã có với hợp đồng tương lai; các src model/train/evaluate/demo vẫn rỗng.
- Còn mở: policy duplicate, GPU/BN, hyperparameter sau smoke; chưa triển khai thêm hoặc train.
- Kiểm tra UTF-8, số dòng (mọi Markdown ≤150), liên kết và phép tính tham số đạt; source chưa thay đổi.

## 10/11/2026 — Chiến lược train và tuning

- Thêm docs/design/Training_Strategy.md: smoke, 8 pilot LR, 4 final độc lập, test cuối.
- Tuning chỉ LR; đề xuất giữ các biến còn lại cố định và chọn theo validation trong cùng ngân sách.
- Chưa chốt giá trị LR/T/E hoặc optimizer/WD/scheduler/batch/BN; cần số đo smoke và rà soát ngân sách.
- Đồng bộ Define, Design và README; chưa triển khai train hoặc chạy thí nghiệm.

## 10/11/2026 — Tập trung nhật ký trong docs

- Chuyển progress/AI_usage vào docs, giữ lịch sử hiện có và cập nhật liên kết.
- Audit_Log.md ghi checklist/kết quả smoke; CHANGELOG.md ghi thay đổi đáng kể về code/pipeline.
- Progress dùng tóm tắt trạng thái, vấn đề còn mở và bước tiếp theo; tránh lặp chi tiết sửa code về sau.
- Chưa có smoke model, nên audit hiện chỉ có checklist NOT_RUN; chưa tạo kết quả giả.

## 10/11/2026 — Chốt bàn giao chờ xác nhận đề tài

- Giữ kế hoạch ViT/CIFAR-100 và các tài liệu hiện tại; chưa coi là phạm vi được giảng viên duyệt.
- Chưa rõ hai dòng ViT classification/Anti Face Deepfake là lựa chọn hay yêu cầu kết hợp.
- Chưa triển khai model/train/evaluate/demo, chưa smoke/tuning/train; giữ audit NOT_RUN.
- Được phép commit/push coursework và tách Practice 2 khỏi nhánh cousework, giữ bản lab local.
- Không kèm dataset/cache/pixel, benchmark hoặc thay đổi các nhánh khác; không viết lại lịch sử.
- Kiểm tra cuối: Markdown/liên kết, Python/JS syntax, số lượng EDA/projection đạt; 82 file lab và 12 checkpoint local nguyên vẹn.

## 14/11/2026 — Khôi phục task model/smoke và tài liệu source

- Phiên 10/11 đã có A/B PASS; C không có summary khi nạp weights; D chưa chạy. Giữ nguyên artifact cũ.
- Đã triển khai model/loader/logger/checkpoint và 2_train.py chỉ hỗ trợ smoke; chưa tuning/final.
- docs/src.md mô tả 5W1H cả 8 file, phân biệt evaluate/demo chưa triển khai.
- Cập nhật xác nhận phạm vi CIFAR-100 trong README/Define/Design và chỉ mục chung.
- A lan_4, B lan_8, C lan_5 PASS 10 batch + val/save/load/replay; D lan_7 dừng sau một update.
- D: Windows commit 2553,90 MiB dưới ngưỡng 3072; chưa kết luận ổn định hoặc đủ tài nguyên train.
- Audit_Log chỉ ghi checklist/bằng chứng; giải pháp giảm bản sao RAM ghi ở CHANGELOG.
- Model weights/checkpoint local không commit; chưa commit/push các thay đổi của giai đoạn này.
- Kiểm tra guard bằng mock: đủ/thiếu RAM, thiếu commit và thu hồi cache đều đúng; loader chia sẻ data, train-only và val xác định đạt.
- Python/Markdown/liên kết/evidence đạt; cuối phiên GPU 0 MiB, không còn smoke đang chạy. D vẫn cần thêm commit trống để recheck.

## 14/11/2026 — Review và chuẩn bị xuất bản

- Review: B/lan_8 cuối workflow còn 2952,80 MiB commit; PASS chưa chứng minh đủ bộ nhớ toàn workflow.
- Cần thêm guard sau checkpoint/replay; chưa sửa phần này hoặc chạy lại trong bước xuất bản.
- Smoke loss chỉ trên 20 train/4 val; timing có khởi động CUDA/check gradient, không dùng xếp hạng model.
- Cho phép commit/push phần coursework lên cousework; dataset/weights/checkpoint vẫn giữ local.
- Giữ file HTML báo cáo vừa được bổ sung trên remote, không viết lại lịch sử nhánh.

## 16/11/2026 — Code src và tài liệu từng module

- Tách 0_smoke.py; 2_train.py hỗ trợ pilot/final/resume, 3_evaluate.py và 4_demo.py đã có code.
- Thêm guard sau save/load/replay và cuối smoke; không thay kết luận GPU A/B/C/D lịch sử.
- Thay docs/src.md bằng docs/src/README.md và 9 Markdown tương ứng các file Python, cùng cấp define/design.
- Mặc định chặn train dài/test: protocol, duplicate policy, batch/LR/WD/epochs chưa chốt; không tự chọn giá trị final.
- Regression CPU tổng hợp: 2 bài integration PASS, bao phủ smoke, guard cuối replay, ngắt/resume, final, evaluate và demo.
- Bằng chứng: [source_pipeline_check.json](../outputs/validation/source_pipeline_check.json); không phải kết quả CIFAR-100 thật.
- Chưa recheck D GPU, chưa tuning/full train/test thật hoặc tạo nội dung notebook báo cáo.
- Chưa commit/push; tiếp theo vẫn cần đủ RAM/commit, chốt duplicate policy và training budget.
- Bổ sung kiểm tra state sau resume khớp hoàn toàn nhánh train liên tục trên dữ liệu tổng hợp; cờ train/test mặc định chặn đúng.
- Thêm docs/src/test_pipeline.md: 5W1H, hai nhóm regression, mock/cleanup, cách chạy/đọc kết quả và giới hạn; không chạy lại test vì chỉ đổi tài liệu.
- Rà soát Markdown workspace: 10 mô tả source dùng What/Why/Which/When/Where/How, Which nêu lựa chọn kỹ thuật cụ thể.
- Đồng bộ AGENTS và quy tắc documentation; không thay logic Python hoặc kết quả thực nghiệm trong lượt sửa tài liệu này.

## 16/11/2026 — Thử chạy chiến lược A

- Lan_9 hoàn tất: ResNet-50 freeze, purpose=resource_probe, 1 epoch trên 45k train/5k val.
- Batch 16, AMP CUDA, workers 0, LR 1e-3, WD=0; giữ split cũ chỉ cho thử tài nguyên, không mở test.
- Không coi đây là pilot chọn LR/final; không chốt duplicate policy hoặc budget chính thức từ lượt thử này.
- Train loss/acc 1,5263/58,73%; val loss/acc 1,3034/64,16%; train+val 173,57 giây, không AMP skip.
- Best/last epoch 1 khớp, backbone/BN giữ nguyên; optimizer head đủ 2813 bước: [verification](../outputs/runs/lan_9/verification.json).
- Nhiều mẫu GPU dùng 342–494 MiB (không phải peak allocator); commit cuối 4672 MiB, guard không dừng.
- Runtime overrides được lưu; default vẫn chặn train dài. Đã ghi [quy trình/phân vai/bàn giao](design/Work_Allocation.md), chưa gán tên hoặc chốt tham số chính thức.
- 16/11: xuất code/tài liệu/lan_9; yêu cầu bổ sung 7 best/last qua Git LFS, có [hướng dẫn](Checkpoints.md); data/cache/replay_reference vẫn local.
- Checkpoint commit 8da0358 local: 7 LFS files verified; push blocked at storage upload (0 bytes), remote remains 28b5b8e. Retry `git push origin cousework`; no checkpoint/data removed.

## 18/11/2026 - ConvNeXt-Base / ViT-B16

- A/B chuyển ConvNeXt-Base (87.668.964 tham số); C/D ViT-B16 (85.875.556), chênh 2,09%, chưa bằng tuyệt đối.
- Source/docs/tests đồng bộ; 3 kiểm thử PASS, AST/liên kết/Markdown/hash đạt: [bằng chứng](../outputs/validation/convnext_pipeline_check.json).
- ResNet cũ là lịch sử; GPU smoke ConvNeXt NOT_RUN, chưa pilot/final/budget mới. A local, ba người nhận B/C/D; đã duyệt commit/push, cần xác nhận remote.
