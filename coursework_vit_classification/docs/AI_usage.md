# Ghi nhận hỗ trợ AI

## 05/11/2026

- Phạm vi hỗ trợ: tổng hợp yêu cầu, thiết kế bốn thí nghiệm, kế hoạch và cấu trúc dự án.
- Các quyết định được thống nhất: CIFAR-100, CNN/ViT với freeze và fine-tune toàn bộ.
- Hướng hiện tại: ConvNeXt-Base và ViT-B/16; ghi nhận ResNet bên dưới là lịch sử trước 18/11.
- Đã tạo file nguồn rỗng và notebook rỗng hợp lệ; chưa sinh pipeline huấn luyện.
- Kiểm chứng: cấu trúc file, UTF-8, số dòng Markdown, liên kết và notebook JSON.
- Chưa có kết quả accuracy, thời gian hoặc VRAM của coursework.

## 05/11/2026 — Dữ liệu và EDA

- Hỗ trợ triển khai config/data/EDA và HTML explorer không dùng CDN.
- Tách report thống kê khỏi RGB gzip local; giữ dataset/pixel trong data/ được ignore.
- Sửa stdout cp1252; tải chính thức/range chậm, chuyển mirror archive và xác minh MD5 chính thức.
- Đã kiểm tra syntax Python/JavaScript; số đo dữ liệu thực được ghi vào outputs/ sau EDA.
- Cập nhật hướng dẫn chạy và tài liệu workflow; không sửa pipeline các lab cũ.
- EDA thực đủ 60.000 ảnh; kiểm tra split, RGB gzip, nhãn và thống kê train đạt.
- Phát hiện 26 nhóm trùng pixel, 12 nhóm xuyên split; không tự xóa hoặc thay split.
- Edge headless: đã xem ảnh giao diện; bộ lọc split/lớp/tên, sắp xếp và phân trang đạt.

## 05/11/2026 — PCA và t-SNE

- Bổ sung phép chiếu vào 1_eda.py và HTML hiện có; không tạo thêm pipeline src.
- PCA fit train cho toàn bộ dataset; t-SNE dùng mẫu train 50 ảnh/lớp, nhãn không tham gia tính tọa độ.
- Kiểm chứng số điểm, finite coordinates, train-only sampling và class counts đạt.
- Edge kiểm tra filter, nền xám, preview pixel thật và zoom/reset; không dùng phép chiếu làm metric model.

## 10/11/2026 — Phạm vi nhánh

- Bỏ Practice 2 khỏi index nhánh cousework bằng git rm --cached, giữ nguyên bản local.
- Đồng bộ README gốc, ignore, attributes và chỉ mục tiến độ theo coursework.
- Kiểm tra file local và đường dẫn Git sau thao tác; chưa commit/push.
- Phân nhóm tài liệu thành docs/define và docs/design; cập nhật liên kết/cây thư mục, giữ nhật ký ở gốc.
- Viết lại Define/Design theo bố cục yêu cầu; thêm hai tài liệu model, đối chiếu torchvision chính thức.
- Ghi rõ ResNet torchvision v1.5, nguồn recipe ViT và giao diện nội bộ chỉ là đề xuất; chưa sinh code train.
- Soạn Training_Strategy.md: tách smoke/tuning/final/test, bảng 8 pilot + 4 final, fixed variables và chọn LR bằng val.
- Chuyển nhật ký vào docs và bổ sung Audit_Log/CHANGELOG theo vai trò riêng; chưa chạy smoke hoặc sửa code train.
- Chốt tài liệu và trạng thái chờ giảng viên xác nhận ViT/Anti Face Deepfake; không triển khai tiếp trong lúc chờ.
- Chuẩn bị xuất bản nhánh cousework theo phạm vi được duyệt, loại lab/dataset/cache, giữ file local.

## 14/11/2026 — Khôi phục task trước

- Đối chiếu log/summary và code đã lưu; nhận diện C bị ngắt, không tự đánh dấu PASS.
- Bổ sung model/loader/logger/smoke, giảm bản sao RAM và guard tài nguyên; giữ artifact cũ.
- docs/src.md mô tả 5W1H theo yêu cầu; cập nhật README/Define/Design với phạm vi đã xác nhận.
- Recheck A/B/C trên CUDA PASS; D tự dừng khi thiếu Windows commit, không hạ ngưỡng bảo vệ.
- Audit lưu kết quả thực; changelog ghi sửa code/pipeline; chưa tuning/train chính thức hoặc push.
- Review nêu giới hạn guard sau replay và bias thời gian batch; giữ raw report, cập nhật chú thích audit.
- Chuẩn bị xuất bản coursework theo phạm vi được duyệt, giữ HTML mới trên remote bằng fast-forward.

## 16/11/2026 — Source và tài liệu per-file

- Hoàn thiện code train/evaluate/demo, tách smoke, bổ sung guard cuối workflow và kiểm tra resume.
- Chuyển mô tả tổng hợp thành docs/src/README.md và từng file Markdown 5W1H theo source.
- Thêm kiểm tra hồi quy CPU bằng model/dữ liệu tổng hợp; không coi đó là lượt train/test CIFAR-100.
- Chưa chạy tuning/final, test thật hoặc commit/push giai đoạn mới.
- Bổ sung mô tả tests/test_pipeline.py trong docs/src/test_pipeline.md, đối chiếu code và report hiện có; chỉ đổi tài liệu.
- Rà soát Markdown và chuyển mục Which sang mô tả model/dataset/công cụ/cấu hình; đồng bộ quy tắc, không sửa code thực thi.
- Chạy thử A một epoch đầy đủ theo phạm vi resource_probe; không dùng để chọn LR, test hoặc kết quả cuối.
- Lan_9 hoàn tất 45k/5k, val 64,16%; đối chiếu checkpoint/backbone/BN và 2813 optimizer steps, lưu verification và telemetry.
- Ghi quy trình tuning → chọn val → final khởi tạo lại, phân vai 6 người và tiêu chí bàn giao trong Work_Allocation.md; chưa gán tên hoặc chạy thêm.
- Xuất bản theo phạm vi được duyệt: coursework và quy tắc liên quan trên cousework; chỉ text artifact, không dataset/weights.
- Yêu cầu bổ sung checkpoint: kiểm tra 7 best/last, head 100 lớp và cấu trúc resume; tạo manifest SHA-256, cấu hình LFS và hướng dẫn lấy về.

## 18/11/2026 — Đồng bộ cặp model gần quy mô

- Thay ResNet-50 bằng ConvNeXt-Base theo lựa chọn mới, giữ ViT-B16 và ma trận A–D.
- Sửa head, freeze LayerNorm, stochastic depth, smoke fingerprint và chặn pilot khác kiến trúc.
- Viết ConvNeXt_Base.md, cập nhật Define/Design/source docs và ghi rõ checkpoint ResNet là lịch sử.
- Kiểm chứng cấu trúc torchvision trên meta và hồi quy CPU tổng hợp; không tải weights hoặc chạy GPU mới.
- Chưa chốt budget hoặc train mới; đã duyệt commit/push cousework, kiểm tra lại evidence/hash và phạm vi trước xuất bản.
- Push d5e6dfc thành công lên cousework ngày 18/11, upload đủ 7 checkpoint LFS (1,3 GB); không còn blocker upload cũ.
- Thêm Training_Monitor.ipynb đọc JSON, tách train/val, so sánh run, bảng trạng thái và refresh tùy chọn; không nạp model.
- Chạy mọi code cell (refresh mặc định tắt), kiểm tra history thật, hai run tổng hợp, bộ lọc và JSON lỗi; xem ảnh biểu đồ; đã duyệt commit/push notebook lên cousework.
- Xóa lan_9 ResNet theo yêu cầu, cập nhật manifest; chạy A smoke lan_9 lỗi mạng, tải weights chính thức rồi lan_10 dừng đúng guard commit 2942,83 MiB.
- Đã duyệt A 2×5 pilot và final 20 epoch; chưa bắt đầu do smoke thiếu tài nguyên, đang chờ giải phóng bộ nhớ; không tự đóng ứng dụng hoặc hạ ngưỡng.
- Được phép đóng ứng dụng: đã đóng Zalo/Discord, Chrome bị từ chối quyền; smoke lan_11 batch16 đạt, bắt đầu A_LR1 lan_12.
- Controller local theo dõi nhiệt/disk, dừng theo lỗi và chọn LR bằng val rồi khởi tạo final mới; không test hoặc tự commit/push.
- Hoàn tất pilot lan_12/13, chọn LR2; final lan_14 bị guard dừng giữa epoch20, giữ epoch19, giải phóng bộ nhớ rồi resume riêng epoch20.
- Đủ 20 epoch, best val81,46% epoch9; kiểm chứng 6 best/last, frozen tensors, optimizer/RNG/history và reset final, cập nhật manifest và A_Results.md.
- Giữ traceback/interruptions.json, GPU trở về 0 MiB; khóa train lặp bằng config, chưa test/commit/push kết quả mới.
- Chạy lại notebook monitor trên 3 history thật (5/5/20 epoch), kiểm tra bảng 81,46% và xem ảnh biểu đồ; đồng bộ trạng thái Define/Design/README.

## 21/11/2026 — Nhập C bàn giao

- Pull da0d199 trên cousework, giữ tài liệu local; chuyển 2 smoke/2 pilot/1 final C thành lan_15–19, không ghi đè A.
- Giữ nguyên raw checkpoint/log/config, lưu mapping và nguồn/hash trong import_C_manifest.json; wrapper C dùng tên local.
- Xác nhận SHA/size mọi file, basic checkpoint epoch/head/split và 14 run cũ bất biến; bổ sung2 final C vào manifest.
- Giữ test/resume nguồn và giới hạn all-four-lock/deep-audit chưa xác nhận; không train, test mới hoặc commit/push.
- Được duyệt push kết quả A: kiểm tra lại SHA-256 toàn bộ 12 checkpoint, trạng thái completed, liên kết/Markdown/AST; chỉ xuất coursework và chỉ mục liên quan.
- Push 68641b3 thành công lên cousework, upload đủ 7 LFS mới (~2,46 GB); đồng bộ ghi nhận xuất bản, không kèm dataset/cache.

## 21/11/2026 — Giải thích lịch sử chạy

- AI đọc artifact config/summary của 14 run, log gián đoạn và history final; đối chiếu tài liệu audit/kết quả.
- Tổng hợp mục đích, cấu hình và kết quả từng run; không thực thi huấn luyện hoặc kiểm thử mới.
- Làm rõ tiêu chí PASS trong Audit_Log từ artifact và source smoke; không chạy lại thí nghiệm.
- Đối chiếu lại 14 summary/config, bổ sung kết quả cụ thể theo từng run vào Audit_Log; tách completed pilot/final khỏi PASS smoke và giữ nhãn lỗi legacy lan_6.

## 25/11/2026 — Đánh giá overfitting và bước thực nghiệm kế tiếp

- AI đối chiếu yêu cầu/protocol, mã data/model/train và artifact final A/C để tư vấn có cần chạy lại.
- Xác định best accuracy đều ở epoch9, validation suy giảm cuối run trong khi train tiếp tục tăng; C rõ hơn A.
- Đề xuất giữ baseline A/C, hoàn tất B/D trước khi quyết định thí nghiệm bổ sung; không dùng test C để tuning.
- Kiểm chứng bằng history/config/summary và quy tắc chọn best trong source; không thực thi train hoặc evaluation mới.
- Cập nhật progress và nhật ký này; không đổi source, hyperparameter hoặc tạo kết quả thí nghiệm mới.

## 27/11/2026 — Tích hợp bàn giao B/C

- AI fetch remote cousework và đối chiếu gói C với manifest; giữ C lan_15–19, chỉ nhập B còn thiếu thành lan_20–23.
- Tải8 file qua GitHub media khi LFS fetch không nhận dữ liệu, xác minh SHA-256/size theo pointer remote.
- Giữ raw artifacts, tạo mapping/provenance B và cập nhật checkpoint manifest; đối chiếu config/history/summary, header weights và split.
- Cập nhật README, Checkpoints, progress và thuộc tính LFS cho smoke replay reference B; không merge remote do force-update và thay đổi ngoài gói.
- Không thực hiện train/evaluate/test mới, không commit/push; kết quả B lấy từ artifact bàn giao.

## 27/11/2026 — Chỉ mục runs

- AI tạo outputs/runs/README.md theo yêu cầu: phân biệt smoke/pilot/final và mô tả toàn bộ23 run từ artifact.
- Đối chiếu trạng thái, LR, accuracy, best epoch và mapping; kiểm tra đường dẫn/số dòng, không thực thi model.

## 27/11/2026 — HTML trình bày ViT từ đề bài PDF

- AI trích2 trang PDF local bằng pypdf, đối chiếu10 nội dung và yêu cầu demo/sản phẩm; tham khảo paper ViT và Torchvision chính thức.
- Dùng skill Sites ở chế độ local để thiết kế HTML/CSS/JS thuần, minh họa tương tác tự dựng và biểu đồ snapshot A/B/C thật.
- Phần code/artifact còn thiếu được đánh dấu chờ bổ sung; không giả Top-5, kết quả che vùng hoặc attention checkpoint.
- Thêm presentation.html, tài liệu thiết kế và liên kết README; giữ source train/evaluate, không chạy model/test mới.
- Edge headless xác nhận desktop/mobile, tương tác, bàn phím, file/HTTP và in15 trang; xem screenshot, kiểm tra links/snapshot/Markdown đạt.
- PDF và screenshot kiểm tra ở thư mục tạm; chưa commit/push hoặc hosting. Bộ demo thực và D vẫn cần hoàn thiện sau.

## 28/11/2026 — Rà soát và chuẩn bị xuất bản đồng bộ

- Đối chiếu docs/source/notebook/artifact, sửa trạng thái lỗi thời và thêm bảng kết quả/provenance A/B/C.
- Ghi B/C là kết quả các thành viên khác; raw gói bàn giao giữ nguyên, local chỉ chuẩn hóa/kiểm tra bản sao.
- Xác minh SHA256/size22 checkpoint/reference và toàn bộ raw nhập; kiểm tra AST/final history, links/Markdown.
- Chuẩn bị commit nối tiếp remote/lịch sử local và push theo chỉ định; không force, không train/test mới hoặc thêm dataset/cache.
