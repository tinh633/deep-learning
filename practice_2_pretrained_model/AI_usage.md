# Ghi nhận sử dụng AI — Practice 2

## 2026-10-26 — Thiết kế lab và cập nhật tài liệu

- Người dùng chọn CIFAR-10, chấp nhận VGG16/ResNet18/DenseNet121 và kế hoạch thí nghiệm.
- Người dùng yêu cầu cấu hình đồng bộ, số lượt cố định, EDA quan trọng và log đầy đủ.
- Người dùng cung cấp ảnh code giảng viên để tham khảo phong cách; xác nhận chi tiết triển khai
  phải phù hợp bài thực tế, không sao chép nguyên cấu hình ví dụ.
- Codex soạn Experiment_Plan.md, bổ sung lý do chọn từng model và giới hạn so sánh.
- Qua thảo luận, thống nhất luồng `.py`, 7 file dự kiến, train/evaluate trực tiếp và logger riêng.
- Codex viết README.md với EDA, logging và trình tự review từng phần.
- Công cụ: đọc/chỉnh sửa file, Git status/diff và kiểm tra tài liệu bằng Python.
- Kiểm tra: giới hạn 150 dòng/file, liên kết Markdown nội bộ và whitespace diff.
- Chưa viết source, tải dữ liệu/weights, chạy EDA/train/test hoặc đo chất lượng model.
- Thay đổi tài liệu mới chưa commit/push; không coi kết quả kiểm tra tài liệu là kiểm thử model.

## 2026-10-26 — Tạo cấu trúc thư mục

- Theo yêu cầu người dùng, tạo src/, data/, outputs/ và visualization/ trong Practice 2.
- Cập nhật cây thư mục trong README; không tạo source rỗng hoặc file giữ chỗ.
- Kiểm tra thư mục tồn tại và giới hạn dòng tài liệu; chưa triển khai/chạy source.

## 2026-10-26 — Tạo Python files và triển khai dữ liệu

- Người dùng yêu cầu tạo các file Python và triển khai tải dữ liệu trong data.py.
- Tạo 7 file; config/data có code, 5 file còn lại chỉ có docstring trạng thái theo yêu cầu.
- Dùng torchvision CIFAR10, split phân tầng lưu JSON, transforms và DataLoader dùng chung.
- Kiểm tra parse 7 file, split lặp lại/không giao nhau/phân bố và từ chối cấu hình khác: qua.
- Mạng sandbox chặn tải; đã chạy lại download với quyền mạng được duyệt.
- Download nguồn chính thức đang chạy chậm; chưa xác nhận dữ liệu thật hoàn tất/toàn vẹn.
- Chưa train, đánh giá model, commit hoặc push.

## 2026-10-26 — EDA và HTML viewer

- Người dùng yêu cầu visual EDA trên HTML. Triển khai src/1_eda.py và visualization/eda.html.
- EDA: split/class counts, pixel range, train RGB stats/histogram, samples có ID,
  preprocessing và nhóm trùng pixel SHA-256 trong/qua split; không tự xóa dữ liệu.
- Viewer: lọc lớp/split, biểu đồ SVG có trục, file picker khi mở trực tiếp hoặc fetch qua HTTP.
- Synthetic full-size split check qua: totals, samples, histogram, finite stats và duplicate groups.
- Node syntax và Edge headless DOM qua cho missing-report/render giả; các fixture tạm đã loại bỏ.
- Edge trong sandbox lỗi GPU/access; chạy kiểm tra ngoài sandbox được duyệt và thành công.
- Chưa có báo cáo dữ liệu thật: tải CIFAR-10 vẫn chưa hoàn tất. Chưa test Live Server IDE.

## Khôi phục download bị dừng

- Kiểm tra tiến trình gốc: đã thoát với lỗi DNS getaddrinfo failed ở 119.767.040 byte.
- Nguồn chính thức đã phản hồi lại; curl resume được chấp nhận và file tiếp tục tăng.
- Tải tiếp bằng retry, không xóa/tải lại từ đầu; sau tải đủ sẽ kiểm tra/giải nén qua torchvision.
- Phiên tải mới 89494; chưa xác nhận hoàn tất hoặc checksum hợp lệ.
- Cập nhật sau đó: phiên 89494 hoàn tất exit 0; torchvision xác minh archive, giải nén và
  load thành công 50.000 train/10.000 test. Chưa chạy EDA thật trong lần kiểm tra trạng thái này.

## Tiếp tục flow — EDA dữ liệu thật

- Chạy 1_eda.py thành công: tạo outputs/cifar10_split.json và outputs/eda.json.
- Split 45.000/5.000/10.000, cân bằng lớp, không giao ID; 0 nhóm trùng pixel chính xác.
- Đối chiếu đủ 360 thumbnail/nhãn với ảnh gốc, histogram totals và RGB stats tính độc lập: qua.
- Cả ba DataLoader cho batch 64×3×224×224 hữu hạn, nhãn đúng phạm vi: qua.
- Edge headless qua HTTP localhost nạp báo cáo thật, render bảng/gallery/histogram: qua.
- Không chạy model/train/test metrics; chờ review EDA trước phần model. Chưa commit/push.

## Gallery toàn bộ dữ liệu và thiết kế HTML

- Theo yêu cầu người dùng, thiết kế lại trang với sidebar, summary cards và gallery phân trang.
- EDA xuất đủ 60.000 ảnh gốc thành 236 atlas PNG lossless; JSON lưu ID/nhãn theo split.
- Viewer có tất cả split/lớp, ID/khoảng ID, sắp xếp, 50/100/200 ảnh/trang và preview pixel/mượt.
- Đối chiếu toàn bộ pixel/nhãn atlas với dataset gốc; không sửa ảnh/dữ liệu/split.
- Edge qua localhost: filter, ID/range, invalid query, trang cuối, preview và atlas HTTP qua.
- Đã xem screenshot desktop/mobile và điều chỉnh CSS responsive. Không train/commit/push.

## Chỉnh giao diện theo Practice 1

- Đọc explorer.html của Practice 1 theo yêu cầu; dùng cùng bảng màu teal, header,
  menu ngang, chiều rộng nội dung, thẻ/bảng và khung gallery cuộn; bỏ sidebar trước đó.
- Giữ duyệt đủ 60.000 ảnh, filter, ID/range, phân trang và preview.
- Edge desktop/mobile: tương tác và không tràn ngang qua; đã xem screenshot thật.
- Sau sắp xếp gallery theo ID, đối chiếu lại đầy đủ 60.000 ảnh/nhãn atlas: qua.

## Tiếp tục flow — triển khai model

- Người dùng yêu cầu tiếp tục sau review HTML/EDA; triển khai model.py và config A–F.
- PretrainedCNN khai báo trực tiếp ba torchvision models; head Linear hoặc MLP,
  freeze backbone trước khi mở nhóm cuối; override train để giữ BatchNorm statistics.
- Kiểm tra đủ 12 model/head/freeze combinations trên input giả 1×3×224×224: logits,
  gradient hữu hạn đúng nhóm, optimizer update, frozen weights/BN không đổi và eval mode qua.
- Mỗi kiểm tra dùng một SGD step chỉ xác minh cơ chế; không phải lượt train A–F dùng Adam.
- Dùng weights=None trong kiểm tra để không tải weights; mặc định API dùng pretrained=True.
- Chưa xác minh ImageNet weights hoặc train CIFAR-10. Chờ review model trước logger/train.
- Cập nhật README/kế hoạch/progress; chưa commit/push.
- Theo yêu cầu bổ sung: kế hoạch giải thích H1/H2, số tham số, lý do 256/Dropout 0,2
  là lựa chọn ban đầu và giới hạn A/D khi thay đồng thời capacity, ReLU và Dropout.
- Theo yêu cầu mới, chuyển giải thích H1/H2 sang Model_QA.md để tích lũy hỏi–đáp về
  model/thông số trong quá trình làm; kế hoạch và README liên kết đến tài liệu này.

## Tiếp tục flow — logging và train

- Triển khai logger.py và 2_train.py trực tiếp; thêm PRETRAINED và requirements.txt.
- Full environment/model/config log, TensorBoard mỗi epoch, validation và best checkpoint.
- Cài TensorBoard 2.21.0 trong .venv sau khi mạng sandbox chặn pip; tải ngoài sandbox được duyệt.
- A/E smoke: weights ngẫu nhiên, 2 epoch trên 8/4 CIFAR thật; reload metrics và event scalars qua.
- Kiểm tra lỗi/interrupt lúc setup qua; không Finished giả. Artifact chỉ trong thư mục tạm đã xóa.
- Môi trường test .venv CPU, khác Python hệ thống; chưa kiểm tra CUDA/pretrained hoặc chất lượng model.
- Bổ sung Training.md để README vẫn dưới 150 dòng. Chưa chạy 18 lượt hoặc commit/push.
- H2 train và log loss options qua; ngắt giữa epoch 2 giữ history/checkpoint epoch 1 đúng.

## 2026-10-27 — Triển khai đánh giá

- Theo yêu cầu tiếp tục flow, triển khai 3_evaluate.py và config EVAL_RUN/RUN_TEST=False.
- Dùng config checkpoint, strict state load, split SHA-256; train eval không augmentation.
- Lưu report từng split: metrics, confusion, per-class/TP/TN/FP/FN, mọi prediction/ID/confidence;
  chọn tối đa 24 ảnh sai confidence cao và xuất PNG riêng, không tạo thêm HTML.
- Logger chia sẻ thu thập môi trường và hỗ trợ evaluate.log; train vẫn dùng train.log mặc định.
- Smoke train/evaluate CIFAR 8/4 thật, random weights: loss/accuracy khớp checkpoint, sklearn
  metrics/zero-support/IDs/error PNG qua; test tắt không bị iterate, test bật chỉ dùng 5 ảnh giả.
- Split hash sai bị từ chối, giữ report cũ; artifact tạm đã xóa, không có kết quả test chính thức.
- Cập nhật tài liệu; chưa tải pretrained, train chính thức, xác minh CUDA hoặc commit/push.

## 2026-10-27 — Chuyển GPU trong `.venv`

- Theo yêu cầu người dùng, dừng tạo môi trường GPU riêng; giữ dùng `.venv`.
- Thay torch CPU bằng torch 2.11.0+cu126 / torchvision 0.26.0+cu126 từ nguồn PyTorch.
- Dừng đúng hai PID đã xác minh là lượt CPU đang chạy để giải phóng DLL; giữ artifact,
  ghi trạng thái dừng chuyển CUDA vào train.log, không coi lượt đó hoàn tất.
- User đặt worker 2; thêm main guard train/evaluate cho Windows, giữ loops trực tiếp.
- Lần smoke worker đầu gặp OpenBLAS thiếu RAM; giới hạn BLAS/OMP/MKL threads và kiểm tra lại.
- `.venv` CUDA True; pretrained ResNet18 A batch 64, worker 2 trên CIFAR thật 128/64 qua
  train/validation/checkpoint/TensorBoard, peak CUDA allocated 488,8 MiB. Artifact tạm đã xóa.
- Pip check qua; ghim requirements CUDA, cập nhật hướng dẫn/rules/progress. Không train GPU chính thức.
- `.venv-gpu` tạo dở trước yêu cầu dừng vẫn được giữ, không sử dụng; không tự xóa môi trường.
- Không commit/push; chưa kiểm tra GPU cho mọi tổ hợp model/head/fine-tune.

## Bàn giao phiên — 2026-10-27

- Người dùng yêu cầu dừng và commit/push trạng thái hiện tại lên origin nhánh Thái.
- Kiểm tra không còn train/evaluate đang chạy; lan_1 CPU chưa hoàn tất, chưa có kết quả GPU chính thức.
- Ghi tiến độ bàn giao, kiểm tra cú pháp/tài liệu và phạm vi Git; kết quả upload báo sau thao tác.

## Bỏ atlas, dùng pixel/canvas như Practice 1

- User xác nhận bỏ eda_images, không xuất 60.000 PNG; yêu cầu kiểm tra và push lại Thái.
- EDA xuất 6 RGB gzip HWC uint8; viewer giải nén/vẽ canvas, hỗ trợ HTTP hoặc chọn file cục bộ.
- Kiểm tra đủ 60.000 pixel/nhãn và hashes qua; Edge kiểm tra canvas/chunk boundaries,
  filter/phân trang/preview, local-file không mạng và mobile width 390 qua.
- Xác minh đúng đường dẫn rồi xóa riêng 236 atlas PNG; giữ archive/batches CIFAR gốc.
- Git chỉ mở ngoại lệ cho data/CIFAR10_web/*.rgb.gz, không upload archive/batches/env.
