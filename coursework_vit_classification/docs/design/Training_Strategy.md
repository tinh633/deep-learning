# Chiến lược huấn luyện và tuning hyperparameter

Cập nhật: 14/11/2026. Đã bắt đầu smoke; pilot/final vẫn là kế hoạch chưa thực thi.
Phạm vi ViT/CIFAR-100 đã được giảng viên xác nhận.
Liên quan: [Define](../define/Requirements_Analysis.md), [Design](Implementation_Plan.md),
[ResNet-50](ResNet50.md), [ViT-B/16](ViT_B16.md).

## 1. Mục tiêu và phạm vi

Giữ bốn chiến lược CNN/ViT × freeze/fine-tune, thêm tuning LR nhỏ để chọn cấu hình có căn cứ.
Không tìm hyperparameter tối ưu toàn cục và không tăng số kiến trúc đang so sánh.
Mục tiêu là trả lời ảnh hưởng của model, mức cập nhật backbone và chi phí tính toán.

```text
Chốt dữ liệu/protocol → smoke A–D → khóa không gian LR
→ 8 pilot tuning ngắn → chọn LR bằng validation
→ 4 run chính thức khởi tạo mới → khóa cấu hình/checkpoint → test và phân tích
```

## 2. Bốn chiến lược nền

| Mã | Model | Backbone | Head mới |
|---|---|---|---|
| A | ResNet-50 | Freeze toàn bộ | Linear(2048, 100), train |
| B | ResNet-50 | Fine-tune toàn bộ | Linear(2048, 100), train |
| C | ViT-B/16 | Freeze toàn bộ | Linear(768, 100), train |
| D | ViT-B/16 | Fine-tune toàn bộ | Linear(768, 100), train |

Mỗi lượt bắt đầu độc lập từ đúng ImageNet-1K weights và head mới, seed cố định.
Không chạy B từ A, D từ C, hoặc final từ checkpoint pilot thắng cuộc.
Resume chỉ tiếp tục một lượt bị ngắt với cùng cấu hình, không được coi là một lượt tuning mới.

## 3. Giai đoạn 1 — Smoke run

- Kiểm tra shape B×100, loss hữu hạn, gradient, freeze/mode và optimizer parameter groups.
- Thử vài batch có backward cho cả A–D; test save/load và resume bằng dữ liệu train/val.
- Đo VRAM, thời gian/batch và tốc độ val để ước tính ngân sách.
- Kiểm tra batch nhỏ/AMP và chính sách BatchNorm của ResNet trước khi khóa cấu hình.
- Không dùng accuracy trên vài batch để chọn model/LR; không truy cập test trong bước này.
- Số batch smoke đủ kiểm tra kỹ thuật, không tính là epoch hoặc run chính thức.
- Ghi checklist/kết luận vào [Audit_Log.md](../Audit_Log.md); sửa pipeline vào [CHANGELOG.md](../CHANGELOG.md).
- Mỗi kết luận ổn định phải kèm số batch và điều kiện đo; chưa chạy ghi NOT_RUN, không suy từ không lỗi.

Nếu không chạy được ViT fine-tune, thống nhất GPU/phương án trước khi tiếp tục.
Chốt chính sách 12 nhóm duplicate xuyên split trước tuning; không đổi dữ liệu giữa các lượt.

## 4. Giai đoạn 2 — Tuning learning rate

Chỉ tuning **LR ban đầu**. Mỗi chiến lược thử hai ứng viên, cùng ngân sách T epoch ngắn.
LR_F1/LR_F2 là hai mức cho freeze; LR_T1/LR_T2 là hai mức thấp hơn dự kiến cho fine-tune.
Chưa gán giá trị số hoặc T trước khi smoke có số đo; khóa toàn bộ ứng viên trước pilot đầu tiên.
Đề xuất dùng cùng cặp LR cho A/C và cùng cặp LR cho B/D để thuận tiện đối chiếu.

| Pilot | Chiến lược | LR thử | Dữ liệu | Ngân sách |
|---|---|---|---|---|
| A_LR1 | ResNet freeze | LR_F1 | Cùng train/val | T epoch |
| A_LR2 | ResNet freeze | LR_F2 | Cùng train/val | T epoch |
| B_LR1 | ResNet fine-tune | LR_T1 | Cùng train/val | T epoch |
| B_LR2 | ResNet fine-tune | LR_T2 | Cùng train/val | T epoch |
| C_LR1 | ViT freeze | LR_F1 | Cùng train/val | T epoch |
| C_LR2 | ViT freeze | LR_F2 | Cùng train/val | T epoch |
| D_LR1 | ViT fine-tune | LR_T1 | Cùng train/val | T epoch |
| D_LR2 | ViT fine-tune | LR_T2 | Cùng train/val | T epoch |

Chọn LR riêng trong từng chiến lược theo best validation accuracy trong T epoch.
Nếu hòa, ưu tiên validation loss thấp hơn; nếu vẫn hòa, chọn ứng viên thứ nhất đã định trước.
Đánh giá sau mỗi epoch với cùng preprocessing xác định; không đánh giá test cho pilot.
Không yêu cầu LR thắng của CNN và ViT bằng nhau; cơ hội tuning mới cần tương đương.
Lượt ngắn có thể ưu ái LR hội tụ sớm, nên kết luận là tốt nhất trong ngân sách pilot, không tuyệt đối.
Không tự thêm ứng viên hoặc kéo dài pilot sau khi xem test; thay đổi ngân sách phải được ghi rõ.

## 5. Các tham số giữ cố định trong tuning

| Thành phần | Quy định |
|---|---|
| Dataset/split | Cùng phiên bản sau xử lý duplicate, IDs và split hash |
| Seed | Seed 42 cho từng lượt; cùng seed head/sampler và cấu hình augmentation |
| Input/head | 224×224, normalization tương ứng weights, một Linear 100 lớp |
| Augmentation/loss | Cùng train transforms; CrossEntropyLoss trên logits |
| Optimizer | Đề xuất AdamW, xác nhận trước pilot; không tuning optimizer |
| Weight decay | Một giá trị đã khóa cho cả ma trận, không tìm kiếm WD trong vòng này |
| Scheduler | Ưu tiên LR cố định cho pilot/final để cô lập ảnh hưởng LR ban đầu; chưa khóa |
| Batch/precision | Khóa sau smoke, ưu tiên cùng effective batch; ghi rõ khác biệt nếu cần |
| Epoch/validation | T epoch mỗi pilot; không early-stop riêng làm lệch ngân sách |
| BN/dropout | Cùng policy giữa hai LR của một chiến lược; freeze backbone giữ eval |

Nếu batch vật lý khác nhau, ghi accumulation; accumulation không tái tạo BatchNorm của batch lớn.
Mọi giá trị cố định cần xuất trong config, không để chỉ tồn tại trong tài liệu.
Không đồng thời thay head, augmentation hoặc số layer được mở trong các pilot LR.

## 6. Giai đoạn 3 — Train chính thức

| Run logic | Model/chế độ | Cấu hình LR | Ngân sách |
|---|---|---|---|
| A_final | ResNet freeze | LR tốt nhất của A_LR1/A_LR2 | E epoch |
| B_final | ResNet fine-tune | LR tốt nhất của B_LR1/B_LR2 | E epoch |
| C_final | ViT freeze | LR tốt nhất của C_LR1/C_LR2 | E epoch |
| D_final | ViT fine-tune | LR tốt nhất của D_LR1/D_LR2 | E epoch |

E lớn hơn T và được thống nhất trước khi bắt đầu final; cả bốn cùng số epoch dự kiến.
Final khởi tạo lại pretrained/head/optimizer/RNG; không warm-start từ pilot.
Giữ các lựa chọn cố định đã công bố; thay đổi khác LR phải ghi thành thay đổi protocol.
Best checkpoint theo max val accuracy, hòa thì min val loss, sau đó epoch sớm hơn.
Last lưu epoch hoàn tất cùng trạng thái optimizer/RNG/scheduler/scaler nếu có để resume.
Một dòng log/epoch; có config/máy/model/precision, history, best.pt và last.pt.

## 7. Giai đoạn 4 — Test cuối và báo cáo

- Chỉ test best checkpoint của A_final/B_final/C_final/D_final sau khi khóa bốn cấu hình.
- Không test cả hai LR pilot hoặc dùng test để chọn lại LR/model/số epoch.
- Lưu ID, nhãn, vector 100 xác suất và checkpoint nguồn để đối chiếu lỗi.
- Báo cáo accuracy/macro-F1, theo lớp và thời gian/VRAM theo Define/Design.
- Đối chiếu freeze giữa model, fine-tune giữa model và lợi ích fine-tune trong từng model.
- Giữ pilot thành bảng riêng; không trộn accuracy pilot và final vào bảng kết quả chính.

## 8. Ngân sách, tracking và những gì chưa chốt

Tổng dự kiến: **8 pilot ngắn + 4 final đầy đủ**, ngoài các kiểm tra smoke kỹ thuật.
Chi phí train gần đúng là 8T + 4E epoch, nhưng mỗi model/chế độ có thời gian và VRAM khác nhau.
Một seed/cấu hình chưa đủ đánh giá độ ổn định; không thêm nhiều seed trong phạm vi ban đầu.
Đây là mở rộng kế hoạch so với chỉ 4 final; cần kiểm tra đủ thời gian sau smoke, không tự giảm ngân sách.

Tên thư mục vẫn outputs/runs/lan_N, không cố định A_LR1 = lan_1.
Config ghi phase=smoke/tuning/final, strategy=A–D, candidate, LR, T/E, seed và nguồn chọn cấu hình.
Mỗi lượt có thư mục riêng; bảng pilot ghi LR, best epoch, val accuracy/loss, thời gian và run nguồn.
Không tạo run hoặc số liệu rỗng trước khi chạy; không ghi đè kết quả cũ.

Còn chốt: GPU, duplicate policy, hai cặp LR, T/E, batch/AMP/BN, AdamW/WD và scheduler.
2_train.py mới có smoke; chưa có tuning/final. Bằng chứng smoke xem Audit_Log, không dùng số smoke làm kết quả cuối.
