# Quy trình và phân chia công việc thực nghiệm

Cập nhật28/11/2026: A local, B/C do các thành viên khác train và đã nhập; D chưa có gói tích hợp.
Vai trò dưới đây giữ thiết kế phân công ban đầu, chưa ghi tên cá nhân.
Ma trận và protocol: [Training_Strategy.md](Training_Strategy.md).
Hiện trạng thực: [progress.md](../progress.md).

## 1. Quy trình áp dụng cho từng chiến lược

```text
Smoke/resource check → 2 pilot LR → chọn LR bằng validation
→ khởi tạo mới từ pretrained → train chính thức đủ E epoch
→ best checkpoint theo validation → test cuối sau khi khóa cấu hình
```

- Smoke kiểm tra code/tài nguyên, không chọn LR theo accuracy vài batch.
- Hai pilot của mỗi chiến lược dùng cùng T epoch ngắn và chỉ thay LR trong cặp ứng viên đã chốt.
- Chọn LR bằng best val accuracy; hòa thì val loss thấp hơn, vẫn hòa chọn ứng viên thứ nhất.
- Run chính thức khởi tạo lại pretrained, head, optimizer và RNG; không nối tiếp checkpoint tuning.
- Resume chỉ dùng khi chính lượt đang chạy bị ngắt, với cùng cấu hình/budget.
- Test không tham gia tuning hoặc chọn checkpoint; giữ nguyên test cho đánh giá cuối.

| Chiến lược | Hai lượt tuning | Lượt chính thức | Đối chiếu |
|---|---|---|---|
| A — ConvNeXt freeze | A_LR1, A_LR2 | A_final với LR được chọn | A↔B, A↔C |
| B — ConvNeXt fine-tune | B_LR1, B_LR2 | B_final với LR được chọn | B↔A, B↔D |
| C — ViT freeze | C_LR1, C_LR2 | C_final với LR được chọn | C↔D, C↔A |
| D — ViT fine-tune | D_LR1, D_LR2 | D_final với LR được chọn | D↔C, D↔B |

Tổng kế hoạch: 8 pilot ngắn và 4 final; lượt smoke/resource probe được ghi riêng.
T và E, hai cặp LR, WD, batch/precision và policy duplicate phải chốt trước khi giao ca chạy.
A đã duyệt 18/11: pilot 5 epoch/LR, final 20 epoch; chi tiết protocol A ở Training_Strategy mục 9.

## 2. Đề xuất phân công nhóm 6 người

| Vai trò dự kiến | Phần phụ trách | Sản phẩm bàn giao |
|---|---|---|
| TV1 — Tích hợp chính | Khóa protocol, data/split, review code/audit, quản lý nguồn kết quả | Config chung, split hash, bảng tiến độ và kiểm tra artifact |
| TV2 — Chiến lược A | Smoke phù hợp, 2 pilot LR, chọn LR, train A_final | Hai kết quả pilot, lý do chọn LR, run A_final |
| TV3 — Chiến lược B | Smoke phù hợp, 2 pilot LR, chọn LR, train B_final | Hai kết quả pilot, lý do chọn LR, run B_final |
| TV4 — Chiến lược C | Smoke phù hợp, 2 pilot LR, chọn LR, train C_final | Hai kết quả pilot, lý do chọn LR, run C_final |
| TV5 — Chiến lược D | Xác nhận đủ bộ nhớ trước, sau đó 2 pilot và D_final | Bằng chứng smoke đạt hoặc blocker, pilot/final khi đủ điều kiện |
| TV6 — Tổng hợp đánh giá | Đánh giá checkpoint đã khóa, so sánh/lỗi, notebook/demo | Report theo lớp, bảng đối chiếu, trình bày và demo |

TV1 rà soát và hỗ trợ các phần, không chỉ nhận kết quả cuối; các chủ chiến lược giải thích được cấu hình của mình.
Đây là phân công trách nhiệm, không mặc định cả sáu người đều có GPU hoặc chạy đồng thời.
Nếu dùng chung một GPU, xếp ca tuần tự; ưu tiên bảo vệ RAM/commit thay vì mở nhiều run.
Không tự đổi model, head, split, augmentation hoặc ngân sách vì máy từng thành viên khác nhau.
Nếu phải khác batch/precision/máy, thống nhất trước và ghi rõ ảnh hưởng đến so sánh tài nguyên.

## 3. Gói bàn giao của mỗi chiến lược

1. Hai pilot: config, run.log, history, summary; bảng LR/best epoch/val accuracy/val loss/thời gian.
2. Quyết định chọn LR: hai run nguồn và tiêu chí val, không viện dẫn test accuracy.
3. Final: config/source version, model.txt, history, summary, best.pt và last.pt.
4. Dữ liệu: class order, split hash, seed, preprocessing, phiên bản thư viện và cấu hình máy.
5. Trạng thái: completed/failed/interrupted và kiểm tra còn thiếu; không đổi FAIL thành PASS khi bàn giao.

Chỉ chạy code hiện có sau khi khai báo cấu hình rõ ràng; không cần viết lại train riêng cho mỗi người.
Mã A_LR1/A_final là nhãn logic; thư mục thực vẫn lan_N theo config run.
Không ghi đè các lan_N khi gộp từ máy khác; giữ bảng ánh xạ run nguồn và vị trí sau khi tích hợp.
Các tham chiếu SELECTED_FROM trong checkpoint phải truy được về pilot nguồn qua bảng ánh xạ.
Dataset/cache không commit; best/last chia sẻ bằng Git LFS, xem [Checkpoints.md](../Checkpoints.md); file đối chiếu vẫn local.

## 4. Điều kiện nhận việc và nghiệm thu

- Trước pilot: smoke đủ, tài nguyên đáp ứng, cùng source/config/split và ứng viên LR đã khóa.
- Trước final: đủ hai pilot hoàn tất, chọn LR theo val và xác nhận E chung.
- Trước test: khóa bốn cấu hình/checkpoint; người chạy đánh giá dùng cùng protocol và không tuning lại.
- Khi lỗi: audit ghi kết quả kiểm tra, changelog ghi thay đổi pipeline, progress ghi blocker/bước tiếp.
- Báo cáo phân biệt pilot/final và resource probe; không trộn số đo vào một bảng xếp hạng.

A chạy local; B/C thành viên khác đã bàn giao2 pilot +1 final, D còn chờ.
Gói nguồn B_ban_giao/C_ban_giao giữ nguyên; bản chuẩn hóa nằm outputs/runs, không nhận là train local.
Lan_9 là ResNet resource_probe lịch sử; không đại diện A ConvNeXt hoặc thay pilot/final mới.
D/GPU vẫn cần kiểm chứng lại sau khi đủ RAM/commit; chưa giao D như một phần đã sẵn sàng train dài.
