# Kết quả và provenance A/B/C

Cập nhật28/11/2026. Đây là **validation**, chưa phải bộ đánh giá test cuối chung.
[Chỉ mục23 run](../outputs/runs/README.md) phân biệt14 smoke,6 pilot và3 final.

## Trách nhiệm và nguồn kết quả

- A: huấn luyện trên máy local của người tích hợp, kết quả chi tiết ở [A_Results](A_Results.md).
- **B và C do các thành viên khác huấn luyện và bàn giao**; local chỉ nhận, kiểm tra và chuẩn hóa bản sao tên run.
- Gói gốc `B_ban_giao/` và `C_ban_giao/` được giữ nguyên nội dung trên nhánh cousework.
- B lấy từ remote commit b754c2f, bản chuẩn hóa local lan_20–23; [manifest B](../outputs/import_B_manifest.json).
- C lấy từ da0d199, đã đối chiếu blob cùng nội dung trên remote b754c2f, local lan_15–19;
  [manifest C](../outputs/import_C_manifest.json).
- D chưa có gói final tích hợp; không tạo điểm0 hoặc suy ra chất lượng D từ smoke thất bại.

## Final đã hoàn tất

| Exp | Model/chế độ | Run | Epoch | LR | Best epoch | Best val | Val loss tại best | Last val |
|---|---|---|---|---|---|---|---|---|
| A | ConvNeXt-Base freeze | lan_14 | 20/20 | 3e-4 | 9 | 81,46% | 0,6138 | 80,94% |
| B | ConvNeXt-Base fine-tune toàn bộ | lan_23 | 20/20 | 3e-5 | 13 | 89,70% | 0,4759 | 88,98% |
| C | ViT-B/16 freeze | lan_19 | 20/20 | 3e-4 | 9 | 82,08% | 0,6238 | 81,30% |

Best checkpoint chọn theo max validation accuracy, hòa thì min loss, hòa tiếp giữ epoch sớm hơn.
Val loss tại best accuracy không nhất thiết là val loss nhỏ nhất trong toàn run.
Final khởi tạo mới từ pretrained/head mới, không warm-start từ weights pilot.
A resume riêng epoch20 sau guard; C resume Colab→Kaggle với phiên bản thư viện khác, giữ metadata nguồn.

## Pilot chọn LR

| Exp | LR1/run | Best val LR1 | LR2/run | Best val LR2 | LR chọn |
|---|---|---|---|---|---|
| A | 1e-3 / lan_12 | 80,64% | 3e-4 / lan_13 | 80,78% | LR2 |
| B | 1e-4 / lan_21 | 87,84% | 3e-5 / lan_22 | 89,42% | LR2 |
| C | 1e-3 / lan_17 | 80,56% | 3e-4 / lan_18 | 81,84% | LR2 |

Mỗi pilot5 epoch, chọn bằng validation theo protocol; không trộn pilot với bảng final.
A/B/C cùng split45k/5k/10k, seed42, batch16, AdamW/WD0, AMP, workers0, không scheduler.
Giữ12 nhóm duplicate xuyên split và công bố; không sửa dữ liệu giữa các run.
GPU/môi trường khác nhau: A/B RTX3050 Laptop, C Tesla T4 ở đầu run và có metadata resume riêng.
Không dùng thời gian ở máy khác nhau làm benchmark trực tiếp CNN/ViT.

## Mức kiểm chứng và giới hạn

- A đã có audit sâu frozen tensors, optimizer/scaler/RNG/history: [experiment_A_check](../outputs/validation/experiment_A_check.json).
- B/C đã kiểm tra SHA-256/size, header checkpoint, model/head, epoch và split; không gọi đó là audit sâu mọi tensor.
- B chưa audit sâu mức cập nhật từng block; C chưa audit sâu mọi frozen tensor như A.
- C không bàn giao checkpoint pilot; thiếu weights pilot không có nghĩa pilot chưa chạy.
- Raw C `selected_from=lan_4` là nguồn LR2/local lan_18; raw B `selected_from=lan_25` là nguồn LR2/local lan_22.
  Phải tra manifest trước khi resume; không đổi metadata gốc hoặc resolve alias trực tiếp như run local.
- Gói C có test nguồn, chưa xác nhận đủ bốn cấu hình/checkpoint được khóa trước test.
  Không dùng test này để tuning; chưa coi là bảng test cuối của nhóm. A/B chưa có test artifact trong final local.
- A/C có dấu hiệu overfitting cuối run; giữ best checkpoint. Gap train online/val không phải đánh giá cùng checkpoint/transform.
- Một seed chưa chứng minh độ ổn định hoặc ưu thế tổng quát; cần D và đánh giá đúng protocol để kết luận đầy đủ.

## Sản phẩm trình bày và phần còn thiếu

- [HTML trình bày](../visualization/presentation.html) bám10 nội dung PDF ViT, có snapshot history A/B/C thật và tương tác minh họa.
- [Hướng dẫn HTML](design/Presentation.md): đọc/trình chiếu/in PDF; canvas attention là minh họa, chưa là attention từ model.
- `Coursework_Report.ipynb` hiện0 cell; `Training_Monitor.ipynb` đọc artifact, không thay thế notebook demo yêu cầu.
- Còn thiếu: D, bộ10–20 ảnh có nhãn, Top-5 trước/sau che,≥3 trường hợp phân tích và notebook demo/hướng dẫn.
- Chưa có bảng macro-F1/lỗi theo lớp/benchmark cuối chung; không điền số liệu giả để lấp khoảng trống.
