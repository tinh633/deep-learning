# Chỉ mục các lần chạy coursework

Cập nhật 27/11/2026. Đối chiếu `config.json`, `history.json`, `summary.json` và manifest bàn giao.
Số `lan_N` là thứ tự lưu artifact local, không phải số lần train chính thức.

## Cách phân biệt

- **Smoke** (`phase=smoke`): kiểm tra code, gradient, tài nguyên, checkpoint/resume trên ít ảnh.
  Epoch 1 trong smoke là mini-epoch, không phải một lượt qua đủ 45.000 ảnh train.
  `passed` chỉ xác nhận các kiểm tra kỹ thuật của lượt đó; accuracy smoke không dùng so sánh model.
- **Pilot** (`phase=tuning`): train đủ tập train, mỗi lượt 5 epoch để chọn learning rate bằng validation.
  Đây là lượt tuning, chưa phải kết quả final để đưa vào bảng so sánh chính.
- **Final** (`phase=final`): train chính thức 20 epoch, khởi tạo mới từ pretrained/head mới,
  dùng LR thắng pilot. `best.pt` được chọn theo validation accuracy, hòa thì validation loss thấp hơn.
- **Resource probe**: thử khả năng chạy, không dùng chọn LR hoặc làm kết quả final.
  Lượt probe ResNet lịch sử đã xóa; xem ghi chú cuối trang.

`completed` nghĩa là hoàn tất ngân sách đã cấu hình, không đồng nghĩa model tốt nhất trong mọi thử nghiệm.
Các accuracy bên dưới đều là **validation**, không phải test; best checkpoint có thể ở trước epoch 20.

## Các lượt train chính thức — dùng cho bảng kết quả chính

| Exp | Model và chế độ | Run | Epoch hoàn tất | LR | Best val accuracy | Best epoch |
|---|---|---|---|---|---|---|
| A | ConvNeXt-Base, freeze backbone | [lan_14](lan_14/) | 20/20 | 3e-4 | 81,46% | 9 |
| B | ConvNeXt-Base, fine-tune toàn bộ | [lan_23](lan_23/) | 20/20 | 3e-5 | 89,70% | 13 |
| C | ViT-B/16, freeze backbone | [lan_19](lan_19/) | 20/20 | 3e-4 | 82,08% | 9 |
| D | ViT-B/16, fine-tune toàn bộ | Chưa có final local | — | — | — | — |

Trong mỗi run final: dùng `best.pt` để đánh giá; `last.pt` lưu trạng thái cuối để resume/đối chiếu.
A có guard dừng giữa epoch 20 rồi resume; history cuối đủ 20 epoch trong cùng run, không tính thêm final.
C có resume Colab → Kaggle; môi trường và giới hạn được ghi trong metadata nguồn.

## Các pilot chọn learning rate

| Exp | Run | Ứng viên | LR | Epoch | Best val accuracy | Quyết định |
|---|---|---|---|---|---|---|
| A | [lan_12](lan_12/) | LR1 | 1e-3 | 5/5 | 80,64% | Không chọn |
| A | [lan_13](lan_13/) | LR2 | 3e-4 | 5/5 | 80,78% | Chọn cho final lan_14 |
| B | [lan_21](lan_21/) | LR1 | 1e-4 | 5/5 | 87,84% | Không chọn |
| B | [lan_22](lan_22/) | LR2 | 3e-5 | 5/5 | 89,42% | Chọn cho final lan_23 |
| C | [lan_17](lan_17/) | LR1 | 1e-3 | 5/5 | 80,56% | Không chọn |
| C | [lan_18](lan_18/) | LR2 | 3e-4 | 5/5 | 81,84% | Chọn cho final lan_19 |

Các pilot trên đều `completed`; final lấy cấu hình LR, không nối tiếp weights của pilot.
D chưa có pilot local. Không gộp kết quả pilot vào bảng final.

## Các smoke — kiểm tra kỹ thuật

| Run | Exp/model | Trạng thái | Vai trò hoặc giới hạn |
|---|---|---|---|
| [lan_1](lan_1/) | A / ResNet-50 freeze cũ | passed | Smoke lịch sử trước khi đổi CNN sang ConvNeXt |
| [lan_2](lan_2/) | B / ResNet-50 fine-tune cũ | passed | Smoke lịch sử, không đại diện B hiện tại |
| [lan_3](lan_3/) | C / ViT-B/16 freeze | interrupted | Bị ngắt, không coi là smoke đạt |
| [lan_4](lan_4/) | A / ResNet-50 freeze cũ | passed | Recheck lịch sử |
| [lan_5](lan_5/) | C / ViT-B/16 freeze | passed | Recheck kỹ thuật |
| [lan_6](lan_6/) | D / ViT-B/16 fine-tune | failed | Guard RAM/Windows commit; nhãn gradients FAIL legacy không chứng minh lỗi gradient |
| [lan_7](lan_7/) | D / ViT-B/16 fine-tune | failed | Guard RAM/Windows commit |
| [lan_8](lan_8/) | B / ResNet-50 fine-tune cũ | passed | Recheck lịch sử |
| [lan_9](lan_9/) | A / ConvNeXt-Base freeze | failed | Lỗi tải pretrained, 0 batch |
| [lan_10](lan_10/) | A / ConvNeXt-Base freeze | failed | Guard bộ nhớ sau 1 batch; chưa hoàn tất workflow |
| [lan_11](lan_11/) | A / ConvNeXt-Base freeze | passed | Smoke batch 16, 32 train/16 val; đủ 12 check |
| [lan_15](lan_15/) | C / ViT-B/16 freeze | passed | Smoke nguồn C, tên nguồn lan_1 |
| [lan_16](lan_16/) | C / ViT-B/16 freeze | passed | Smoke nguồn C, tên nguồn lan_2 |
| [lan_20](lan_20/) | B / ConvNeXt-Base fine-tune | passed | Smoke nguồn B, 32 train/16 val; đủ 12 check |

`replay_reference.pt` ở lan_20 chỉ dùng đối chiếu smoke, không phải checkpoint phân loại để báo cáo.

## Nguồn bàn giao và lưu ý khi đọc tên run

- C: `smoke/lan_1, lan_2` → local `lan_15, lan_16`;
  `training/lan_3, lan_4, lan_5` → local `lan_17, lan_18, lan_19`.
- B: `smoke, LR1, LR2, final` → local `lan_20, lan_21, lan_22, lan_23`.
- Giữ nguyên raw config/log/checkpoint: `selected_from=lan_4` trong C nghĩa là local `lan_18`;
  `selected_from=lan_25` trong B nghĩa là local `lan_22`. Phải áp dụng mapping trước khi resume.
- C chỉ bàn giao checkpoint final; thiếu checkpoint pilot không có nghĩa pilot chưa chạy.
- Gói C có artifact test nguồn; chưa xác nhận đủ bốn checkpoint được khóa trước test.
  Không dùng kết quả test đó để tuning tiếp; nhập bàn giao không chạy test mới.
- `lan_9` ResNet resource probe lịch sử đã xóa; thư mục `lan_9` hiện tại là ConvNeXt smoke lỗi tải mạng.
  Không gán kết quả probe cũ cho thư mục hiện tại.

Chi tiết: [manifest B](../import_B_manifest.json), [manifest C](../import_C_manifest.json),
[checkpoint/hash](../checkpoints.json), [audit kỹ thuật](../../docs/Audit_Log.md),
[protocol](../../docs/design/Training_Strategy.md), [tiến độ](../../docs/progress.md).
