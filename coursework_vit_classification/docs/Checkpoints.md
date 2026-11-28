# Tải và sử dụng checkpoint

Cập nhật 27/11/2026: manifest có 21 checkpoint và 1 smoke replay reference; B ở lan_20–23, C ở lan_15–19.
Danh sách chính xác, dung lượng và SHA-256: [outputs/checkpoints.json](../outputs/checkpoints.json).
ConvNeXt smoke/pilot/final đã hoàn tất, kiểm chứng và upload đủ 7 LFS mới tại commit 68641b3 trên cousework.
18/11: đã upload đủ 7 checkpoint Git LFS (1,3 GB) cùng code ConvNeXt tại commit d5e6dfc.

## 1. Các file được chia sẻ

| Run | File | Model/chế độ | Phạm vi thực tế |
|---|---|---|---|
| lan_1 | last.pt | ResNet-50 freeze | Smoke 20 ảnh train |
| lan_2 | last.pt | ResNet-50 fine-tune | Smoke 20 ảnh train |
| lan_4 | last.pt | ResNet-50 freeze | Smoke 20 ảnh train, recheck |
| lan_5 | last.pt | ViT-B/16 freeze | Smoke 20 ảnh train |
| lan_8 | last.pt | ResNet-50 fine-tune | Smoke 20 ảnh train, recheck |
| lan_11 | last.pt | ConvNeXt-Base freeze | Smoke 32 ảnh train, batch 16 |
| lan_12 | best.pt / last.pt | ConvNeXt-Base freeze | Pilot LR1, 5 epoch |
| lan_13 | best.pt / last.pt | ConvNeXt-Base freeze | Pilot LR2, 5 epoch |
| lan_14 | best.pt / last.pt | ConvNeXt-Base freeze | Final 20 epoch, best epoch 9 |
| lan_19 | best.pt / last.pt | ViT-B16 freeze | Final C 20 epoch, best epoch9; nguồn da0d199 |
| lan_20 | last.pt / replay_reference.pt | ConvNeXt-Base fine-tune | Smoke B, replay chỉ đối chiếu |
| lan_21/22 | best.pt / last.pt | ConvNeXt-Base fine-tune | Hai pilot B, mỗi lượt 5 epoch |
| lan_23 | best.pt / last.pt | ConvNeXt-Base fine-tune | Final B 20 epoch; best val 89,70% |

Checkpoint smoke ghi epoch=1 nhưng đó là mini-epoch, không phải một lượt qua đủ 45.000 ảnh.
Lan_9 ResNet cũ đã xóa; tên lan_9 mới là smoke ConvNeXt thất bại tải mạng, không có checkpoint.
lan_3 bị ngắt, lan_6/7 không hoàn tất checkpoint; không tạo file để bù kết quả thiếu.
replay_reference.pt chỉ phục vụ đối chiếu; B giữ bản nguồn để bảo toàn gói bàn giao, không dùng cho inference.
Dataset, pixel và cache pretrained tiếp tục được ignore.
Model A dùng tiếp: lan_14/best.pt; xem [A_Results](A_Results.md), không chọn checkpoint bằng test.
Mapping C: smoke lan_15/16, pilot lan_17/18, final lan_19; xem [import manifest](../outputs/import_C_manifest.json).
Raw selected_from=lan_4 của C là pilot nguồn/local lan_18, không phải local lan_4; không sửa metadata trong weights.
Gói C chỉ có checkpoint final; checkpoint pilot không được cung cấp. Đã có test nguồn, không chạy lại khi nhập.
Mapping B: smoke/LR1/LR2/final → lan_20/21/22/23; [manifest B](../outputs/import_B_manifest.json).
Raw B selected_from=lan_25 là LR2 nguồn/local lan_22; phải resolve mapping trước resume.
Nhập từ remote b754c2f; B/C do các thành viên khác train, local chuẩn hóa bản sao. Gói nguồn B_ban_giao/C_ban_giao giữ nguyên khi đồng bộ.

## 2. Tải bằng Git LFS

Cài Git LFS trên máy, rồi từ thư mục repository:

```powershell
git lfs install
git switch cousework
git pull --ff-only origin cousework
git lfs pull
git lfs ls-files
```

Ví dụ chỉ lấy checkpoint smoke ViT lịch sử:

```powershell
git lfs pull --include="coursework_vit_classification/outputs/runs/lan_5/*.pt"
```

Nếu file .pt chỉ chứa vài dòng bắt đầu bằng version https://git-lfs..., đó là pointer,
chưa phải weights; cần git lfs pull trước khi torch.load.
Không tải file từ trang xem pointer rồi cho rằng đã lấy model đầy đủ.

## 3. Giới hạn tương thích của ResNet lịch sử

- Checkpoint ResNet còn lại chỉ đọc với source cũ; code hiện chỉ hỗ trợ ConvNeXt-Base và ViT-B/16.
- Muốn tái lập ResNet dùng source commit 8da0358 trong checkout riêng, không đổi metadata checkpoint.
- Model mới phải được smoke/train và xuất checkpoint mới bằng pipeline hiện tại.
- Giữ RUN_TEST=False; smoke/probe không đủ điều kiện official test final.
- Không dùng smoke/probe làm SELECTED_FROM của final vì chưa qua quy trình chọn LR chính thức.

Chi tiết: [demo](src/4_demo.md), [evaluate](src/3_evaluate.md), [resume](src/logger.md).
Full checkpoint có model/optimizer/scaler, epoch, config và RNG; không phải chỉ state_dict đơn lẻ.
Code dùng state['model'] và dựng đúng kiến trúc/head 100 lớp, không thay bằng head ImageNet 1.000 lớp.
Chỉ nạp pickle/checkpoint tin cậy; môi trường phiên bản thực tế nằm trong config.json từng run.

## 4. Giới hạn checkpoint smoke cũ

Các checkpoint smoke phục vụ kiểm chứng code, không dùng chứng minh accuracy sau train đầy đủ.
Một số lượt cũ thiếu metadata preprocessing; evaluator hiện từ chối thay vì đoán cấu hình.
Không tự sửa checkpoint cũ hoặc đổi phase để vượt cờ kiểm tra.
Muốn tiếp tục train chính thức phải theo [Training_Strategy](design/Training_Strategy.md),
khởi tạo lại pretrained sau tuning; resume không đồng nghĩa chuyển smoke thành final.
