# Tải và sử dụng checkpoint

Cập nhật: 16/11/2026. Có 7 full checkpoint được lưu bằng Git LFS, tổng khoảng 1,21 GiB.
Danh sách chính xác, dung lượng và SHA-256: [outputs/checkpoints.json](../outputs/checkpoints.json).
Đây là checkpoint lịch sử ResNet/ViT; chưa có ConvNeXt hoặc final của cặp mới ngày 18/11.
Upload LFS trước bị kẹt; danh sách local không chứng minh remote đã có đủ model.

## 1. Các file được chia sẻ

| Run | File | Model/chế độ | Phạm vi thực tế |
|---|---|---|---|
| lan_1 | last.pt | ResNet-50 freeze | Smoke 20 ảnh train |
| lan_2 | last.pt | ResNet-50 fine-tune | Smoke 20 ảnh train |
| lan_4 | last.pt | ResNet-50 freeze | Smoke 20 ảnh train, recheck |
| lan_5 | last.pt | ViT-B/16 freeze | Smoke 20 ảnh train |
| lan_8 | last.pt | ResNet-50 fine-tune | Smoke 20 ảnh train, recheck |
| lan_9 | best.pt và last.pt | ResNet-50 freeze | Resource probe: 1 epoch 45k train/5k val |

Checkpoint smoke ghi epoch=1 nhưng đó là mini-epoch, không phải một lượt qua đủ 45.000 ảnh.
lan_9 mới là một epoch đầy đủ; validation accuracy 64,16%, chưa test, chưa tuning LR.
lan_3 bị ngắt, lan_6/7 không hoàn tất checkpoint; không tạo file để bù kết quả thiếu.
replay_reference.pt chỉ chứa state đối chiếu của bài kiểm tra, không chia sẻ như checkpoint sử dụng.
Dataset, pixel và cache pretrained tiếp tục được ignore.

## 2. Tải bằng Git LFS

Cài Git LFS trên máy, rồi từ thư mục repository:

```powershell
git lfs install
git switch cousework
git pull --ff-only origin cousework
git lfs pull
git lfs ls-files
```

Để chỉ lấy lượt A resource probe, có thể dùng:

```powershell
git lfs pull --include="coursework_vit_classification/outputs/runs/lan_9/*.pt"
```

Nếu file .pt chỉ chứa vài dòng bắt đầu bằng version https://git-lfs..., đó là pointer,
chưa phải weights; cần git lfs pull trước khi torch.load.
Không tải file từ trang xem pointer rồi cho rằng đã lấy model đầy đủ.

## 3. Giới hạn tương thích của lan_9

- Lan_9 dùng ResNet-50; code hiện chỉ hỗ trợ ConvNeXt-Base và ViT-B/16 nên từ chối nạp.
- Muốn tái lập ResNet dùng source commit 8da0358 trong checkout riêng, không đổi metadata checkpoint.
- Model mới phải được smoke/train và xuất checkpoint mới bằng pipeline hiện tại.
- Giữ RUN_TEST=False; lan_9 là resource_probe phase tuning, không đủ điều kiện official test final.
- Không dùng lan_9 làm SELECTED_FROM của final vì chưa qua quy trình chọn LR chính thức.

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
