# model.py — Model và chế độ học

Source: [model.py](../../src/model.py).
Kiến trúc: [ConvNeXt-Base](../design/ConvNeXt_Base.md), [ViT-B/16](../design/ViT_B16.md).

| 5W1H | Mô tả |
|---|---|
| What | ConvNeXt-Base/ViT-B/16, head 100 lớp, freeze/fine-tune và nạp model đã train |
| Why | Cùng classifier đơn giản, đặt đúng phần được học |
| Which | ConvNeXt-Base hoặc ViT-B/16, weights IMAGENET1K_V1, freeze hoặc fine_tune |
| When | Trước optimizer, trước train pass hoặc khi đọc checkpoint |
| Where | Weights ImageNet-1K cache local; state_dict trong checkpoint run |
| How | torchvision, meta+mmap/assign, Linear mới và requires_grad/mode |

## Các hàm

- build_model(name, strategy, pretrained=True): trả model, không trả tuple metadata.
- get_head(model): trả classifier[2] của ConvNeXt hoặc heads.head của ViT.
- set_train_mode(model, strategy): freeze giữ backbone eval/head train; fine_tune model.train().
- model_from_checkpoint(state, device): dựng kiến trúc trên meta, strict load rồi chuyển device/eval.

ConvNeXt head Linear(1024,100), ViT head Linear(768,100); đầu ra là logits B×100.
Pretrained chọn rõ IMAGENET1K_V1, tải nếu thiếu, kiểm tra hash prefix URL trước khi nạp.
Meta/mmap giảm bản sao RAM, không thay model hoặc lượng parameters.
pretrained=False không tải weights; evaluate/demo nạp trực tiếp state_dict checkpoint, không tải pretrained lại.
Freeze chỉ mở Linear mới, giữ cả LayerNorm cuối của ConvNeXt bất biến.
Fine-tune mở toàn bộ parameters và bật stochastic depth của ConvNeXt; không có BatchNorm.
Validation/test/demo luôn eval; model_from_checkpoint không tự áp dụng augmentation hoặc softmax.

Chỉ hỗ trợ hai model và hai strategy đã chốt; tên không hợp lệ bị từ chối.
Checkpoint ResNet cũ bị từ chối; không đổi nhãn metadata để nạp vào ConvNeXt.
Không có teacher/distillation hoặc nhiều head.
Strategy partial (chỉ ConvNeXt, mã thí nghiệm E/B'): mở block cuối stage 3 (features[5][-1]), block cuối
stage 4 (features[7][-1]), LayerNorm cuối và head = 10.679.396 tham số; phần còn lại freeze và eval.
set_train_mode(partial) chỉ bật train cho các module được mở. ViT + partial bị từ chối.
Chưa smoke GPU hay train; cần duyệt protocol trước khi chạy pilot.
