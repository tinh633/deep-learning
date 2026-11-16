# model.py — Model và chế độ học

Source: [model.py](../../src/model.py).
Kiến trúc: [ResNet50](../design/ResNet50.md), [ViT-B/16](../design/ViT_B16.md).

| 5W1H | Mô tả |
|---|---|
| What | ResNet-50/ViT-B/16, head 100 lớp, freeze/fine-tune và nạp model đã train |
| Why | Cùng classifier đơn giản, đặt đúng phần được học |
| Which | ResNet-50 hoặc ViT-B/16, weights IMAGENET1K_V1, freeze hoặc fine_tune |
| When | Trước optimizer, trước train pass hoặc khi đọc checkpoint |
| Where | Weights ImageNet-1K cache local; state_dict trong checkpoint run |
| How | torchvision, meta+mmap/assign, Linear mới và requires_grad/mode |

## Các hàm

- build_model(name, strategy, pretrained=True): trả model, không trả tuple metadata.
- get_head(model): trả fc của ResNet hoặc heads.head của ViT.
- set_train_mode(model, strategy): freeze giữ backbone eval/head train; fine_tune model.train().
- model_from_checkpoint(state, device): dựng kiến trúc trên meta, strict load rồi chuyển device/eval.

ResNet head Linear(2048,100), ViT head Linear(768,100); đầu ra là logits B×100.
Pretrained chọn rõ IMAGENET1K_V1, tải nếu thiếu, kiểm tra hash prefix URL trước khi nạp.
Meta/mmap giảm bản sao RAM, không thay model hoặc lượng parameters.
pretrained=False không tải weights; evaluate/demo nạp trực tiếp state_dict checkpoint, không tải pretrained lại.
Freeze chỉ mở head. Fine-tune mở toàn bộ parameters; BatchNorm train trong ResNet fine-tune.
Validation/test/demo luôn eval; model_from_checkpoint không tự áp dụng augmentation hoặc softmax.

Chỉ hỗ trợ hai model và hai strategy đã chốt; tên không hợp lệ bị từ chối.
Không có teacher/distillation, nhiều head hoặc partial unfreeze trong phiên bản này.
