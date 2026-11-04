# model.py — Backbone pretrained và classifier

Source: [model.py](../../src/model.py).
PretrainedCNN(name, fine_tune=False, pretrained=True) là nn.Module dùng chung
cho train/evaluate/notebook. Tên không hỗ trợ làm phát sinh ValueError.

## Khởi tạo model

| name | Weights khi pretrained=True | Backbone | Feature vector |
|---|---|---|---:|
| vgg16 | VGG16_Weights.IMAGENET1K_V1 | original.features | 512 |
| densenet121 | DenseNet121_Weights.IMAGENET1K_V1 | original.features | 1024 |

Torchvision lấy weights từ cache hoặc tải nếu thiếu. pretrained=False dùng
khởi tạo ngẫu nhiên để sau đó nạp state_dict khi evaluate/resume.
Không giữ classifier ImageNet1.000 lớp của model gốc.
Head mới là Linear(feature_size,10), khởi tạo ngẫu nhiên và có tham số được học.

## Freeze và fine-tune

Đầu tiên toàn bộ backbone có requires_grad=False; head vẫn requires_grad=True.
Khi fine_tune=True, chỉ mở lại các nhóm cuối:

- VGG16: convolution24,26,28 trong block5.
- DenseNet121: denseblock4 và norm5.

Freeze không bỏ forward: ảnh vẫn đi qua backbone để trích xuất đặc trưng.
Train chỉ đưa parameter.requires_grad=True vào optimizer.
A train head; B/C train head và nhóm cuối ngay từ epoch đầu,
không có lịch tự mở thêm lớp sau một số epoch.

## Vì sao override train(mode=True)?

super().train(mode) đặt mode của model/head; backbone được đưa về eval.
Nếu mode=True và fine_tune=True, các nhóm cuối được bật train,
sau đó mọi BatchNorm trong backbone lại được đặt eval.
Do đó running_mean/running_var của BatchNorm giữ nguyên.
Tham số affine weight/bias ở nhóm mở vẫn nhận gradient và có thể được cập nhật.
requires_grad và train/eval kiểm soát hai việc khác nhau;
eval không tự tắt gradient, còn freeze không tự khóa running statistics.
model.eval() gọi train(False), đưa toàn model vào chế độ đánh giá.

## forward(images)

1. backbone trích feature map.
2. DenseNet thêm ReLU sau features; VGG đã có activation trong backbone.
3. AdaptiveAvgPool2d(1,1) lấy trung bình mỗi kênh.
4. Flatten tạo vector512 hoặc1024 chiều.
5. Head Linear trả logits `[B,10]`.

GAP và Flatten không có tham số học. Linear là fully connected có train.
Không softmax trong forward vì CrossEntropyLoss nhận logits;
evaluate gọi softmax khi cần confidence và xác suất ROC.
Cùng loại head không có nghĩa hai model cùng số tham số hoặc năng lực.

Liên quan: [train](2_train.md), [config](config.md), [Model_QA](../../Model_QA.md).
