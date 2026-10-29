"""What: Tạo pretrained backbone và head phân loại CIFAR-10.
Why: So sánh freeze/fine-tune với chính sách nhất quán cho hai model.
Who: Train/evaluate dùng PretrainedCNN; notebook xem cấu trúc CPU.
When: Khởi tạo trước forward; train(mode) giữ chính sách BatchNorm.
Where: Weights torchvision và model_state trong checkpoint local.
How: Head Linear 10 lớp, freeze backbone và tùy chọn mở nhóm cuối.
"""

from torch import nn
from torchvision import models


class PretrainedCNN(nn.Module):
    """Explicit torchvision backbones with the same CIFAR-10 head policy."""

    def __init__(self, name, fine_tune=False, pretrained=True):
        super().__init__()
        if name not in ('vgg16', 'densenet121'):
            raise ValueError(f'Unsupported model: {name}')
        self.name = name
        self.fine_tune = fine_tune

        if name == 'vgg16':
            weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
            original = models.vgg16(weights=weights)
            self.backbone = original.features
            self.feature_size = 512
            self.final_blocks = ('24', '26', '28')  # Conv layers in VGG block 5.
        else:
            weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
            original = models.densenet121(weights=weights)
            self.backbone = original.features
            self.feature_size = 1024
            self.final_blocks = ('denseblock4', 'norm5')

        self.weights_name = str(weights) if weights is not None else 'None (random initialization)'
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.head = nn.Linear(self.feature_size, 10)

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        if fine_tune:
            for block_name in self.final_blocks:
                block = self.backbone.get_submodule(block_name)
                for parameter in block.parameters():
                    parameter.requires_grad = True
        self.train()

    def train(self, mode=True):
        # model.train() must not re-enable frozen BatchNorm running statistics.
        super().train(mode)
        self.backbone.eval()
        if mode and self.fine_tune:
            for block_name in self.final_blocks:
                self.backbone.get_submodule(block_name).train()
            for layer in self.backbone.modules():
                if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                    layer.eval()
        return self

    def forward(self, images):
        features = self.backbone(images)
        if self.name == 'densenet121':
            features = self.relu(features)
        features = self.pool(features)
        features = self.flatten(features)
        logits = self.head(features)
        return logits
