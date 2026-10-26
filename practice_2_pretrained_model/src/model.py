from torch import nn
from torchvision import models


class PretrainedCNN(nn.Module):
    """Explicit torchvision backbones with the same CIFAR-10 head policy."""

    def __init__(self, name, head='H1', fine_tune=False, pretrained=True):
        super().__init__()
        if name not in ('vgg16', 'resnet18', 'densenet121'):
            raise ValueError(f'Unsupported model: {name}')
        if head not in ('H1', 'H2'):
            raise ValueError(f'Unsupported head: {head}')
        self.name = name
        self.head_name = head
        self.fine_tune = fine_tune

        if name == 'vgg16':
            weights = models.VGG16_Weights.DEFAULT if pretrained else None
            original = models.vgg16(weights=weights)
            self.backbone = original.features
            self.feature_size = 512
            self.final_blocks = ('24', '26', '28')  # Conv layers in VGG block 5.
        elif name == 'resnet18':
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            original = models.resnet18(weights=weights)
            original.fc = nn.Identity()
            self.backbone = original
            self.feature_size = 512
            self.final_blocks = ('layer4',)
        else:
            weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
            original = models.densenet121(weights=weights)
            self.backbone = original.features
            self.feature_size = 1024
            self.final_blocks = ('denseblock4', 'norm5')

        self.weights_name = str(weights) if weights is not None else 'None (random initialization)'
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        if head == 'H1':
            self.head = nn.Linear(self.feature_size, 10)
        else:
            self.head = nn.Sequential(
                nn.Linear(self.feature_size, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 10),
            )

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
        if self.name != 'resnet18':
            features = self.pool(features)
            features = self.flatten(features)
        logits = self.head(features)
        return logits
