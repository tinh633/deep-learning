"""What: Model pretrained và freeze/fine-tune. Why: Cùng head đơn giản 100 lớp.
Which: ConvNeXt-Base, ViT-B/16 và ResNet-50 (B: chỉ mở layer4 + fc). When: Tạo model hoặc đặt lại mode trước train.
Where: ImageNet-1K weights cache local. How: torchvision, thay Linear, requires_grad.
"""
import hashlib
from urllib.parse import urlparse

import torch
from torch import nn
from torchvision import models

from config import DATA_DIR


def build_model(name, strategy, pretrained=True):
    if strategy not in ('freeze', 'fine_tune', 'fine_tune_layer4'):
        raise ValueError('strategy phải là freeze, fine_tune hoặc fine_tune_layer4')
    if strategy == 'fine_tune_layer4' and name != 'resnet50':
        raise ValueError('fine_tune_layer4 chỉ dành cho resnet50')
    if pretrained:
        torch.hub.set_dir(str(DATA_DIR / 'torch_cache'))
    if name == 'convnext_base':
        weights = models.ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
    elif name == 'vit_b_16':
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
    elif name == 'resnet50':
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    else:
        raise ValueError(f'Model không hỗ trợ: {name}')

    if pretrained:
        filename = urlparse(weights.url).path.rsplit('/', 1)[-1]
        path = DATA_DIR / 'torch_cache' / 'checkpoints' / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        prefix = path.stem.rsplit('-', 1)[-1]
        if not path.exists():
            torch.hub.download_url_to_file(weights.url, str(path), hash_prefix=prefix)
        with path.open('rb') as stream:
            if not hashlib.file_digest(stream, 'sha256').hexdigest().startswith(prefix):
                raise ValueError('Pretrained weights checksum mismatch')
        # Không tạo thêm một bản weights khởi tạo lớn trong RAM trước khi nạp pretrained.
        with torch.device('meta'):
            model = new_architecture(name)
        state = torch.load(path, mmap=True, map_location='cpu', weights_only=True)
        model.load_state_dict(state, assign=True, strict=True)
    else:
        model = new_architecture(name)
    if name == 'convnext_base':
        model.classifier[2] = nn.Linear(1024, 100)
    elif name == 'resnet50':
        model.fc = nn.Linear(2048, 100)
    else:
        model.heads.head = nn.Linear(768, 100)

    for parameter in model.parameters():
        parameter.requires_grad = strategy == 'fine_tune'
    if strategy == 'fine_tune_layer4':
        # B: ResNet-50 giữ conv1, layer1-3 đóng băng; chỉ layer4 (14.964.736) + fc được học.
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True
    head = get_head(model)
    for parameter in head.parameters():
        parameter.requires_grad = True
    return model


def new_architecture(name):
    if name == 'convnext_base':
        return models.convnext_base()
    if name == 'vit_b_16':
        return models.vit_b_16()
    return models.resnet50()


def get_head(model):
    if hasattr(model, 'classifier'):
        return model.classifier[2]
    return model.fc if hasattr(model, 'fc') else model.heads.head


def set_train_mode(model, strategy):
    if strategy == 'freeze':
        # Giữ backbone ở eval để tắt stochastic depth khi chỉ train head.
        model.eval()
        get_head(model).train()
    elif strategy == 'fine_tune':
        model.train()
    elif strategy == 'fine_tune_layer4':
        # conv1/layer1-3 giữ BatchNorm ở eval (dùng thống kê ImageNet); chỉ layer4 và fc ở train.
        model.eval()
        model.layer4.train()
        model.fc.train()
    else:
        raise ValueError('strategy phải là freeze, fine_tune hoặc fine_tune_layer4')


def model_from_checkpoint(state, device):
    settings = state['settings']
    with torch.device('meta'):
        model = build_model(settings['model'], settings['strategy'], pretrained=False)
    model.load_state_dict(state['model'], assign=True, strict=True)
    return model.to(device).eval()
