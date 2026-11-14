"""What: Hai model pretrained và freeze/fine-tune. Why: Cùng head đơn giản 100 lớp.
Who: Train/evaluate/demo. When: Tạo model hoặc đặt lại mode trước train.
Where: ImageNet-1K weights cache local. How: torchvision, thay Linear, requires_grad.
"""
import hashlib
from urllib.parse import urlparse

import torch
from torch import nn
from torchvision import models

from config import DATA_DIR


def build_model(name, strategy, pretrained=True):
    if strategy not in ('freeze', 'fine_tune'):
        raise ValueError('strategy phải là freeze hoặc fine_tune')
    if pretrained:
        torch.hub.set_dir(str(DATA_DIR / 'torch_cache'))
    if name == 'resnet50':
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    elif name == 'vit_b_16':
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
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
            model = models.resnet50() if name == 'resnet50' else models.vit_b_16()
        state = torch.load(path, mmap=True, map_location='cpu', weights_only=True)
        model.load_state_dict(state, assign=True, strict=True)
    else:
        model = models.resnet50() if name == 'resnet50' else models.vit_b_16()
    if name == 'resnet50':
        model.fc = nn.Linear(2048, 100)
    else:
        model.heads.head = nn.Linear(768, 100)

    for parameter in model.parameters():
        parameter.requires_grad = strategy == 'fine_tune'
    head = get_head(model)
    for parameter in head.parameters():
        parameter.requires_grad = True
    return model


def get_head(model):
    return model.fc if hasattr(model, 'fc') else model.heads.head


def set_train_mode(model, strategy):
    if strategy == 'freeze':
        # requires_grad=False không đủ giữ BatchNorm running statistics.
        model.eval()
        get_head(model).train()
    elif strategy == 'fine_tune':
        model.train()
    else:
        raise ValueError('strategy phải là freeze hoặc fine_tune')
