"""What: Hai model pretrained và freeze/fine-tune/partial. Why: Cùng head đơn giản 100 lớp.
Which: ConvNeXt-Base và ViT-B/16. When: Tạo model hoặc đặt lại mode trước train.
Where: ImageNet-1K weights cache local. How: torchvision, thay Linear, requires_grad.
"""
import hashlib
from urllib.parse import urlparse

import torch
from torch import nn
from torchvision import models

from config import DATA_DIR


STRATEGIES = ('freeze', 'fine_tune', 'partial')


def partial_modules(model):
    # Partial (ConvNeXt): block cuối stage 3 (features[5]) và stage 4 (features[7]), LayerNorm cuối, head.
    return [model.features[5][-1], model.features[7][-1], model.classifier[0], model.classifier[2]]


def build_model(name, strategy, pretrained=True):
    if strategy not in STRATEGIES:
        raise ValueError('strategy phải là freeze, fine_tune hoặc partial')
    if strategy == 'partial' and name != 'convnext_base':
        raise ValueError('partial hiện chỉ hỗ trợ convnext_base')
    if pretrained:
        torch.hub.set_dir(str(DATA_DIR / 'torch_cache'))
    if name == 'convnext_base':
        weights = models.ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
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
            model = models.convnext_base() if name == 'convnext_base' else models.vit_b_16()
        state = torch.load(path, mmap=True, map_location='cpu', weights_only=True)
        model.load_state_dict(state, assign=True, strict=True)
    else:
        model = models.convnext_base() if name == 'convnext_base' else models.vit_b_16()
    if name == 'convnext_base':
        model.classifier[2] = nn.Linear(1024, 100)
    else:
        model.heads.head = nn.Linear(768, 100)

    for parameter in model.parameters():
        parameter.requires_grad = strategy == 'fine_tune'
    head = get_head(model)
    for parameter in head.parameters():
        parameter.requires_grad = True
    if strategy == 'partial':
        for module in partial_modules(model):
            for parameter in module.parameters():
                parameter.requires_grad = True
    return model


def get_head(model):
    return model.classifier[2] if hasattr(model, 'classifier') else model.heads.head


def set_train_mode(model, strategy):
    if strategy == 'freeze':
        # Giữ backbone ở eval để tắt stochastic depth khi chỉ train head.
        model.eval()
        get_head(model).train()
    elif strategy == 'fine_tune':
        model.train()
    elif strategy == 'partial':
        # Phần đóng băng giữ eval; chỉ các module được mở mới train (gồm stochastic depth của block đó).
        model.eval()
        for module in partial_modules(model):
            module.train()
    else:
        raise ValueError('strategy phải là freeze, fine_tune hoặc partial')


def model_from_checkpoint(state, device):
    settings = state['settings']
    with torch.device('meta'):
        model = build_model(settings['model'], settings['strategy'], pretrained=False)
    model.load_state_dict(state['model'], assign=True, strict=True)
    return model.to(device).eval()
