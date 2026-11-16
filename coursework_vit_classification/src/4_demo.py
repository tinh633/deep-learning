"""What: Dự đoán một ảnh bằng checkpoint. Why: Demo đóng trong 100 lớp CIFAR-100.
Who: Thành viên trình bày. When: Sau khi có checkpoint và chủ động chọn ảnh.
Where: Ảnh local, outputs/runs/lan_N/demo. How: Dùng chung model/transform, eval và Top-5.
"""
import sys
from pathlib import Path

import torch
from PIL import Image

import config
from data import evaluation_transform, preprocessing_settings
from logger import guard_memory, read_checkpoint, resolve_run, save_json
from model import model_from_checkpoint


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    if config.DEMO_IMAGE is None:
        raise SystemExit('Chưa chọn DEMO_IMAGE trong config.py')
    if config.EVAL_CHECKPOINT not in ('best.pt', 'last.pt'):
        raise ValueError('Checkpoint phải là best.pt hoặc last.pt')
    image_path = Path(config.DEMO_IMAGE)
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    folder = resolve_run(config.EVAL_RUN)
    guard_memory()
    state = read_checkpoint(folder / config.EVAL_CHECKPOINT)
    settings = state['settings']
    if settings['phase'] == 'smoke' and not config.ALLOW_SMOKE_CHECKPOINT:
        raise SystemExit('Checkpoint smoke chưa phù hợp demo chất lượng; cần cho phép rõ ràng.')
    if settings.get('preprocessing') != preprocessing_settings() or len(settings['classes']) != 100:
        raise ValueError('Checkpoint không tương thích preprocessing/classes hiện tại')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_from_checkpoint(state, device)
    epoch = state['epoch']
    del state
    with Image.open(image_path) as image:
        tensor = evaluation_transform()(image.convert('RGB')).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(tensor)
        if logits.shape != (1, 100) or not torch.isfinite(logits).all():
            raise ValueError('Logits sai shape hoặc NaN/Inf')
        scores, labels = logits.softmax(1).topk(5, dim=1)
    top5 = [dict(label=int(label), class_name=settings['classes'][int(label)], score=float(score))
            for score, label in zip(scores[0].cpu(), labels[0].cpu())]
    guard_memory()
    output = folder / 'demo'
    output.mkdir(exist_ok=True)
    number = 1
    while (output / f'prediction_{number}.json').exists():
        number += 1
    save_json(output / f'prediction_{number}.json', dict(image=str(image_path), model=settings['model'],
              checkpoint=config.EVAL_CHECKPOINT, phase=settings['phase'], epoch=epoch, top5=top5,
              limitation='Closed set 100 CIFAR classes; confidence is not calibrated probability.'))
    for item in top5:
        print(f'{item["class_name"]}: {item["score"]:.2%}')
    print('Ảnh ngoài 100 lớp vẫn bị ép vào một lớp; không dùng confidence để khẳng định ảnh đúng miền.')
