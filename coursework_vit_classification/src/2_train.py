"""What: Train một pilot hoặc final. Why: Cùng vòng lặp cho bốn chiến lược A–D.
Who: Nhóm sau khi chốt protocol. When: Sau smoke và cấu hình đã được duyệt.
Where: outputs/runs/lan_N. How: Loop train/val trực tiếp, AdamW, best/last và resume epoch.
"""
import hashlib
import json
import math
import random
import sys
import time

import numpy as np
import torch
from torch import nn

import config
from data import build_loaders, preprocessing_settings
from logger import (create_run, guard_memory, latest_checkpoint, load_checkpoint,
                    memory_snapshot, read_checkpoint, resolve_run, save_checkpoint, save_json)
from model import build_model, model_from_checkpoint, set_train_mode


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    if config.PHASE not in ('tuning', 'final'):
        raise SystemExit('2_train.py chỉ chạy tuning/final. Smoke: chạy src/0_smoke.py.')
    if not config.TRAINING_APPROVED or config.DUPLICATE_POLICY != 'keep_reported':
        raise SystemExit('Chưa chốt protocol/duplicate policy; không tự bắt đầu train dài.')
    if config.EXPERIMENT not in config.EXPERIMENTS:
        raise ValueError('EXPERIMENT phải là A/B/C/D')
    for name in ('BATCH_SIZE', 'EPOCHS'):
        value = getattr(config, name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f'{name} phải là số nguyên dương đã chốt')
    if config.LEARNING_RATE is None or not math.isfinite(config.LEARNING_RATE) or config.LEARNING_RATE <= 0:
        raise ValueError('Chưa khai báo LEARNING_RATE hợp lệ')
    if config.WEIGHT_DECAY is None or not math.isfinite(config.WEIGHT_DECAY) or config.WEIGHT_DECAY < 0:
        raise ValueError('Chưa khai báo WEIGHT_DECAY hợp lệ')
    if config.PHASE == 'tuning' and config.CANDIDATE not in ('LR1', 'LR2'):
        raise ValueError('Tuning cần CANDIDATE=LR1 hoặc LR2')
    if config.PHASE == 'final' and (config.CANDIDATE is not None or config.SELECTED_FROM is None):
        raise ValueError('Final cần SELECTED_FROM, CANDIDATE=None')
    if config.PHASE == 'tuning' and config.SELECTED_FROM is not None:
        raise ValueError('Pilot khởi tạo từ pretrained, không chọn từ một run khác')

    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp = config.USE_AMP and device.type == 'cuda'
    guard_memory()
    train_loader, val_loader, classes = build_loaders(config.BATCH_SIZE, config.NUM_WORKERS)
    experiment = config.EXPERIMENTS[config.EXPERIMENT]
    settings = dict(phase=config.PHASE, purpose=config.RUN_PURPOSE, experiment=config.EXPERIMENT, **experiment,
                    candidate=config.CANDIDATE, selected_from=config.SELECTED_FROM,
                    seed=config.SEED, batch_size=config.BATCH_SIZE, workers=config.NUM_WORKERS,
                    lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY,
                    epochs=config.EPOCHS, use_amp=amp, tf32=False, device=str(device),
                    gpu=torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
                    duplicate_policy=config.DUPLICATE_POLICY, preprocessing=preprocessing_settings(),
                    split_sha256=hashlib.sha256(config.SPLIT_FILE.read_bytes()).hexdigest(),
                    classes=classes, optimizer='AdamW', scheduler=None, loss='CrossEntropyLoss',
                    bn_policy='eval when frozen; train when fine-tuning',
                    weights='IMAGENET1K_V1', train_count=len(train_loader.dataset), val_count=len(val_loader.dataset))
    if config.PHASE == 'final':
        pilot_folder = resolve_run(config.SELECTED_FROM)
        pilot = read_checkpoint(pilot_folder / 'best.pt')['settings']
        pilot_status = (pilot_folder / 'summary.json').read_text(encoding='utf-8')
        if pilot.get('phase') != 'tuning' or json.loads(pilot_status).get('status') != 'completed':
            raise ValueError('SELECTED_FROM phải là pilot đã hoàn tất')
        if pilot.get('purpose', 'experiment') != 'experiment':
            raise ValueError('Không chọn LR final từ resource_probe')
        for key in ('experiment', 'lr', 'weight_decay', 'batch_size', 'seed', 'preprocessing',
                    'split_sha256', 'duplicate_policy', 'use_amp', 'workers'):
            if pilot.get(key) != settings[key]:
                raise ValueError(f'Final khác pilot đã chọn ở {key}')

    resume_path = latest_checkpoint(resolve_run(config.RESUME_RUN)) if config.RESUME_RUN else None
    folder, log = create_run(settings, resume_path.parent if resume_path else None)
    summary = dict(status='running', phase=config.PHASE)
    try:
        # Resume đọc weights từ checkpoint, không tải lại pretrained.
        if resume_path is None:
            model = build_model(settings['model'], settings['strategy']).to(device)
        else:
            state = read_checkpoint(resume_path)
            model = model_from_checkpoint(state, device)
            del state
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                     lr=settings['lr'], weight_decay=settings['weight_decay'], foreach=False)
        scaler = torch.amp.GradScaler('cuda', enabled=amp, init_scale=128)
        criterion = nn.CrossEntropyLoss()
        (folder / 'model.txt').write_text(str(model), encoding='utf-8')
        settings.update(total_parameters=sum(p.numel() for p in model.parameters()),
                        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
        if resume_path is None:
            save_json(folder / 'config.json', settings)
        start_epoch, history, best_accuracy, best_loss = 1, [], -1.0, float('inf')
        if resume_path is not None:
            state = read_checkpoint(resume_path)
            extra = state['extra']
            history, best_accuracy, best_loss = extra['history'], extra['best_accuracy'], extra['best_loss']
            del state
            start_epoch = load_checkpoint(resume_path, model, optimizer, scaler,
                                          train_loader.generator, expected=settings) + 1
            guard_memory()
            log.info('Resume %s, bắt đầu epoch %s', resume_path.name, start_epoch)

        for epoch in range(start_epoch, settings['epochs'] + 1):
            start = time.perf_counter()
            set_train_mode(model, settings['strategy'])
            loss_sum = correct = count = skipped = 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=amp):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Train loss NaN/Inf')
                scale_before = scaler.get_scale()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                skipped += int(scaler.get_scale() < scale_before)
                loss_sum += float(loss.detach()) * len(labels)
                correct += int((outputs.detach().argmax(1) == labels).sum())
                count += len(labels)
                guard_memory()
            train_loss, train_accuracy = loss_sum / count, correct / count

            optimizer.zero_grad(set_to_none=True)
            model.eval()
            val_sum = val_correct = val_count = 0
            with torch.inference_mode():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Validation loss NaN/Inf')
                    val_sum += float(loss) * len(labels)
                    val_correct += int((outputs.argmax(1) == labels).sum())
                    val_count += len(labels)
                    guard_memory()
            val_loss, val_accuracy = val_sum / val_count, val_correct / val_count
            improved = val_accuracy > best_accuracy or (val_accuracy == best_accuracy and val_loss < best_loss)
            if improved:
                best_accuracy, best_loss = val_accuracy, val_loss
            row = dict(epoch=epoch, train_loss=train_loss, train_accuracy=train_accuracy,
                       val_loss=val_loss, val_accuracy=val_accuracy, lr=optimizer.param_groups[0]['lr'],
                       seconds=time.perf_counter()-start, skipped_amp_steps=skipped)
            history.append(row)
            extra = dict(history=history, best_accuracy=best_accuracy, best_loss=best_loss)
            guard_memory()
            if improved:
                save_checkpoint(folder/'best.pt', model, optimizer, scaler, epoch, train_loader.generator, settings, extra)
                guard_memory()
            save_checkpoint(folder/'last.pt', model, optimizer, scaler, epoch, train_loader.generator, settings, extra)
            guard_memory()
            save_json(folder/'history.json', history)
            log.info('Epoch %d/%d | train %.4f %.2f%% | val %.4f %.2f%% | LR %.3g | %.1fs | skipped %d',
                     epoch, settings['epochs'], train_loss, 100*train_accuracy,
                     val_loss, 100*val_accuracy, row['lr'], row['seconds'], skipped)
        summary.update(status='completed', epochs=len(history), best_accuracy=best_accuracy, best_loss=best_loss)
    except BaseException as error:
        summary.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                       error=f'{type(error).__name__}: {error}')
        log.exception('Run dừng; giữ checkpoint epoch hoàn tất, không ghi checkpoint giữa batch.')
        raise
    finally:
        summary['memory_at_end'] = memory_snapshot()
        save_json(folder/'summary.json', summary)
        for handler in list(log.handlers):
            handler.close()
            log.removeHandler(handler)
