"""What: Smoke bốn chiến lược trước tuning. Why: Kiểm tra đúng pipeline và tài nguyên.
Who: Nhóm coursework. When: Sau data/EDA, trước train chính thức.
Where: outputs/runs/lan_N phase=smoke. How: Mini-epoch train/val trực tiếp và save/replay.
"""
import gc
import hashlib
import random
import sys
import time

import numpy as np
import torch
from torch import nn
from torchvision import models

import config
from data import build_loaders
from logger import (create_run, guard_memory, load_checkpoint, memory_snapshot,
                    save_checkpoint, save_json, state_fingerprints)
from model import build_model, get_head, set_train_mode


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if config.PHASE != 'smoke':
        raise SystemExit('Hiện chỉ hỗ trợ phase=smoke; chưa khóa tuning/final config.')
    if config.NUM_WORKERS != 0:
        raise SystemExit('Smoke replay hiện yêu cầu NUM_WORKERS=0 để kiểm chứng RNG trực tiếp.')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    failed = []

    for code in config.SMOKE_EXPERIMENTS:
        random.seed(config.SEED)
        np.random.seed(config.SEED)
        torch.manual_seed(config.SEED)
        experiment = config.EXPERIMENTS[code]
        name, strategy = experiment['model'], experiment['strategy']
        lr = config.SMOKE_LR_FREEZE if strategy == 'freeze' else config.SMOKE_LR_FINE_TUNE
        amp = config.USE_AMP and device.type == 'cuda'
        weights = (models.ResNet50_Weights.IMAGENET1K_V1 if name == 'resnet50'
                   else models.ViT_B_16_Weights.IMAGENET1K_V1)
        settings = dict(phase='smoke', experiment=code, model=name, strategy=strategy,
                        seed=config.SEED, lr=lr, optimizer='AdamW',
                        weight_decay=config.SMOKE_WEIGHT_DECAY, scheduler=None,
                        batch_size=config.SMOKE_BATCH_SIZE, workers=config.NUM_WORKERS,
                        precision='AMP float16' if amp else 'FP32', tf32=False,
                        device=str(device), gpu=torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
                        train_images=config.SMOKE_TRAIN_IMAGES, val_images=config.SMOKE_VAL_IMAGES,
                        scope='one miniature epoch; not a full CIFAR-100 epoch; no test access',
                        scheduled_experiments=list(config.SMOKE_EXPERIMENTS),
                        weights=str(weights), weights_url=weights.url,
                        bn_policy='eval when frozen; train when fine-tuning',
                        split_sha256=hashlib.sha256(config.SPLIT_FILE.read_bytes()).hexdigest())
        folder, log = create_run(settings)
        checks = dict.fromkeys(['source_split', 'trainable', 'optimizer', 'shape', 'finite_loss',
                                'gradients', 'state_update', 'validation', 'checkpoint',
                                'resume_replay', 'resources', 'stability'], 'NOT_RUN')
        summary = dict(status='running', checks=checks)
        model = optimizer = scaler = train_loader = val_loader = None
        frozen_before = reference_state = None
        current_check = 'source_split'
        durations, losses = [], []
        try:
            settings['memory_before_load'] = guard_memory()
            train_loader, val_loader, classes = build_loaders(
                config.SMOKE_BATCH_SIZE, config.NUM_WORKERS,
                config.SMOKE_TRAIN_IMAGES, config.SMOKE_VAL_IMAGES)
            settings.update(classes=classes, train_ids=train_loader.dataset.indices,
                            val_ids=val_loader.dataset.indices,
                            train_transform=str(train_loader.dataset.dataset.transform),
                            val_transform=str(val_loader.dataset.dataset.transform))
            log.info('Loading pretrained %s', weights)
            model = build_model(name, strategy).to(device)
            (folder / 'model.txt').write_text(str(model), encoding='utf-8')
            head = get_head(model)
            head_ids = {id(p) for p in head.parameters()}
            backbone = next(p for p in model.parameters() if id(p) not in head_ids)
            head_before = head.weight.detach().cpu().clone()
            backbone_before = backbone.detach().cpu().clone()
            if strategy == 'freeze':
                frozen_before = {k: v for k, v in state_fingerprints(model).items()
                                 if not k.startswith(('fc.', 'heads.'))}
            trainable = [p for p in model.parameters() if p.requires_grad]
            current_check = 'trainable'
            expected = 204900 if name == 'resnet50' else 76900
            assert (sum(p.numel() for p in trainable) == expected if strategy == 'freeze'
                    else all(p.requires_grad for p in model.parameters())), 'Trainable mismatch'
            settings.update(total_parameters=sum(p.numel() for p in model.parameters()),
                            trainable_parameters=sum(p.numel() for p in trainable))
            save_json(folder / 'config.json', settings)
            checks['source_split'] = checks['trainable'] = 'PASS'
            optimizer = torch.optim.AdamW(trainable, lr=lr,
                                         weight_decay=config.SMOKE_WEIGHT_DECAY, foreach=False)
            current_check = 'optimizer'
            assert {id(p) for group in optimizer.param_groups for p in group['params']} == {id(p) for p in trainable}
            checks['optimizer'] = 'PASS'
            criterion = nn.CrossEntropyLoss()
            scaler = torch.amp.GradScaler('cuda', enabled=amp, init_scale=128)
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats()

            # Một mini-epoch: 20 ảnh train, 10 batch. Không ghi log từng batch.
            set_train_mode(model, strategy)
            for images, labels in train_loader:
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                start = time.perf_counter()
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=amp):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                current_check = 'shape'
                assert outputs.shape == (len(labels), 100), 'Output shape'
                current_check = 'finite_loss'
                assert torch.isfinite(outputs).all() and torch.isfinite(loss), 'Non-finite output/loss'
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                current_check = 'gradients'
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable), 'Missing/non-finite gradient'
                assert all(p.grad is None for p in model.parameters() if not p.requires_grad), 'Frozen gradient'
                scaler.step(optimizer)
                scaler.update()
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                durations.append(time.perf_counter() - start)
                losses.append(float(loss.detach()))
                current_check = 'resources'
                guard_memory()
            checks['shape'] = checks['finite_loss'] = checks['gradients'] = 'PASS'
            current_check = 'state_update'
            assert not torch.equal(head_before, head.weight.detach().cpu()), 'Head did not update'
            if strategy == 'freeze':
                state_after = state_fingerprints(model)
                assert all(v == state_after[k] for k, v in frozen_before.items()), 'Frozen backbone/buffer changed'
            else:
                assert not torch.equal(backbone_before, backbone.detach().cpu()), 'Backbone did not update'
            checks['state_update'] = 'PASS'
            train_peak = torch.cuda.max_memory_allocated() / 2**20 if device.type == 'cuda' else None

            # Validation không cập nhật weights/buffers; không nạp official test.
            optimizer.zero_grad(set_to_none=True)
            current_check = 'validation'
            model.eval()
            buffers_before = {k: v.detach().cpu().clone() for k, v in model.named_buffers()}
            val_losses = []
            with torch.inference_mode():
                for images, labels in val_loader:
                    probe = images.to(device)
                    probe_logits = model(probe)
                    val_loss = criterion(probe_logits, labels.to(device))
                    assert torch.isfinite(val_loss)
                    val_losses.append(float(val_loss))
            assert all(not module.training for module in model.modules()), 'Validation mode'
            assert all(p.grad is None for p in model.parameters()), 'Validation gradients'
            assert all(torch.equal(v, dict(model.named_buffers())[k].detach().cpu())
                       for k, v in buffers_before.items()), 'Validation buffer changed'
            checks['validation'] = 'PASS'

            # Mini-epoch boundary: checkpoint bao gồm optimizer/scaler/RNG/loader generator.
            checkpoint = folder / 'last.pt'
            current_check = 'checkpoint'
            summary['memory_before_checkpoint'] = guard_memory()
            save_checkpoint(checkpoint, model, optimizer, scaler, 1, train_loader.generator, settings)
            epoch = load_checkpoint(checkpoint, model, optimizer, scaler, train_loader.generator)
            with torch.inference_mode():
                assert torch.allclose(model(probe), probe_logits, atol=1e-6, rtol=1e-5), 'Reload logits'
            checks['checkpoint'] = 'PASS'

            # Replay một bước sau boundary: so với nhánh tham chiếu liên tục, không resume giữa batch.
            current_check = 'resume_replay'
            for replay in (False, True):
                if replay:
                    epoch = load_checkpoint(checkpoint, model, optimizer, scaler, train_loader.generator)
                assert epoch == 1
                set_train_mode(model, strategy)
                images, labels = next(iter(train_loader))
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=amp):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if not replay:
                    torch.save(model.state_dict(), folder / 'replay_reference.pt')
                    reference_loss = float(loss.detach())
                else:
                    reference_state = torch.load(folder / 'replay_reference.pt', mmap=True,
                                                 map_location='cpu', weights_only=True)
                    assert abs(float(loss.detach()) - reference_loss) < 1e-5, 'Resume loss mismatch'
                    assert all(torch.allclose(v, model.state_dict()[k].detach().cpu(), atol=1e-6, rtol=1e-5)
                               for k, v in reference_state.items()), 'Resume state mismatch'
            checks['resume_replay'] = checks['stability'] = 'PASS'
            checks['resources'] = 'PASS'
            summary.update(status='passed', batches=len(durations), train_loss=float(np.mean(losses)),
                           val_loss=float(np.mean(val_losses)), train_batch_seconds=durations,
                           train_peak_allocated_mib=train_peak,
                           workflow_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20 if device.type == 'cuda' else None,
                           resume_tolerance=dict(atol=1e-6, rtol=1e-5), miniature_epoch=1)
            summary['memory_at_end'] = memory_snapshot()
            save_json(folder / 'history.json', [dict(miniature_epoch=1, train_loss=summary['train_loss'],
                                                    val_loss=summary['val_loss'], lr=lr)])
            log.info('Mini-epoch 1 | train loss %.4f | val loss %.4f | mean batch %.3fs | peak train %s MiB',
                     summary['train_loss'], summary['val_loss'], np.mean(durations), train_peak)
            log.info('Checks: %s', checks)
        except Exception as error:
            failed.append(code)
            checks[current_check] = 'FAIL'
            summary.update(status='failed', error=f'{type(error).__name__}: {error}')
            summary.update(failed_check=current_check, batches_completed=len(durations),
                           memory_at_failure=memory_snapshot())
            log.exception('Smoke %s failed; completed checks retained', code)
        finally:
            save_json(folder / 'summary.json', summary)
            for handler in list(log.handlers):
                handler.close()
                log.removeHandler(handler)
            model = optimizer = scaler = train_loader = val_loader = None
            frozen_before = reference_state = None
            head = backbone = trainable = None
            outputs = loss = probe = probe_logits = images = labels = None
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            if summary['status'] == 'failed' and 'RAM/Windows commit' in summary.get('error', ''):
                raise SystemExit('Dừng smoke do thiếu bộ nhớ hệ thống; xem summary.json, không tự chạy tiếp.')
    if failed:
        raise SystemExit(f'Smoke thất bại ở {failed}; xem run.log và summary.json, chưa tuning.')
