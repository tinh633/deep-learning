"""What: Ghi log/config/TensorBoard và lưu/khôi phục trạng thái epoch.
Why: Giữ bằng chứng thí nghiệm và resume mà không làm rối vòng học.
Who: Các script train/evaluate gọi hàm cần thiết trong module này.
When: Đầu run, cuối mỗi epoch và khi resume/kết thúc.
Where: outputs/runs/lan_N chứa config, history, log, best/last và events.
How: Log theo epoch, hash nhận diện file, ghi file tạm rồi replace.
"""

import json
import hashlib
import logging
import math
import os
import platform
import random
import sys

import numpy as np
import psutil
import torch
import torchvision


def resume_training(folder, model, optimizer, train_loader, settings, logger):
    """Restore a completed epoch; return its saved history."""
    saved = torch.load(folder / 'last.pt', map_location='cpu', weights_only=False)
    for key in ('model_name', 'experiment', 'model_config', 'train', 'data', 'split_sha256'):
        if saved['settings'][key] != settings[key]:
            raise ValueError(f'Resume setting changed: {key}')

    history = json.loads((folder / 'history.json').read_text(encoding='utf-8'))
    history = [row for row in history if row['epoch'] <= saved['epoch']]
    if [row['epoch'] for row in history] != list(range(1, saved['epoch'] + 1)):
        raise ValueError('History does not match checkpoint epoch')
    model.load_state_dict(saved['model_state'])
    model.weights_name = saved['weights']
    optimizer.load_state_dict(saved['optimizer_state'])
    random.setstate(saved['random_state'])
    np.random.set_state(saved['numpy_state'])
    torch.set_rng_state(saved['torch_state'])
    train_loader.generator.set_state(saved['loader_state'])
    if next(model.parameters()).is_cuda and saved['cuda_state'] is not None:
        torch.cuda.set_rng_state_all(saved['cuda_state'])
    logger.info('Resuming | epoch=%d | optimizer/random states restored', saved['epoch'])
    save_json(folder / 'history.json', history)
    return history


def save_epoch(folder, model, optimizer, train_loader, settings, history, row):
    """Persist best/last weights and the state needed for a later resume."""
    if not all(math.isfinite(value) for value in row.values()):
        raise ValueError('Non-finite epoch metrics')
    history.append(row)
    best = max(history, key=lambda item: (item['val_accuracy'], -item['val_loss'], -item['epoch']))
    saved = dict(
        model_state=model.state_dict(), settings=settings, weights=model.weights_name,
        epoch=row['epoch'], metrics=row, classes=train_loader.dataset.dataset.classes,
        optimizer_state=optimizer.state_dict(), random_state=random.getstate(),
        numpy_state=np.random.get_state(), torch_state=torch.get_rng_state(),
        cuda_state=torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else None,
        loader_state=train_loader.generator.get_state())
    if best['epoch'] == row['epoch']:
        torch.save(saved, folder / 'best.pt.tmp')
        (folder / 'best.pt.tmp').replace(folder / 'best.pt')
    save_json(folder / 'history.json', history)
    torch.save(saved, folder / 'last.pt.tmp')
    (folder / 'last.pt.tmp').replace(folder / 'last.pt')
    return best['epoch']


def file_sha256(path):
    """Identify a split/checkpoint without loading the whole file into memory."""
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def create_run(runs_folder):
    runs_folder.mkdir(parents=True, exist_ok=True)
    number = 1
    while True:
        folder = runs_folder / f'lan_{number}'
        try:
            folder.mkdir()
            return folder
        except FileExistsError:
            number += 1


def open_log(folder, filename='train.log'):
    logger = logging.getLogger(str((folder / filename).resolve()))
    logger.setLevel(logging.INFO)
    logger.propagate = False
    close_log(logger)
    for handler in (logging.FileHandler(folder / filename, encoding='utf-8'),
                    logging.StreamHandler()):
        handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))
        logger.addHandler(handler)
    return logger


def close_log(logger):
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


def get_environment(device):
    environment = dict(
        os=platform.platform(), python=sys.version,
        cpu=platform.processor() or 'unavailable',
        cpu_physical_cores=psutil.cpu_count(logical=False),
        cpu_logical_cores=psutil.cpu_count(), ram_bytes=psutil.virtual_memory().total,
        torch=str(torch.__version__), torchvision=str(torchvision.__version__),
        cuda_build=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
        device=str(device), gpu=[], precision='float32; AMP disabled',
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG', 'unset'),
        blas_threads={name: os.environ.get(name, 'unset') for name in
                      ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
    )
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            gpu = torch.cuda.get_device_properties(index)
            environment['gpu'].append(dict(index=index, name=gpu.name, vram_bytes=gpu.total_memory))
    return environment


def log_setup(logger, writer, folder, model, criterion, optimizer, loaders, settings, device):
    environment = get_environment(device)
    parameters = dict(total=sum(p.numel() for p in model.parameters()),
                      trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
                      frozen=sum(p.numel() for p in model.parameters() if not p.requires_grad),
                      trainable_names=[n for n, p in model.named_parameters() if p.requires_grad])
    groups = []
    for group in optimizer.param_groups:
        options = {key: value for key, value in group.items() if key != 'params'}
        options['parameter_count'] = sum(p.numel() for p in group['params'])
        groups.append(options)
    data = {}
    for name, loader in zip(('train', 'validation'), loaders):
        data[name] = dict(size=len(loader.dataset), batches=len(loader),
                          batch_size=loader.batch_size, num_workers=loader.num_workers,
                          pin_memory=loader.pin_memory, sampler=type(loader.sampler).__name__,
                          transform=str(loader.dataset.dataset.transform))
    metadata = dict(settings=settings, environment=environment, parameters=parameters,
                    data=data, weights=model.weights_name, feature_size=model.feature_size,
                    optimizer=dict(name=type(optimizer).__name__, groups=groups),
                    loss=dict(name=type(criterion).__name__, reduction=criterion.reduction,
                              ignore_index=criterion.ignore_index,
                              label_smoothing=criterion.label_smoothing), scheduler=None,
                    batchnorm='running statistics frozen; affine trained only in opened blocks')
    save_json(folder / 'config.json', metadata)
    logger.info('Environment: %s', environment)
    logger.info('Configuration: %s', settings)
    logger.info('Data: %s', data)
    logger.info('Model: %s\n%s', model.name, model)
    logger.info('Weights: %s | feature size: %s | Parameters: %s',
                model.weights_name, model.feature_size, parameters)
    logger.info('Loss: %s | Optimizer: %s | Scheduler: none', criterion, optimizer)
    logger.info('BatchNorm statistics frozen | best: val accuracy, then val loss, then earliest epoch')
    logger.info('Test not evaluated | Output: %s', folder)
    writer.add_text('Run/configuration', json.dumps(metadata, indent=2), 0)
    writer.add_text('Model/architecture', str(model), 0)
    writer.flush()


def log_epoch(logger, writer, row, best_epoch, total_epochs):
    logger.info(
        'Epoch %d/%d | train_loss=%.6f | train_acc=%.2f%% | val_loss=%.6f | '
        'val_acc=%.2f%% | lr=%g | seconds=%.2f | best_epoch=%d',
        row['epoch'], total_epochs, row['train_loss'], row['train_accuracy'] * 100,
        row['val_loss'], row['val_accuracy'] * 100, row['lr'], row['seconds'], best_epoch)
    for key in ('train_loss', 'val_loss', 'train_accuracy', 'val_accuracy', 'lr', 'seconds'):
        writer.add_scalar(key, row[key], row['epoch'])
    writer.flush()
