import json
import logging
import os
import platform
import sys

import psutil
import torch
import torchvision


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
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    for handler in (logging.FileHandler(folder / filename, encoding='utf-8'),
                    logging.StreamHandler()):
        handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))
        logger.addHandler(handler)
    return logger


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
    (folder / 'config.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
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
