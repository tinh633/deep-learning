import os

# Windows workers import this file; avoid multiplying BLAS thread pools.
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import random
import time

import numpy as np
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

import config
from data import build_loaders
from logger import (close_log, create_run, log_epoch, log_setup,
                    open_log, resume_training, save_epoch)
from model import PretrainedCNN


# Windows DataLoader workers import this file in child processes.
if __name__ == '__main__':
    # 1. Seed, thiết bị và thư mục lưu kết quả.
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if config.RESUME_RUN:
        folder = config.OUTPUT_FOLDER / 'runs' / config.RESUME_RUN
    else:
        folder = create_run(config.OUTPUT_FOLDER / 'runs')
    logger = open_log(folder)
    writer = None
    started = time.perf_counter()
    history = []

    try:
        print(device)
        logger.info('Starting | model=%s | experiment=%s | pretrained=%s | device=%s',
                    config.MODEL_NAME, config.EXPERIMENT, config.PRETRAINED, device)
        writer = SummaryWriter(log_dir=str(folder / 'tensorboard'), flush_secs=2)
        # 2. Dữ liệu -> model -> loss -> optimizer.
        train_loader, val_loader, _ = build_loaders(config.DATA, download=False)
        model = PretrainedCNN(
            name=config.MODEL_NAME,
            head=config.HEAD,
            fine_tune=config.FINE_TUNE,
            pretrained=config.PRETRAINED and config.RESUME_RUN is None,
        )
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = optim.Adam(trainable_parameters, lr=config.LEARNING_RATE,
                               weight_decay=config.WEIGHT_DECAY)
        settings = config.get_settings()

        # 3. Resume nếu có, hoặc ghi cấu hình của lượt mới.
        if config.RESUME_RUN:
            history = resume_training(folder, model, optimizer, train_loader, settings, logger)
        else:
            log_setup(logger, writer, folder, model, criterion, optimizer,
                      (train_loader, val_loader), settings, device)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)

        # 4. Mỗi epoch: train trước, validation sau.
        for epoch in range(len(history) + 1, config.EPOCHS + 1):
            epoch_started = time.perf_counter()
            # Train: tính gradient và cập nhật tham số được phép học.
            model.train()
            train_loss, train_correct = 0.0, 0

            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                if not torch.isfinite(loss):
                    raise ValueError('Non-finite training loss')
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * len(labels)
                train_correct += (outputs.argmax(1) == labels).sum().item()

            # Validation: đo loss/accuracy, không cập nhật trọng số.
            model.eval()
            val_loss, val_correct = 0.0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * len(labels)
                    val_correct += (outputs.argmax(1) == labels).sum().item()

            row = {
                'epoch': epoch,
                'train_loss': train_loss / len(train_loader.dataset),
                'train_accuracy': train_correct / len(train_loader.dataset),
                'val_loss': val_loss / len(val_loader.dataset),
                'val_accuracy': val_correct / len(val_loader.dataset),
                'lr': optimizer.param_groups[0]['lr'],
                'seconds': time.perf_counter() - epoch_started,
            }
            # 5. Lưu best/last và ghi kết quả một lần mỗi epoch.
            best_epoch = save_epoch(folder, model, optimizer, train_loader, settings, history, row)
            log_epoch(logger, writer, row, best_epoch, config.EPOCHS)

        logger.info('Finished | epochs=%d | checkpoint=%s', len(history), folder / 'best.pt')
    except KeyboardInterrupt:
        logger.warning('Interrupted | completed_epochs=%d', len(history))
        raise
    except Exception:
        logger.exception('Failed | completed_epochs=%d', len(history))
        raise
    finally:
        peak = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None
        logger.info('Summary | seconds=%.2f | completed_epochs=%d | GPU_peak_bytes=%s',
                    time.perf_counter() - started, len(history), peak)
        if writer is not None:
            writer.close()
        close_log(logger)
