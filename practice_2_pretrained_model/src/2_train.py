import os

# Windows workers import this file; avoid multiplying BLAS thread pools.
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import hashlib
import json
import math
import random
import time

import numpy as np
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

from config import (DATA, EXPERIMENT, EXPERIMENTS, MODEL_NAME, OUTPUT_FOLDER,
                    PRETRAINED, SPLIT_FILE, TRAIN)
from data import build_loaders
from logger import create_run, log_epoch, log_setup, open_log
from model import PretrainedCNN


# Windows DataLoader workers import this file in child processes.
if __name__ == '__main__':
    experiment = dict(EXPERIMENTS[EXPERIMENT])
    data_settings = dict(DATA, batch_size=experiment['batch_size'])
    random.seed(TRAIN['seed'])
    np.random.seed(TRAIN['seed'])
    torch.manual_seed(TRAIN['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    folder = create_run(OUTPUT_FOLDER / 'runs')
    logger = open_log(folder)
    writer = None
    started = time.perf_counter()
    completed_epochs = 0
    optimizer_steps = 0
    best_epoch, best_accuracy, best_loss = 0, -1.0, float('inf')
    history = []

    try:
        print(device)
        logger.info('Starting | model=%s | experiment=%s | pretrained=%s | device=%s',
                    MODEL_NAME, EXPERIMENT, PRETRAINED, device)
        writer = SummaryWriter(log_dir=str(folder / 'tensorboard'))
        train_loader, val_loader, _ = build_loaders(data_settings, download=False)
        model = PretrainedCNN(MODEL_NAME, experiment['head'], experiment['fine_tune'], PRETRAINED)
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                               lr=experiment['lr'], weight_decay=TRAIN['weight_decay'])
        settings = dict(model_name=MODEL_NAME, experiment=EXPERIMENT, model_config=experiment,
                        pretrained=PRETRAINED, train=dict(TRAIN), data=data_settings,
                        split_file=str(SPLIT_FILE),
                        split_sha256=hashlib.sha256(SPLIT_FILE.read_bytes()).hexdigest(),
                        source='torchvision.datasets.CIFAR10',
                        input_shape=[3, data_settings['image_size'], data_settings['image_size']])
        log_setup(logger, writer, folder, model, criterion, optimizer,
                  (train_loader, val_loader), settings, device)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)

        for epoch in range(1, TRAIN['epochs'] + 1):
            epoch_started = time.perf_counter()
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
                optimizer_steps += 1

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

            row = dict(epoch=epoch, train_loss=train_loss / len(train_loader.dataset),
                       train_accuracy=train_correct / len(train_loader.dataset),
                       val_loss=val_loss / len(val_loader.dataset),
                       val_accuracy=val_correct / len(val_loader.dataset),
                       lr=optimizer.param_groups[0]['lr'], seconds=time.perf_counter() - epoch_started)
            if not all(math.isfinite(value) for value in row.values()):
                raise ValueError('Non-finite epoch metrics')
            if (row['val_accuracy'] > best_accuracy or
                    (row['val_accuracy'] == best_accuracy and row['val_loss'] < best_loss)):
                best_epoch, best_accuracy, best_loss = epoch, row['val_accuracy'], row['val_loss']
                checkpoint = dict(model_state=model.state_dict(), settings=settings,
                                  weights=model.weights_name, epoch=epoch, metrics=row,
                                  classes=train_loader.dataset.dataset.classes)
                torch.save(checkpoint, folder / 'best.pt.tmp')
                (folder / 'best.pt.tmp').replace(folder / 'best.pt')
            history.append(row)
            (folder / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
            completed_epochs = epoch
            log_epoch(logger, writer, row, best_epoch, TRAIN['epochs'])

        logger.info('Finished | epochs=%d | best_epoch=%d | val_acc=%.2f%% | val_loss=%.6f | checkpoint=%s',
                    completed_epochs, best_epoch, best_accuracy * 100, best_loss, folder / 'best.pt')
    except KeyboardInterrupt:
        logger.warning('Interrupted | completed_epochs=%d | saved history/checkpoint preserved', completed_epochs)
        raise
    except Exception:
        logger.exception('Failed | completed_epochs=%d', completed_epochs)
        raise
    finally:
        peak = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None
        logger.info('Summary | seconds=%.2f | optimizer_steps=%d | completed_epochs=%d | GPU_peak_bytes=%s',
                    time.perf_counter() - started, optimizer_steps, completed_epochs, peak)
        if writer is not None:
            writer.close()
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
