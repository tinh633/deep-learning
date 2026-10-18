import logging

import torch

from config import RUNS_FOLDER


def create_run():
    RUNS_FOLDER.mkdir(parents=True, exist_ok=True)
    number = 1
    while True:
        folder = RUNS_FOLDER / f'lan_{number}'
        try:
            folder.mkdir()
            return folder
        except FileExistsError:
            number += 1


def open_log(folder, filename):
    logger = logging.getLogger(str(folder / filename))
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    for handler in [logging.FileHandler(folder / filename, mode='a', encoding='utf-8'),
                    logging.StreamHandler()]:
        handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))
        logger.addHandler(handler)
    return logger


def log_setup(logger, model, criterion, optimizer, scheduler, settings, device):
    logger.info('Configuration: %s', settings)
    logger.info('Device: %s | GPU: %s | PyTorch: %s', device,
                torch.cuda.get_device_name(device) if device.type == 'cuda' else 'none', torch.__version__)
    logger.info('Model: %s | Parameters: %d', model, sum(p.numel() for p in model.parameters()))
    logger.info('Loss: %s | options=%s', criterion,
                {k: v for k, v in vars(criterion).items() if not k.startswith('_') and k != 'training'})
    logger.info('Optimizer: %s | scheduler: %s', optimizer, scheduler)
    logger.info('Transform: ToTensor | best: maximum val accuracy, tie: minimum val loss | test not evaluated')


def log_epoch(logger, row, best_epoch, total_epochs):
    logger.info(
        'Epoch %d/%d | train_loss=%.6f | train_acc=%.2f%% | '
        'val_loss=%.6f | val_acc=%.2f%% | best_epoch=%d | lr=%g',
        row['epoch'], total_epochs,
        row['train_loss'], row['train_accuracy'] * 100,
        row['val_loss'], row['val_accuracy'] * 100,
        best_epoch, row['lr'],
    )
