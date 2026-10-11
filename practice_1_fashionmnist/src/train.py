import logging
from pathlib import Path

import torch
from torch import nn, optim

torch.manual_seed(42)
from data_pipeline import train_loader, val_loader, transform
from model import model

num_epochs = 30
learning_rate = 0.001
decay_step = 10
decay_gamma = 0.3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=decay_step, gamma=decay_gamma)

# Lưu log riêng cho mỗi lần chạy, giữ nguyên log cũ.
runs_folder = Path(__file__).resolve().parents[1] / 'outputs' / 'runs'
runs_folder.mkdir(parents=True, exist_ok=True)
run_number = 1
while True:
    run_folder = runs_folder / f'lan_{run_number}'
    try:
        run_folder.mkdir()
        break
    except FileExistsError:
        run_number += 1
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s',
                    handlers=[logging.FileHandler(run_folder / 'train.log'), logging.StreamHandler()], force=True)
logging.info('Run: %s | epochs=%d | seed=%d', run_folder.name, num_epochs, torch.initial_seed())
logging.info('Device: %s | GPU: %s | PyTorch: %s', device,
             torch.cuda.get_device_name(device) if device.type == 'cuda' else 'none', torch.__version__)
logging.info('Data: train=%d | batch_size=%d | batches=%d | shuffle=%s | transform=%s',
             len(train_loader.dataset), train_loader.batch_size, len(train_loader),
             type(train_loader.sampler).__name__, transform)
logging.info('Model: %s\nParameters: %d', model, sum(p.numel() for p in model.parameters()))
logging.info('Loss: %s | options=%s', type(criterion).__name__,
             {key: value for key, value in vars(criterion).items() if not key.startswith('_') and key != 'training'})
logging.info('Optimizer: %s', type(optimizer).__name__)
logging.info('LR decay: StepLR | step_size=%d | gamma=%s', decay_step, decay_gamma)
for group_number, group in enumerate(optimizer.param_groups, 1):
    logging.info('Optimizer group %d: %s', group_number,
                 {key: value for key, value in group.items() if key != 'params'})
logging.info('Start training | best checkpoint: minimum validation loss | test not evaluated')
best_val_loss = float('inf')
best_epoch = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * len(labels)
    model.eval()
    val_loss, correct = 0.0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_loss += criterion(outputs, labels).item() * len(labels)
            correct += (outputs.argmax(1) == labels).sum().item()
    val_loss /= len(val_loader.dataset)
    if val_loss < best_val_loss:
        best_val_loss, best_epoch = val_loss, epoch + 1
        torch.save(model.state_dict(), run_folder / 'best.pt')
    logging.info('Epoch %d/%d | train_loss=%.4f | val_loss=%.6f | val_acc=%.2f%% | best_epoch=%d | lr=%s',
                 epoch + 1, num_epochs, running_loss / len(train_loader.dataset),
                 val_loss, 100 * correct / len(val_loader.dataset), best_epoch,
                 [group['lr'] for group in optimizer.param_groups])
    scheduler.step()  # Giảm lr sau khi hoàn tất epoch; log ghi lr đã dùng.

torch.save(model.state_dict(), run_folder / 'model.pt')
logging.info('Finished training | model saved: %s', run_folder / 'model.pt')
logging.info('Best checkpoint: best.pt | epoch=%d | val_loss=%.6f', best_epoch, best_val_loss)
logging.shutdown()
