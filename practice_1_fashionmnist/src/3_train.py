import json

import torch
from torch import nn, optim

from config import DATA, MODEL, TRAIN
from data import build_loaders
from metrics import evaluate
from model import FashionMLP
from logger import create_run, log_epoch, log_setup, open_log


# 1. Data and model
torch.manual_seed(TRAIN['seed'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader, val_loader, _ = build_loaders()
model = FashionMLP(**MODEL).to(device)

# 2. Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=TRAIN['lr'],
    weight_decay=TRAIN['weight_decay'],
)
scheduler = optim.lr_scheduler.StepLR(
    optimizer, TRAIN['decay_step'], TRAIN['decay_gamma']
)

# 3. Output folder and logging
folder = create_run()
logger = open_log(folder, 'train.log')
settings = dict(
    model_config=dict(MODEL),
    data_config=dict(DATA),
    train_config=dict(TRAIN),
)
log_setup(logger, model, criterion, optimizer, scheduler, settings, device)

best_score = (-1.0, float('-inf'))
best_epoch = 0
history = []

# 4. Training: each epoch goes through all training batches
for epoch in range(1, TRAIN['epochs'] + 1):
    model.train()
    total_loss, correct = 0.0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        correct += (outputs.argmax(1) == labels).sum().item()

    # 5. Validation after all training batches
    val = evaluate(model, val_loader, device)

    row = dict(
        epoch=epoch,
        train_loss=total_loss / len(train_loader.dataset),
        train_accuracy=correct / len(train_loader.dataset),
        val_loss=val['loss'],
        val_accuracy=val['accuracy'],
        lr=optimizer.param_groups[0]['lr'],
    )
    history.append(row)

    # 6. Save the best checkpoint and record this epoch
    checkpoint = dict(
        **settings,
        model_state=model.state_dict(),
        epoch=epoch,
        validation=val,
        selection='val_accuracy_then_val_loss',
        transform='ToTensor',
    )
    score = (val['accuracy'], -val['loss'])

    if score > best_score:
        best_score, best_epoch = score, epoch
        torch.save(checkpoint, folder / 'best.pt')

    log_epoch(logger, row, best_epoch, TRAIN['epochs'])
    scheduler.step()

    (folder / 'metrics.json').write_text(json.dumps(history, indent=2), encoding='utf-8')

# Last epoch weights are saved separately from best.pt
torch.save(checkpoint, folder / 'model.pt')
logger.info('Finished | best_epoch=%d | output=%s', best_epoch, folder)
