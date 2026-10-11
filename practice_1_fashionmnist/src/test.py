import logging
from pathlib import Path

import torch
from torch import nn

from data_pipeline import val_loader, test_loader
from model import model

run_name = 'lan_10'  # Chọn run muốn kiểm tra; chưa chốt mô hình cuối.
checkpoint = 'best.pt'
run_folder = Path(__file__).resolve().parents[1] / 'outputs' / 'runs' / run_name
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load(run_folder / checkpoint, map_location=device, weights_only=True))
model.to(device)
model.eval()
criterion = nn.CrossEntropyLoss()

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s',
                    handlers=[logging.FileHandler(run_folder / 'test.log', mode='a'),
                              logging.StreamHandler()], force=True)
logging.info('Run=%s | checkpoint=%s | device=%s | loss=%s', run_name, checkpoint, device, criterion)
results = {}
with torch.no_grad():
    for name, loader in [('Validation', val_loader), ('Test', test_loader)]:
        total_loss, correct = 0.0, 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item() * len(labels)
            correct += (outputs.argmax(1) == labels).sum().item()
        loss = total_loss / len(loader.dataset)
        accuracy = 100 * correct / len(loader.dataset)
        results[name] = (loss, accuracy)
        logging.info('%s | samples=%d | loss=%.4f | accuracy=%.2f%%',
                     name, len(loader.dataset), loss, accuracy)
logging.info('Test - Validation | loss=%+.4f | accuracy=%+.2f percentage points',
             results['Test'][0] - results['Validation'][0],
             results['Test'][1] - results['Validation'][1])
logging.shutdown()
