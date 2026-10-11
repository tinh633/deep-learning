import logging
from pathlib import Path

import torch
from torch import nn

from data_pipeline import val_loader
from model import model

# None: chọn lần train mới nhất có model.pt; hoặc đổi thành 'lan_2', ...
run_name = None
checkpoint = 'best.pt'  # model.pt để kiểm tra trọng số epoch cuối.
runs_folder = Path(__file__).resolve().parents[1] / 'outputs' / 'runs'
if run_name is None:
    saved_runs = [p for p in runs_folder.glob('lan_*') if (p / checkpoint).is_file()]
    if not saved_runs:
        raise FileNotFoundError(f'No {checkpoint} found. Run src/train.py first.')
    run_folder = max(saved_runs, key=lambda p: int(p.name.split('_')[1]))
else:
    run_folder = runs_folder / run_name

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load(run_folder / checkpoint, map_location=device, weights_only=True))
model.to(device)
model.eval()
criterion = nn.CrossEntropyLoss()
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s',
                    handlers=[logging.FileHandler(run_folder / 'evaluate.log', mode='a'),
                              logging.StreamHandler()], force=True)
logging.info('Validation | run=%s | samples=%d | loss=%s | device=%s',
             run_folder.name, len(val_loader.dataset), criterion, device)
logging.info('Checkpoint: %s', checkpoint)
total_loss, correct = 0.0, 0
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        total_loss += criterion(outputs, labels).item() * len(labels)
        correct += (outputs.argmax(1) == labels).sum().item()
logging.info('Validation loss=%.4f | accuracy=%.2f%% | test not evaluated',
             total_loss / len(val_loader.dataset), 100 * correct / len(val_loader.dataset))
logging.shutdown()
