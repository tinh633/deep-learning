"""Evaluate best.pt of a run (default lan_10) and export data for the HTML viewer."""
import json
from pathlib import Path
import re
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from data_pipeline import train_dataset, val_dataset, test_dataset
from model import model

run_name = sys.argv[1] if len(sys.argv) > 1 else 'lan_10'  # Dòng lệnh: model_report.py [lan_N]
run_folder = Path(__file__).resolve().parents[1] / 'outputs' / 'runs' / run_name
log = (run_folder / 'train.log').read_text()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load(run_folder / 'best.pt', map_location=device, weights_only=True))
model.to(device).eval()
criterion = nn.CrossEntropyLoss()
report = {'run': run_name, 'checkpoint': 'best.pt', 'architecture': str(model),
          'training_log': log, 'splits': {}, 'history': []}
for epoch, loss, val_loss, accuracy, best in re.findall(
        r'Epoch (\d+)/\d+ \| train_loss=([\d.]+) \| val_loss=([\d.]+) \| val_acc=([\d.]+)% \| best_epoch=(\d+)', log):
    report['history'].append({'epoch': int(epoch), 'train_loss': float(loss),
                              'val_loss': float(val_loss), 'val_accuracy': float(accuracy), 'best_epoch': int(best)})
report['best_epoch'] = report['history'][-1]['best_epoch'] if report['history'] else None
with torch.no_grad():
    for name, dataset in [('train', train_dataset), ('validation', val_dataset), ('test', test_dataset)]:
        matrix = np.zeros((10, 10), dtype=int)
        mistakes, total_loss, position = [], 0.0, 0
        for images, labels in DataLoader(dataset, batch_size=256, shuffle=False):
            outputs = model(images.to(device))
            total_loss += criterion(outputs, labels.to(device)).item() * len(labels)
            predicted = outputs.argmax(1).cpu().numpy()
            confidence = outputs.softmax(1).max(1).values.cpu().numpy()
            for offset, (true, pred) in enumerate(zip(labels.numpy(), predicted)):
                matrix[true, pred] += 1
                if true != pred:
                    index = position + offset
                    original = int(dataset.indices[index]) if hasattr(dataset, 'indices') else index
                    mistakes.append({'index': original, 'true': int(true), 'predicted': int(pred),
                                     'confidence': float(confidence[offset])})
            position += len(labels)
        report['splits'][name] = {'count': len(dataset), 'loss': total_loss / len(dataset),
                                 'accuracy': float(matrix.trace() / len(dataset)),
                                 'confusion': matrix.tolist(), 'mistakes': mistakes}
        print(name, report['splits'][name]['loss'], report['splits'][name]['accuracy'], 'errors:', len(mistakes))
(run_folder / 'model_report.json').write_text(json.dumps(report, ensure_ascii=False), encoding='utf-8')
print('Saved model_report.json; no training performed.')
