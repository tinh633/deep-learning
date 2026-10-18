"""Export checkpoint scores, mistakes and figures for the HTML viewer."""
import json
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from config import EVALUATE, RUNS_FOLDER, ROOT, NAMES
from data import build_loaders
from metrics import classification_metrics, load_checkpoint

def plot_report(report, run_folder, preview_dataset):
    names = json.loads((ROOT / 'outputs/eda.json').read_text())['names']
    plots = run_folder / 'plots'
    plots.mkdir(exist_ok=True)
    splits = list(report['splits'])
    counts = {}
    fig, axes = plt.subplots(1, len(splits), figsize=(7 * len(splits), 6))
    for ax, split in zip(axes, splits):
        matrix = np.array(report['splits'][split]['confusion'])
        tp = matrix.diagonal()
        fn, fp = matrix.sum(1) - tp, matrix.sum(0) - tp
        counts[split] = dict(TP=tp.tolist(), FN=fn.tolist(), FP=fp.tolist(),
                             TN=(matrix.sum() - tp - fn - fp).tolist())
        ax.imshow(matrix, cmap='Blues')
        for i in range(10):
            for j in range(10):
                ax.text(j, i, matrix[i, j], ha='center', va='center', fontsize=7)
        ax.set(title=split, xlabel='Predicted', ylabel='True')
        ax.set_xticks(range(10), names, rotation=90)
        ax.set_yticks(range(10), names)
    fig.tight_layout()
    fig.savefig(plots / 'confusion.png')
    plt.close(fig)
    (plots / 'counts.json').write_text(json.dumps(counts, indent=2))
    fig, axes = plt.subplots(len(splits), 4, figsize=(18, 4 * len(splits)))
    for i, split in enumerate(splits):
        for j, metric in enumerate(['TP', 'TN', 'FP', 'FN']):
            axes[i, j].bar(range(10), counts[split][metric])
            axes[i, j].set(title=split + ' ' + metric, xlabel='Class ID')
    fig.tight_layout()
    fig.savefig(plots / 'counts.png')
    plt.close(fig)
    for name, keys in [('loss', ['train_loss', 'val_loss']),
                       ('accuracy', ['train_accuracy', 'val_accuracy'])]:
        fig, ax = plt.subplots()
        for key in keys:
            ax.plot([r['epoch'] for r in report['history']],
                    [r[key] for r in report['history']], label=key)
        ax.set(xlabel='Epoch', ylabel=name, title='Train online / validation')
        ax.legend()
        fig.savefig(plots / (name + '.png'))
        plt.close(fig)
    fig, axes = plt.subplots(1, 2)
    for ax, metric in zip(axes, ['loss', 'accuracy']):
        ax.bar(splits, [report['splits'][split][metric] for split in splits])
        ax.set(title='Checkpoint ' + metric)
    fig.tight_layout()
    fig.savefig(plots / 'scores.png')
    plt.close(fig)

    preview = report['splits']['test' if 'test' in report['splits'] else 'validation']
    images = preview_dataset.dataset.data if hasattr(preview_dataset, 'indices') else preview_dataset.data
    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    for row, key in enumerate(['correct_examples', 'mistakes']):
        for ax, item in zip(axes[row], preview[key][:5]):
            ax.imshow(images[item['index']], cmap='gray')
            ax.set_title(f"True: {names[item['true']]}\nPred: {names[item['predicted']]}", fontsize=8)
        for ax in axes[row]:
            ax.axis('off')
    fig.suptitle('Selected checkpoint: correct examples (top), mistakes (bottom)')
    fig.tight_layout()
    fig.savefig(plots / 'predictions.png')
    plt.close(fig)


run_folder = RUNS_FOLDER / EVALUATE['run']
log = (run_folder / 'train.log').read_text()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, metadata = load_checkpoint(run_folder / EVALUATE['checkpoint'], device)
train, val, test = build_loaders(metadata['data_config'])
train_dataset, val_dataset, test_dataset = train.dataset, val.dataset, test.dataset
criterion = nn.CrossEntropyLoss()
report = {'run': EVALUATE['run'], 'checkpoint': EVALUATE['checkpoint'], 'architecture': str(model),
          'training_log': log, 'splits': {}, 'history': []}
report['best_epoch'] = metadata['epoch']
report['history'] = json.loads((run_folder / 'metrics.json').read_text())
splits = [('train', train_dataset), ('validation', val_dataset)]
if EVALUATE['include_test']:
    splits.append(('test', test_dataset))
with torch.no_grad():
    for name, dataset in splits:
        matrix = np.zeros((10, 10), dtype=int)
        mistakes, correct_examples, total_loss, position = [], [], 0.0, 0
        for images, labels in DataLoader(dataset, batch_size=256, shuffle=False):
            outputs = model(images.to(device))
            total_loss += criterion(outputs, labels.to(device)).item() * len(labels)
            predicted = outputs.argmax(1).cpu().numpy()
            confidence = outputs.softmax(1).max(1).values.cpu().numpy()
            for offset, (true, pred) in enumerate(zip(labels.numpy(), predicted)):
                matrix[true, pred] += 1
                if true != pred or len(correct_examples) < 5:
                    index = position + offset
                    original = int(dataset.indices[index]) if hasattr(dataset, 'indices') else index
                    item = dict(index=original, true=int(true), predicted=int(pred),
                                confidence=float(confidence[offset]))
                    (mistakes if true != pred else correct_examples).append(item)
            position += len(labels)
        report['splits'][name] = {'count': len(dataset), 'loss': total_loss / len(dataset),
                                 'accuracy': float(matrix.trace() / len(dataset)),
                                 'confusion': matrix.tolist(), 'mistakes': mistakes,
                                 'correct_examples': correct_examples,
                                 **classification_metrics(matrix)}
        print(name, report['splits'][name]['loss'], report['splits'][name]['accuracy'], 'errors:', len(mistakes))
(run_folder / ('model_report.json' if EVALUATE['include_test'] else 'validation_report.json')).write_text(json.dumps(report, ensure_ascii=False), encoding='utf-8')
(ROOT / 'outputs' / 'model_report.json').write_text(json.dumps(report), encoding='utf-8')
plot_report(report, run_folder, test_dataset if EVALUATE['include_test'] else val_dataset)

# Export class metrics and experiment history without retraining or ranking by test.
rows = [dict(split=split, label=NAMES[i], **row)
        for split, result in report['splits'].items()
        for i, row in enumerate(result['classes'])]
with (run_folder / 'classification.csv').open('w', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

experiments = []
for folder in sorted(RUNS_FOLDER.glob('lan_*'), key=lambda p: int(p.name.split('_')[1])):
    if not (folder / 'best.pt').exists() or not (folder / 'metrics.json').exists():
        continue
    saved = torch.load(folder / 'best.pt', map_location='cpu', weights_only=True)
    history = json.loads((folder / 'metrics.json').read_text())
    minimum = min(history, key=lambda row: row['val_loss'])
    experiments.append(dict(run=folder.name, model=saved['model_config'], train=saved['train_config'],
        hidden_sizes=[int(weight.shape[0]) for key, weight in saved['model_state'].items()
                      if key.endswith('.weight') and weight.ndim == 2][:-1],
        data=saved['data_config'], recorded_epochs=len(history), checkpoint_epoch=saved['epoch'],
        validation=saved['validation'], minimum_val_loss=minimum['val_loss'],
        minimum_loss_epoch=minimum['epoch'], selection=saved.get('selection', 'unknown'),
        completed='Finished' in (folder / 'train.log').read_text(encoding='utf-8')))
(ROOT / 'outputs/experiments.json').write_text(json.dumps(experiments, indent=2), encoding='utf-8')
print('Saved report and plots; test included:', EVALUATE['include_test'])
