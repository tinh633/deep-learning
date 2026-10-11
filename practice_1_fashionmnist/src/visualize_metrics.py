"""Plot lan_10 evaluation and recorded training history."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

run_name = 'lan_10'
root = Path(__file__).resolve().parents[1]
run_folder = root / 'outputs' / 'runs' / run_name
report = json.loads((run_folder / 'model_report.json').read_text(encoding='utf-8'))
names = json.loads((root / 'outputs' / 'eda.json').read_text(encoding='utf-8'))['names']
plots = run_folder / 'plots'
plots.mkdir(exist_ok=True)
splits = ['train', 'validation', 'test']
counts = {}

# Với nhiều lớp, TP/TN/FP/FN được tính cho từng lớp theo one-vs-rest.
fig, axes = plt.subplots(1, 3, figsize=(22, 7))
for ax, split in zip(axes, splits):
    matrix = np.array(report['splits'][split]['confusion'])
    tp = matrix.diagonal()
    fn = matrix.sum(axis=1) - tp
    fp = matrix.sum(axis=0) - tp
    tn = matrix.sum() - tp - fn - fp
    counts[split] = {'TP': tp.tolist(), 'TN': tn.tolist(), 'FP': fp.tolist(), 'FN': fn.tolist()}
    ax.imshow(matrix, cmap='Blues')
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(matrix[i, j]), ha='center', va='center', fontsize=8,
                    color='white' if matrix[i, j] > matrix.max()/2 else 'black')
    ax.set(title=split, xlabel='Predicted label', ylabel='True label')
    ax.set_xticks(range(10), names, rotation=90)
    ax.set_yticks(range(10), names)
fig.suptitle(f'{run_name}/{report["checkpoint"]} — confusion matrices (counts)')
fig.tight_layout()
fig.savefig(plots / 'confusion_matrices.png', dpi=150)
plt.close(fig)

fig, axes = plt.subplots(3, 4, figsize=(22, 13))
for row, split in enumerate(splits):
    for column, metric in enumerate(['TP', 'TN', 'FP', 'FN']):
        ax = axes[row, column]
        ax.bar(range(10), counts[split][metric])
        ax.set(title=f'{split} — {metric}', ylabel='Images')
        ax.set_xticks(range(10), names, rotation=90)
fig.suptitle('One-vs-rest per class; do not sum TN across classes as unique images')
fig.tight_layout()
fig.savefig(plots / 'tp_tn_fp_fn.png', dpi=150)
plt.close(fig)
(plots / 'tp_tn_fp_fn.json').write_text(json.dumps({'classes': names, 'counts': counts}, indent=2), encoding='utf-8')

# So sánh cả ba tập tại cùng checkpoint, không phải đường theo epoch.
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].bar(splits, [report['splits'][s]['loss'] for s in splits])
axes[0].set(title='Loss at selected checkpoint', ylabel='CrossEntropyLoss')
axes[1].bar(splits, [100 * report['splits'][s]['accuracy'] for s in splits])
axes[1].set(title='Accuracy at selected checkpoint', ylabel='Accuracy (%)', ylim=(0, 100))
fig.suptitle(f'{run_name}/{report["checkpoint"]} — best epoch {report["best_epoch"]}')
fig.tight_layout()
fig.savefig(plots / 'split_loss_accuracy.png', dpi=150)
plt.close(fig)

history = report['history']
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
epochs = [row['epoch'] for row in history]
axes[0].plot(epochs, [row['train_loss'] for row in history], label='Train (online)')
axes[0].plot(epochs, [row['val_loss'] for row in history], label='Validation')
axes[0].set(title='Recorded epoch losses', xlabel='Epoch', ylabel='Loss')
axes[1].plot(epochs, [row['val_accuracy'] for row in history], label='Validation')
axes[1].set(title='Recorded epoch accuracy', xlabel='Epoch', ylabel='Accuracy (%)')
for ax in axes:
    ax.axvline(report['best_epoch'], color='gray', linestyle='--', label='Best checkpoint')
    ax.legend()
    ax.grid(alpha=0.2)
fig.suptitle('No train accuracy or per-epoch test metrics in original log; not fabricated')
fig.tight_layout()
fig.savefig(plots / 'epoch_curves.png', dpi=150)
plt.close(fig)
print('Saved 4 plots and per-class TP/TN/FP/FN:', plots)
