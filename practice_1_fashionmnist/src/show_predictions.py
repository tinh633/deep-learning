"""Draw predicted vs actual test images of a run (default lan_10) from model_report.json."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

run_name = sys.argv[1] if len(sys.argv) > 1 else 'lan_10'  # Dòng lệnh: show_predictions.py [lan_N]
root = Path(__file__).resolve().parents[1]
run_folder = root / 'outputs' / 'runs' / run_name
report = json.loads((run_folder / 'model_report.json').read_text(encoding='utf-8'))
names = json.loads((root / 'outputs' / 'eda.json').read_text(encoding='utf-8'))['names']
raw = root / 'data' / 'FashionMNIST_raw'
images = np.fromfile(raw / 't10k-images-idx3-ubyte', dtype=np.uint8, offset=16).reshape(-1, 28, 28)
labels = np.fromfile(raw / 't10k-labels-idx1-ubyte', dtype=np.uint8, offset=8)

# model_report liệt kê mọi ảnh test bị sai; ảnh không có trong danh sách được dự đoán đúng.
mistakes = {m['index']: m for m in report['splits']['test']['mistakes']}
if len(images) != report['splits']['test']['count'] or len(labels) != len(images):
    raise ValueError('Test data does not match model_report.json')
plots = run_folder / 'plots'
plots.mkdir(exist_ok=True)


def draw(indices, title, filename):
    fig, axes = plt.subplots(4, 5, figsize=(12, 11), layout='constrained')
    for ax, index in zip(axes.ravel(), indices):
        predicted = mistakes[index]['predicted'] if index in mistakes else int(labels[index])
        correct = predicted == labels[index]
        ax.imshow(images[index], cmap='gray')
        ax.set_title(f'#{index}\nPredicted: {names[predicted]}\nActual: {names[labels[index]]}',
                     fontsize=8, color='green' if correct else 'red')
        ax.axis('off')
    fig.suptitle(title)
    fig.savefig(plots / filename, dpi=150)
    plt.close(fig)


first = range(20)
draw(first, f'{run_name}/{report["checkpoint"]} — first 20 test images '
            f'({sum(i in mistakes for i in first)} wrong, red)', 'predictions.png')
confident = sorted(mistakes.values(), key=lambda m: -m['confidence'])[:20]
draw([m['index'] for m in confident],
     f'{run_name}/{report["checkpoint"]} — 20 most confident test mistakes', 'confident_mistakes.png')
print('Saved predictions.png and confident_mistakes.png in', plots)
