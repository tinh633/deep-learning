"""Review saved MLP predictions for runs 10-12, without retraining."""
import csv
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from config import ROOT, NAMES, RUNS_FOLDER, DATA_FOLDER

RUN_NAMES = ('lan_10', 'lan_11', 'lan_12')
OUTPUT = ROOT / 'outputs' / 'mlp_error_analysis'
SPLIT = 'validation'


def save_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def image_array():
    candidates = [DATA_FOLDER / 'train-images-idx3-ubyte.gz',
                  DATA_FOLDER / 'train-images-idx3-ubyte',
                  DATA_FOLDER.parent / 'FashionMNIST' / 'raw' / 'train-images-idx3-ubyte.gz',
                  DATA_FOLDER.parent / 'FashionMNIST' / 'raw' / 'train-images-idx3-ubyte']
    for path in candidates:
        if path.is_file():
            opener = gzip.open if path.suffix == '.gz' else open
            with opener(path, 'rb') as stream:
                raw = stream.read()
            if len(raw) >= 16 and int.from_bytes(raw[:4], 'big') == 2051:
                count = int.from_bytes(raw[4:8], 'big')
                height = int.from_bytes(raw[8:12], 'big')
                width = int.from_bytes(raw[12:16], 'big')
                data = np.frombuffer(raw, dtype=np.uint8, offset=16)
                if data.size == count * height * width:
                    return data.reshape(count, height, width)
    return None


def image_gallery(filename, items, pixels, title):
    if not items:
        return
    columns = 4
    rows = (len(items) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(13, 3.5 * rows))
    for ax in np.asarray(axes).reshape(-1):
        ax.axis('off')
    for ax, item in zip(np.asarray(axes).reshape(-1), items):
        index = item['index']
        if not (0 <= index < len(pixels)):
            continue
        ax.imshow(pixels[index], cmap='gray', vmin=0, vmax=255)
        ax.set_title(f"ID {index} | {item['confidence']:.1%}\n"
                     f"True: {NAMES[item['true']]}\nPred: {NAMES[item['predicted']]}", fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(OUTPUT / filename, dpi=150)
    plt.close(fig)


review_path = ROOT / 'outputs' / 'run_review.json'
if not review_path.is_file():
    raise FileNotFoundError(f'Missing {review_path}. Keep the existing review data in outputs/.')
review = json.loads(review_path.read_text(encoding='utf-8'))
runs = {entry['run']: entry for entry in review['runs']}
OUTPUT.mkdir(parents=True, exist_ok=True)
summary, class_rows, mistake_rows = [], [], []
predictions = {}

for run in RUN_NAMES:
    entry = runs[run]
    checkpoint = RUNS_FOLDER / run / entry['checkpoint']
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Missing checkpoint: {checkpoint}')
    checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if checksum != entry['sha256']:
        raise ValueError(f'Checkpoint hash differs from run_review.json: {run}')
    data = entry['splits'][SPLIT]
    matrix = np.asarray(data['confusion'], dtype=int)
    if matrix.shape != (10, 10) or matrix.sum() != data['count']:
        raise ValueError(f'Invalid confusion matrix: {run}')
    errors = int(matrix.sum() - np.trace(matrix))
    if errors != data['errors'] or errors != len(data['mistakes']):
        raise ValueError(f'Inconsistent mistake count: {run}')
    mistake_ids = [item['index'] for item in data['mistakes']]
    if len(set(mistake_ids)) != len(mistake_ids):
        raise ValueError(f'Duplicate original image ID: {run}')
    predictions[run] = {item['index']: item for item in data['mistakes']}
    summary.append(dict(run=run, checkpoint=entry['checkpoint'], sha256=checksum,
                        split=SPLIT, samples=data['count'], accuracy=data['accuracy'],
                        loss=data['loss'], macro_f1=data['macro_f1'], errors=errors,
                        errors_confidence_90=sum(item['confidence'] >= 0.9 for item in data['mistakes'])))
    for i, metrics in enumerate(data['classes']):
        class_rows.append(dict(run=run, split=SPLIT, class_id=i, class_name=NAMES[i], **metrics))
    for item in data['mistakes']:
        mistake_rows.append(dict(run=run, split=SPLIT, image_id=item['index'],
                                 true=NAMES[item['true']], predicted=NAMES[item['predicted']],
                                 confidence=item['confidence']))
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(matrix, cmap='Blues')
    ax.set_xticks(range(10), NAMES, rotation=60, ha='right')
    ax.set_yticks(range(10), NAMES)
    ax.set(xlabel='Predicted', ylabel='True', title=f'{run} - validation confusion (count)')
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(matrix[i, j]), ha='center', va='center', fontsize=7,
                    color='white' if matrix[i, j] > matrix.max() * .55 else 'black')
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(OUTPUT / f'{run}_confusion.png', dpi=150)
    plt.close(fig)

save_csv(OUTPUT / 'run_summary.csv', summary, list(summary[0]))
save_csv(OUTPUT / 'class_metrics.csv', class_rows, list(class_rows[0]))
save_csv(OUTPUT / 'mistakes.csv', mistake_rows, list(mistake_rows[0]))

comparisons = []
for earlier, later in [('lan_10', 'lan_11'), ('lan_11', 'lan_12'), ('lan_10', 'lan_12')]:
    old, new = set(predictions[earlier]), set(predictions[later])
    fixed = sorted(old - new)
    added = sorted(new - old)
    both_wrong = sorted(old & new)
    comparisons.append(dict(earlier=earlier, later=later, fixed=len(fixed),
                            newly_wrong=len(added), both_wrong=len(both_wrong),
                            both_correct=runs[earlier]['splits'][SPLIT]['count'] - len(old | new),
                            fixed_ids=fixed, newly_wrong_ids=added))
(OUTPUT / 'paired_comparisons.json').write_text(json.dumps(comparisons, indent=2), encoding='utf-8')

fig, ax = plt.subplots(figsize=(9, 4))
positions = np.arange(10)
for n, run in enumerate(RUN_NAMES):
    values = [row['recall'] for row in class_rows if row['run'] == run]
    ax.plot(positions, values, marker='o', label=run)
ax.set_xticks(positions, NAMES, rotation=45, ha='right')
ax.set_ylim(0, 1.05)
ax.set(ylabel='Recall', title='Recall by FashionMNIST class - validation')
ax.grid(alpha=.25)
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT / 'recall_by_class.png', dpi=150)
plt.close(fig)

pixels = image_array()
if pixels is not None:
    for run in RUN_NAMES:
        mistakes = list(predictions[run].values())
        confident = sorted((item for item in mistakes if item['confidence'] >= .9),
                           key=lambda item: -item['confidence'])[:12]
        confusions = [item for item in mistakes if
                      (item['true'], item['predicted']) in {(6, 0), (0, 6), (4, 2), (2, 4)}][:12]
        image_gallery(f'{run}_high_confidence.png', confident, pixels,
                      f'{run} - confident mistakes (validation)')
        image_gallery(f'{run}_class_confusions.png', confusions, pixels,
                      f'{run} - shirt / T-shirt, coat / pullover mistakes')
    for item in comparisons:
        earlier, later = item['earlier'], item['later']
        fixed_items = [predictions[earlier][idx] for idx in item['fixed_ids'][:12]]
        new_items = [predictions[later][idx] for idx in item['newly_wrong_ids'][:12]]
        image_gallery(f'{earlier}_to_{later}_fixed.png', fixed_items, pixels,
                      f'{earlier} wrong -> {later} correct')
        image_gallery(f'{earlier}_to_{later}_new_errors.png', new_items, pixels,
                      f'{earlier} correct -> {later} wrong')
else:
    print('Image galleries skipped: FashionMNIST raw training images not found locally.')

print('Verified best.pt (SHA256), sample counts and validation mistake IDs.')
for row in summary:
    print(f"{row['run']}: accuracy={row['accuracy']:.2%}, mistakes={row['errors']}, "
          f"high-confidence errors={row['errors_confidence_90']}")
for row in comparisons:
    print(f"{row['earlier']} -> {row['later']}: fixed={row['fixed']}, "
          f"new errors={row['newly_wrong']}, both wrong={row['both_wrong']}")
print('Saved analysis to:', OUTPUT)
