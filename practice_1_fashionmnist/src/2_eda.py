"""Generate basic EDA plus deeper train/validation analysis for the HTML viewer."""
import json
import gzip
import hashlib
import struct

import numpy as np

from config import DATA_FOLDER, ROOT, FILES, NAMES, DATA
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_split(folder, prefix, expected):
    image_data = gzip.decompress((folder / f'{prefix}-images-idx3-ubyte.gz').read_bytes())
    label_data = gzip.decompress((folder / f'{prefix}-labels-idx1-ubyte.gz').read_bytes())
    if struct.unpack('>IIII', image_data[:16]) != (2051, expected, 28, 28):
        raise ValueError('Invalid image IDX header')
    if struct.unpack('>II', label_data[:8]) != (2049, expected):
        raise ValueError('Invalid label IDX header')
    images = np.frombuffer(image_data, dtype=np.uint8, offset=16).reshape(expected, 28, 28)
    labels = np.frombuffer(label_data, dtype=np.uint8, offset=8)
    if len(labels) != expected or labels.max() > 9:
        raise ValueError('Invalid labels')
    return images, labels


def statistics(images, labels):
    flat = images.reshape(len(images), -1)
    histogram = np.bincount(flat.ravel(), minlength=256)
    values = np.arange(256, dtype=np.float64)
    mean = float(histogram @ values / histogram.sum())
    std = float(np.sqrt(histogram @ ((values - mean) ** 2) / histogram.sum()))
    brightness = flat.mean(axis=1)
    foreground = (flat > 0).mean(axis=1)
    return {
        'count': len(images), 'counts': np.bincount(labels, minlength=10).tolist(),
        'mean': mean, 'std': std, 'normalized_mean': mean / 255, 'normalized_std': std / 255,
        'min': int(flat.min()), 'max': int(flat.max()), 'histogram': histogram.tolist(),
        'zero_ratio': float(histogram[0] / histogram.sum()),
        'saturated_ratio': float(histogram[255] / histogram.sum()),
        'blank': int(np.count_nonzero(flat.max(axis=1) == 0)),
        'constant': int(np.count_nonzero(flat.max(axis=1) == flat.min(axis=1))),
        'brightness_quantiles': np.quantile(brightness, [0, .01, .25, .5, .75, .99, 1]).tolist(),
        'brightness_histogram': np.histogram(brightness, bins=32, range=(0, 256))[0].tolist(),
        'classes': [
            {'mean': float(brightness[labels == c].mean()),
             'foreground': float(foreground[labels == c].mean()),
             'mean_image': np.rint(images[labels == c].mean(axis=0)).astype(np.uint8).ravel().tolist()}
            for c in range(10)
        ],
    }


def duplicates(images, labels, train_count):
    groups = {}
    for index, image in enumerate(images):
        key = hashlib.sha256(image.tobytes()).digest()
        groups.setdefault(key, []).append(index)
    repeated = [indices for indices in groups.values() if len(indices) > 1]
    def within(start, end):
        sizes = [sum(start <= i < end for i in group) for group in repeated]
        return {'groups': sum(s > 1 for s in sizes), 'extra_copies': sum(max(0, s - 1) for s in sizes)}
    return {
        'groups': len(repeated), 'extra_copies': sum(len(g) - 1 for g in repeated),
        'train': within(0, train_count), 'test': within(train_count, len(images)),
        'cross_split_groups': sum(any(i < train_count for i in g) and any(i >= train_count for i in g) for g in repeated),
        'conflicting_label_groups': sum(len(set(int(labels[i]) for i in g)) > 1 for g in repeated),
        'examples': [[{'split': 'train' if i < train_count else 'test', 'index': i if i < train_count else i-train_count, 'label': int(labels[i])} for i in g] for g in repeated[:10]],
    }


train, train_labels = read_split(DATA_FOLDER, 'train', 60000)
test, test_labels = read_split(DATA_FOLDER, 't10k', 10000)
images, labels = np.concatenate([train, test]), np.concatenate([train_labels, test_labels])
report = dict(names=NAMES, train=statistics(train, train_labels),
              test=statistics(test, test_labels), all=statistics(images, labels),
              duplicates=duplicates(images, labels, len(train)), checksums=FILES,
              source='https://github.com/zalandoresearch/fashion-mnist')
output = ROOT / 'outputs'
output.mkdir(exist_ok=True)
(output / 'eda.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print('Basic EDA saved to outputs/eda.json; no HTML changes.')

# Use precisely the same original indices as the training pipeline; test is not used below.
train_ids, val_ids = train_test_split(np.arange(len(train)), test_size=DATA['val_size'],
                                    random_state=DATA['seed'], stratify=train_labels)
flat = train.reshape(len(train), -1)
mask = train > 30
rows, columns = mask.any(2), mask.any(1)
width = 28 - columns.argmax(1) - columns[:, ::-1].argmax(1)
height = 28 - rows.argmax(1) - rows[:, ::-1].argmax(1)
width[~mask.any((1, 2))] = 0
height[~mask.any((1, 2))] = 0
mass = flat.sum(1).clip(1)
features = np.column_stack([
    flat.mean(1), flat.std(1), mask.mean((1, 2)), width, height,
    (train.sum(1) * np.arange(28)).sum(1) / mass,
    (train.sum(2) * np.arange(28)).sum(1) / mass])
keys = ['brightness', 'contrast', 'foreground', 'width', 'height', 'center_x', 'center_y']
plots = ROOT / 'visualization' / 'eda'
plots.mkdir(exist_ok=True)
deep = dict(names=NAMES, seed=DATA['seed'], threshold=30, splits={}, examples=[], near_pairs=[],
            feature_names=keys, fitted_on='training subset only; no test data')
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
for name, ids in [('train', train_ids), ('validation', val_ids)]:
    values = features[ids]
    counts = np.bincount(train_labels[ids], minlength=10)
    deep['splits'][name] = dict(count=len(ids), counts=counts.tolist(),
        quantiles=np.quantile(values, [0, .01, .25, .5, .75, .99, 1], axis=0).tolist(),
        mean=values.mean(0).tolist(), std=values.std(0).tolist(),
        classes=[dict(count=int(counts[c]), mean=values[train_labels[ids] == c].mean(0).tolist()) for c in range(10)])
    for ax, column in zip(axes.flat, range(4)):
        ax.hist(values[:, column], bins=30, density=True, alpha=.45, label=name)
        ax.set(title=keys[column], ylabel='Density')
        ax.legend()
fig.tight_layout()
fig.savefig(plots / 'distributions.png', dpi=140)
plt.close(fig)

fig, axes = plt.subplots(2, 2, figsize=(16, 9))
for ax, column in zip(axes.flat, range(4)):
    ax.boxplot([features[train_ids][train_labels[train_ids] == c, column] for c in range(10)],
               tick_labels=NAMES, showfliers=False)
    ax.set(title=keys[column] + ' by class - train', ylabel=keys[column])
    ax.tick_params(axis='x', rotation=60)
fig.tight_layout()
fig.savefig(plots / 'classes.png', dpi=140)
plt.close(fig)

fig, axes = plt.subplots(2, 10, figsize=(18, 5))
for c in range(10):
    samples = train[train_ids[train_labels[train_ids] == c]]
    axes[0, c].imshow(samples.mean(0), cmap='gray', vmin=0, vmax=255)
    axes[1, c].imshow(samples.std(0), cmap='magma', vmin=0, vmax=128)
    axes[0, c].set_title(NAMES[c], fontsize=8)
    for ax in axes[:, c]:
        ax.set_xticks([])
        ax.set_yticks([])
fig.suptitle('Train: mean image (top), pixel standard deviation (bottom)')
fig.tight_layout()
fig.savefig(plots / 'mean_variation.png', dpi=140)
plt.close(fig)

# Deterministic stratified samples keep PCA/nearest-neighbor computation bounded.
rng = np.random.default_rng(DATA['seed'])
sample_ids = np.concatenate([rng.choice(train_ids[train_labels[train_ids] == c], 500, replace=False) for c in range(10)])
val_sample = np.concatenate([rng.choice(val_ids[train_labels[val_ids] == c], 100, replace=False) for c in range(10)])
pca = PCA(n_components=32, svd_solver='randomized', random_state=DATA['seed'])
projection = pca.fit_transform(flat[sample_ids].astype('float32') / 255)
val_projection = pca.transform(flat[val_sample].astype('float32') / 255)
deep['pca'] = dict(train_samples=len(sample_ids), validation_samples=len(val_sample),
                   components=32, train_indices=sample_ids.tolist(), validation_indices=val_sample.tolist(),
                   explained_variance=pca.explained_variance_ratio_.tolist())
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, points, ids, name in [(axes[0], projection, sample_ids, 'train sample'),
                              (axes[1], val_projection, val_sample, 'validation sample')]:
    for c in range(10):
        selected = train_labels[ids] == c
        ax.scatter(points[selected, 0], points[selected, 1], s=5, alpha=.4, label=NAMES[c])
    ax.set(title=name, xlabel='PC1', ylabel='PC2')
axes[1].legend(fontsize=8, markerscale=2)
fig.tight_layout()
fig.savefig(plots / 'pca.png', dpi=140)
plt.close(fig)

for split, ids in [('train', train_ids), ('validation', val_ids)]:
    for column, direction, title in [(0, 1, 'darkest'), (0, -1, 'brightest'), (1, 1, 'lowest contrast')]:
        chosen = ids[np.argsort(direction * features[ids, column], kind='stable')[:5]]
        for index in chosen:
            deep['examples'].append(dict(split=split, reason=title, index=int(index),
                label=int(train_labels[index]), features=features[index].tolist(), pixels=train[index].ravel().tolist()))
neighbor_ids = rng.choice(sample_ids, 2000, replace=False)
pixels = flat[neighbor_ids].astype('float32') / 255
neighbors = NearestNeighbors(n_neighbors=2).fit(pixels)
distances, indices = neighbors.kneighbors(pixels)
pairs = sorted({tuple(sorted((int(i), int(j)))) for i, j in zip(neighbor_ids, neighbor_ids[indices[:, 1]])},
               key=lambda pair: float(np.mean((flat[pair[0]].astype(float) - flat[pair[1]]) ** 2)))
for a, b in pairs[:8]:
    deep['near_pairs'].append(dict(indices=[a, b], labels=[int(train_labels[a]), int(train_labels[b])],
                                  rmse=float(np.sqrt(np.mean(((flat[a].astype(float) - flat[b]) / 255) ** 2))),
                                  pixels=[train[a].ravel().tolist(), train[b].ravel().tolist()]))
deep['neighbors'] = dict(samples=2000, metric='pixel RMSE on [0,1]', scope='sampled train only')
deep['checks'] = dict(disjoint=not bool(set(train_ids) & set(val_ids)), covered=len(train_ids) + len(val_ids),
                      blank=int((flat.max(1) == 0).sum()), constant=int((flat.max(1) == flat.min(1)).sum()))
(output / 'deep_eda.json').write_text(json.dumps(deep, ensure_ascii=False), encoding='utf-8')
print('Deep EDA saved: outputs/deep_eda.json and visualization/eda/*.png')
