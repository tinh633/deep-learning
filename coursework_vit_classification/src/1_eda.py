"""What: Tải và khảo sát CIFAR-100. Why: Hiểu dữ liệu trước huấn luyện.
Who: Người thực hiện coursework, HTML đọc report. When: Trước train.
Where: data/CIFAR100_web và outputs/eda.json. How: NumPy thống kê, gzip cho viewer.
"""
import gzip
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from threadpoolctl import threadpool_limits

from config import DATA_DIR, OUTPUT_DIR, SEED, SPLIT_FILE
from data import load_datasets, load_split

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

print('Bước 1/5: tải/kiểm tra CIFAR-100 (tải lần đầu; các lần sau dùng dữ liệu local).', flush=True)
original, test = load_datasets(download=True)
train_ids, val_ids = load_split(original.targets)
splits = {'train': (original, train_ids), 'validation': (original, val_ids),
          'test': (test, list(range(len(test))))}
report = {'created': datetime.now(timezone.utc).isoformat(), 'seed': SEED,
          'classes': original.classes, 'source': 'https://www.cs.toronto.edu/~kriz/cifar.html',
          'split_sha256': hashlib.sha256(SPLIT_FILE.read_bytes()).hexdigest(),
          'splits': {}, 'pixel_sources': {}}
web_dir = DATA_DIR / 'CIFAR100_web'
web_dir.mkdir(parents=True, exist_ok=True)

print('Bước 2/5: xuất RGB gzip local để duyệt toàn bộ 60.000 ảnh.', flush=True)
for source, dataset in (('train', original), ('test', test)):
    chunks = []
    for start in range(0, len(dataset), 10000):
        name = f'{source}_{start // 10000}.rgb.gz'
        with gzip.open(web_dir / name, 'wb') as output:
            output.write(dataset.data[start:start + 10000].tobytes())
        chunks.append({'file': f'../data/CIFAR100_web/{name}', 'start': start,
                       'count': min(10000, len(dataset) - start)})
    report['pixel_sources'][source] = chunks

print('Bước 3/5: thống kê split, RGB train, độ sáng/độ tương phản và ảnh trùng.', flush=True)
hashes = defaultdict(list)
for split, (dataset, ids) in splits.items():
    pixels = dataset.data[ids]
    labels = np.asarray(dataset.targets)[ids]
    brightness = pixels.mean(axis=(1, 2, 3)) / 255
    contrast = pixels.std(axis=(1, 2, 3)) / 255
    report['splits'][split] = {
        'count': len(ids), 'source': 'test' if split == 'test' else 'train',
        'ids': ids, 'labels': labels.tolist(),
        'class_counts': np.bincount(labels, minlength=100).tolist(),
        'brightness': np.round(brightness, 4).tolist(),
        'contrast': np.round(contrast, 4).tolist(),
        'brightness_quantiles': np.quantile(brightness, [0, .25, .5, .75, 1]).tolist(),
        'contrast_quantiles': np.quantile(contrast, [0, .25, .5, .75, 1]).tolist()}
    if split == 'train':
        flat = pixels.reshape(-1, 3)
        report['train_rgb'] = {
            'mean': (flat.mean(axis=0) / 255).tolist(),
            'std': (flat.std(axis=0) / 255).tolist(),
            'histogram': [np.bincount(flat[:, c], minlength=256).tolist() for c in range(3)]}
        report['train_class_brightness'] = [float(brightness[labels == c].mean()) for c in range(100)]
    for local, source_id in enumerate(ids):
        digest = hashlib.sha256(pixels[local].tobytes()).hexdigest()
        hashes[digest].append({'split': split, 'id': source_id, 'label': int(labels[local])})
report['exact_duplicates'] = [items for items in hashes.values() if len(items) > 1]
report['cross_split_duplicate_groups'] = sum(
    len({item['split'] for item in group}) > 1 for group in report['exact_duplicates'])

print('Bước 4/5: PCA 60.000 ảnh và t-SNE 5.000 ảnh train (50 ảnh/lớp).', flush=True)
# Average pooling 32x32 -> 8x8 RGB giữ bố cục màu, không dùng nhãn làm feature.
features = {}
for source, dataset in (('train', original), ('test', test)):
    features[source] = dataset.data.reshape(-1, 8, 4, 8, 4, 3).mean(axis=(2, 4))
    features[source] = features[source].reshape(-1, 192).astype(np.float32) / 255
with threadpool_limits(limits=4):
    pca = PCA(n_components=32, svd_solver='randomized', random_state=SEED)
    pca.fit(features['train'][train_ids])
    reduced = {source: pca.transform(values) for source, values in features.items()}
    rng = np.random.default_rng(SEED)
    train_labels = np.asarray(original.targets)
    sample_ids = []
    for label in range(100):
        candidates = np.asarray(train_ids)[train_labels[train_ids] == label]
        sample_ids.extend(rng.choice(candidates, size=50, replace=False).tolist())
    sample_ids = sorted(sample_ids)
    tsne = TSNE(n_components=2, perplexity=30, max_iter=750, init='pca',
                learning_rate='auto', random_state=SEED, n_jobs=4)
    tsne_xy = tsne.fit_transform(reduced['train'][sample_ids])
projection = {
    'seed': SEED, 'features': '8x8 RGB block means (192 dimensions); no learned features',
    'pca_fit_split': 'train', 'pca_fit_count': len(train_ids),
    'pca_explained_variance_2d': float(pca.explained_variance_ratio_[:2].sum()),
    'pca': {split: np.round(reduced[info['source']][info['ids'], :2], 5).tolist()
            for split, info in report['splits'].items()},
    'tsne': {'split': 'train', 'ids': sample_ids,
             'labels': train_labels[sample_ids].tolist(),
             'xy': np.round(tsne_xy, 5).tolist(), 'per_class': 50,
             'perplexity': 30, 'max_iter': 750, 'pca_dimensions': 32,
             'kl_divergence': float(tsne.kl_divergence_)}}

print('Bước 5/5: ghi report và tọa độ (không nhúng pixel ảnh vào JSON).', flush=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / 'projection.json').write_text(json.dumps(projection), encoding='utf-8')
path = OUTPUT_DIR / 'eda.json'
path.write_text(json.dumps(report, ensure_ascii=False), encoding='utf-8')
print(f'Hoàn tất: {path}\nMở visualization/explorer.html qua HTTP server.', flush=True)
