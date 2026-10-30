"""What: Khảo sát ảnh, phân bố lớp, RGB, ảnh trùng và preprocessing.
Why: Hiểu dữ liệu trước train và tạo bằng chứng cho phần trình bày.
Who: Người học chạy script, notebook/HTML đọc report để xem.
When: Sau khi tải dataset, trước các lượt train chính thức.
Where: Đọc data/ và split; ghi outputs/eda.json, data/CIFAR10_web/.
How: Thống kê theo split, RGB chỉ từ train, hash pixel và xuất RGB gzip.
"""

import os

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import base64
import gzip
import hashlib
import io
import json
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from torchvision import transforms

from config import DATA, DATA_FOLDER, IMAGENET_MEAN, IMAGENET_STD, OUTPUT_FOLDER
from data import load_datasets, load_split


def image_url(image):
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')


# 1. Load original data and the fixed split.
original, test = load_datasets(download=False)
train_ids, val_ids = load_split(original.targets)
splits = dict(train=(original, train_ids), validation=(original, val_ids),
              test=(test, list(range(len(test)))))
report = dict(created=datetime.now(timezone.utc).isoformat(), seed=DATA['seed'],
              classes=original.classes, splits={}, duplicates=[],
              source='CIFAR-10 official Python batches', pixel_sources={})

# 2. Export compressed RGB chunks for the HTML viewer.
hash_groups = defaultdict(list)
pixel_folder = DATA_FOLDER / 'CIFAR10_web'
pixel_folder.mkdir(parents=True, exist_ok=True)
# Browser-readable RGB bytes, not PNG atlases or Python pickle batches.
for source_name, dataset in (('train', original), ('test', test)):
    chunks = []
    for start in range(0, len(dataset), 10000):
        pixels = dataset.data[start:start + 10000].tobytes()
        filename = f'{source_name}_{start // 10000}.rgb.gz'
        with (pixel_folder / filename).open('wb') as output:
            with gzip.GzipFile(filename='', mode='wb', fileobj=output, mtime=0) as compressed:
                compressed.write(pixels)
        chunks.append(dict(file=f'../data/CIFAR10_web/{filename}', start=start,
                           count=len(pixels) // 3072, sha256=hashlib.sha256(pixels).hexdigest()))
    report['pixel_sources'][source_name] = chunks

# 3. Class counts, pixel statistics and duplicate detection.
rgb_sum = np.zeros(3, dtype=np.float64)
rgb_squared_sum = np.zeros(3, dtype=np.float64)
pixel_count = 0
histogram = np.zeros((3, 256), dtype=np.int64)
feature_rows = {name: [] for name in splits}
projection_rows = {name: [] for name in splits}
class_rgb_sum = np.zeros((10, 3), dtype=np.float64)
class_pixel_count = np.zeros(10, dtype=np.int64)

for split_name, (dataset, ids) in splits.items():
    labels = np.asarray(dataset.targets)
    ids = np.asarray(ids)
    minimum, maximum = 255, 0
    for start in range(0, len(ids), 1000):
        batch_ids = ids[start:start + 1000]
        batch = dataset.data[batch_ids]
        minimum = min(minimum, int(batch.min()))
        maximum = max(maximum, int(batch.max()))
        # Per-image descriptors on original pixels, before augmentation/resize.
        pixels = batch.astype(np.float32) / 255
        # 8x8 RGB block averages retain coarse appearance without a model or image export.
        thumbnail = pixels.reshape(len(batch), 8, 4, 8, 4, 3).mean(axis=(2, 4))
        projection_rows[split_name].append(thumbnail.reshape(len(batch), -1))
        gray = pixels.mean(axis=3)
        high, low = pixels.max(axis=3), pixels.min(axis=3)
        saturation = np.divide(high - low, high, out=np.zeros_like(high), where=high > 0)
        edge = (np.abs(np.diff(gray, axis=1)).mean(axis=(1, 2)) +
                np.abs(np.diff(gray, axis=2)).mean(axis=(1, 2))) / 2
        feature_rows[split_name].append(np.column_stack([
            gray.mean(axis=(1, 2)), gray.std(axis=(1, 2)), saturation.mean(axis=(1, 2)), edge]))
        if split_name == 'train':
            values = batch.astype(np.float64) / 255
            rgb_sum += values.sum(axis=(0, 1, 2))
            rgb_squared_sum += np.square(values).sum(axis=(0, 1, 2))
            pixel_count += len(batch) * batch.shape[1] * batch.shape[2]
            for channel in range(3):
                histogram[channel] += np.bincount(batch[..., channel].ravel(), minlength=256)
            for label in range(10):
                selected = batch[labels[batch_ids] == label]
                class_rgb_sum[label] += selected.sum(axis=(0, 1, 2), dtype=np.float64) / 255
                class_pixel_count[label] += len(selected) * 1024
        for index, image in zip(batch_ids, batch):
            digest = hashlib.sha256(image.tobytes()).hexdigest()
            hash_groups[digest].append(dict(split=split_name, id=int(index), label=int(labels[index])))
    report['splits'][split_name] = dict(
        count=len(ids), counts=np.bincount(labels[ids], minlength=10).tolist(),
        shape=list(dataset.data.shape[1:]), dtype=str(dataset.data.dtype),
        pixel_range=[minimum, maximum])
    gallery_ids = np.sort(ids)
    report['splits'][split_name]['gallery'] = dict(
        ids=gallery_ids.tolist(), labels=labels[gallery_ids].tolist(),
        source='test' if split_name == 'test' else 'train')
    print(f'{split_name}: {len(ids)} images; class counts={report["splits"][split_name]["counts"]}')

mean = rgb_sum / pixel_count
std = np.sqrt(np.maximum(0, rgb_squared_sum / pixel_count - mean ** 2))
report['train_rgb'] = dict(mean=mean.tolist(), std=std.tolist(), histogram=histogram.tolist())
# Descriptive split comparisons; no fitting or preprocessing decisions from test.
feature_names = ['brightness', 'contrast', 'saturation', 'edge']
features = {name: np.concatenate(rows) for name, rows in feature_rows.items()}
train_features = features['train']
deep = dict(feature_names=feature_names, features={}, per_class=[], extremes={},
            correlation=np.corrcoef(train_features, rowvar=False).tolist(),
            method='Original RGB/255; brightness=mean RGB; contrast=std of mean-channel gray; '
                   'saturation=mean HSV S; edge=mean absolute adjacent gray differences. '
                   'Population std; train-only percentiles and example selection; descriptive split comparison.')
for column, name in enumerate(feature_names):
    bins = np.linspace(0, 1 if name in ('brightness', 'saturation') else 0.5, 31)
    # Include every observed descriptor in the histogram.
    bins[-1] = max(bins[-1], max(float(values[:, column].max()) for values in features.values()) + 1e-6)
    train_mean = float(train_features[:, column].mean())
    train_std = float(train_features[:, column].std())
    deep['features'][name] = dict(bins=bins.tolist(), splits={})
    for split_name, values in features.items():
        data = values[:, column]
        counts = np.histogram(data, bins=bins)[0]
        assert counts.sum() == len(data)
        deep['features'][name]['splits'][split_name] = dict(
            count=len(data), mean=float(data.mean()), std=float(data.std()),
            quantiles=np.quantile(data, [0, .01, .25, .5, .75, .99, 1]).tolist(),
            histogram=counts.tolist(),
            mean_shift_train_std=float((data.mean() - train_mean) / train_std) if train_std else None)
    order = np.argsort(train_features[:, column], kind='stable')
    deep['extremes'][name] = dict(
        low=[dict(id=int(train_ids[i]), label=int(original.targets[train_ids[i]]),
                  value=float(train_features[i, column])) for i in order[:4]],
        high=[dict(id=int(train_ids[i]), label=int(original.targets[train_ids[i]]),
                   value=float(train_features[i, column])) for i in order[-4:][::-1]])
train_labels = np.asarray(original.targets)[train_ids]
for label in range(10):
    values = train_features[train_labels == label]
    deep['per_class'].append(dict(label=label, count=len(values),
        rgb_mean=(class_rgb_sum[label] / class_pixel_count[label]).tolist(),
        mean=values.mean(axis=0).tolist(), std=values.std(axis=0).tolist()))
report['deep_eda'] = deep
# One common dataset map: fit PCA only on train, project every original image.
projection_data = {name: np.concatenate(rows) for name, rows in projection_rows.items()}
pca = PCA(n_components=2, svd_solver='randomized', random_state=DATA['seed'])
pca.fit(projection_data['train'])
report['dataset_projection'] = dict(
    method='PCA on 8x8 RGB block means /255; centered, no per-feature scaling',
    fit_split='train', seed=DATA['seed'], feature_shape=[8, 8, 3],
    explained_variance=pca.explained_variance_ratio_.tolist(), coordinates={})
for split_name, (_, ids) in splits.items():
    coordinates = pca.transform(projection_data[split_name])[np.argsort(ids)]
    assert coordinates.shape == (len(ids), 2) and np.isfinite(coordinates).all()
    # Coordinates follow the sorted gallery IDs/labels already stored in this report.
    report['dataset_projection']['coordinates'][split_name] = coordinates.round(5).tolist()
for digest, members in hash_groups.items():
    if len(members) > 1:
        report['duplicates'].append(dict(sha256=digest, members=members,
                                         cross_split=len({m['split'] for m in members}) > 1))

# 4. Illustrate preprocessing with a train image only; forced flip illustrates the possible augmentation.
index = int(train_ids[0])
image = Image.fromarray(original.data[index])
resize = transforms.Resize((DATA['image_size'], DATA['image_size']),
                           interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)
resized = resize(image)
flipped = transforms.functional.hflip(resized)
tensor = transforms.ToTensor()(flipped)
normalized = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)(tensor)
restored = normalized.clone()
for channel, mean_value, std_value in zip(restored, IMAGENET_MEAN, IMAGENET_STD):
    channel.mul_(std_value).add_(mean_value)
report['preprocessing'] = dict(
    split='train', id=index, label=int(original.targets[index]),
    original=image_url(image), resized=image_url(resized), flipped=image_url(flipped),
    normalized_min=float(normalized.min()), normalized_max=float(normalized.max()),
    normalized_channel_mean=normalized.mean(dim=(1, 2)).tolist(),
    restored=image_url(transforms.ToPILImage()(restored.clamp(0, 1))))
# 5. Validate counts and save the EDA report.
assert len(train_ids) == 45000 and len(val_ids) == 5000 and len(test) == 10000
assert not set(train_ids) & set(val_ids)
assert all(sum(split['counts']) == split['count'] for split in report['splits'].values())
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
target = OUTPUT_FOLDER / 'eda.json'
target.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False), encoding='utf-8')
print(f'Saved {target}; exact duplicate groups: {len(report["duplicates"])}')
