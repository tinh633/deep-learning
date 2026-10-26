import base64
import gzip
import hashlib
import io
import json
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
from PIL import Image
from torchvision import transforms

from config import DATA, DATA_FOLDER, IMAGENET_MEAN, IMAGENET_STD, OUTPUT_FOLDER
from data import load_datasets, load_split


def image_url(image):
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')


# Raw images remain unchanged; only train/validation guide model decisions.
original, test = load_datasets(download=False)
train_ids, val_ids = load_split(original.targets)
splits = dict(train=(original, train_ids), validation=(original, val_ids),
              test=(test, list(range(len(test)))))
report = dict(created=datetime.now(timezone.utc).isoformat(), seed=DATA['seed'],
              classes=original.classes, splits={}, duplicates=[],
              source='CIFAR-10 official Python batches', pixel_sources={})
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
rgb_sum = np.zeros(3, dtype=np.float64)
rgb_squared_sum = np.zeros(3, dtype=np.float64)
pixel_count = 0
histogram = np.zeros((3, 256), dtype=np.int64)

for split_name, (dataset, ids) in splits.items():
    labels = np.asarray(dataset.targets)
    ids = np.asarray(ids)
    minimum, maximum = 255, 0
    for start in range(0, len(ids), 1000):
        batch_ids = ids[start:start + 1000]
        batch = dataset.data[batch_ids]
        minimum = min(minimum, int(batch.min()))
        maximum = max(maximum, int(batch.max()))
        if split_name == 'train':
            values = batch.astype(np.float64) / 255
            rgb_sum += values.sum(axis=(0, 1, 2))
            rgb_squared_sum += np.square(values).sum(axis=(0, 1, 2))
            pixel_count += len(batch) * batch.shape[1] * batch.shape[2]
            for channel in range(3):
                histogram[channel] += np.bincount(batch[..., channel].ravel(), minlength=256)
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
for digest, members in hash_groups.items():
    if len(members) > 1:
        report['duplicates'].append(dict(sha256=digest, members=members,
                                         cross_split=len({m['split'] for m in members}) > 1))

# Use a train image only; forced flip illustrates the possible augmentation.
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
assert len(train_ids) == 45000 and len(val_ids) == 5000 and len(test) == 10000
assert not set(train_ids) & set(val_ids)
assert all(sum(split['counts']) == split['count'] for split in report['splits'].values())
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
target = OUTPUT_FOLDER / 'eda.json'
target.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False), encoding='utf-8')
print(f'Saved {target}; exact duplicate groups: {len(report["duplicates"])}')
