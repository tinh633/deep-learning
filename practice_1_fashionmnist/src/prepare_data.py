"""Download verified FashionMNIST and prepare EDA for the Live Server viewer."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import struct
import urllib.request

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/'
FILES = {
    'train-images-idx3-ubyte.gz': '8d4fb7e6c68d591d4c3dfef9ec88bf0d',
    'train-labels-idx1-ubyte.gz': '25c81989df183df01b3e8a0aad5dffbe',
    't10k-images-idx3-ubyte.gz': 'bef4ecab320f06d8554ea6380940ec79',
    't10k-labels-idx1-ubyte.gz': 'bb300cfdad3c16e7a12a480ee83cd310',
}
NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def download():
    folder = ROOT / 'data' / 'FashionMNIST_raw'
    folder.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        target = folder / name
        if not target.exists() or hashlib.md5(target.read_bytes()).hexdigest() != expected:
            print(f'Downloading {name}', flush=True)
            request = urllib.request.Request(SOURCE + name, headers={'User-Agent': 'FashionMNIST-Practice-1'})
            with urllib.request.urlopen(request, timeout=120) as response:
                content = response.read()
            if hashlib.md5(content).hexdigest() != expected:
                raise ValueError(f'Checksum mismatch: {name}')
            temporary = target.with_suffix('.part')
            temporary.write_bytes(content)
            temporary.replace(target)
        print(f'MD5 verified: {name}', flush=True)
        # Preserve extracted IDX files for direct loading in subsequent scripts.
        unpacked = target.with_suffix('')
        content = gzip.decompress(target.read_bytes())
        if not unpacked.exists() or unpacked.read_bytes() != content:
            temporary = unpacked.with_suffix('.part')
            temporary.write_bytes(content)
            temporary.replace(unpacked)
    return folder


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download-only', action='store_true')
    args = parser.parse_args()
    folder = download()
    if args.download_only:
        return
    train, train_labels = read_split(folder, 'train', 60000)
    test, test_labels = read_split(folder, 't10k', 10000)
    images = np.concatenate([train, test])
    labels = np.concatenate([train_labels, test_labels])
    print('Computing full-dataset EDA...', flush=True)
    report = {'names': NAMES, 'train': statistics(train, train_labels),
              'test': statistics(test, test_labels), 'all': statistics(images, labels),
              'duplicates': duplicates(images, labels, len(train)), 'checksums': FILES,
              'source': 'https://github.com/zalandoresearch/fashion-mnist'}
    output = ROOT / 'outputs'
    output.mkdir(exist_ok=True)
    (output / 'eda.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    target = ROOT / 'visualization' / 'explorer.html'
    print(f'Data ready for Live Server: {target}', flush=True)
    print(json.dumps({'train': len(train), 'test': len(test), 'duplicates': report['duplicates']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
