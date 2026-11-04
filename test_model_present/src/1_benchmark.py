"""What: Evaluate ImageNet-1K classifiers and measure inference resources.
Why: Report accuracy and actual local latency/throughput/VRAM together.
Who: Explicitly started by the presenter; notebook only reads saved reports.
When: After a labeled ImageNet manifest has been prepared.
Where: data/manifest.csv -> unique outputs/lan_N directory.
How: A separate process per model, eval/inference_mode and synchronized timing.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import csv
import hashlib
import json
import platform
import shutil
from datetime import datetime, timezone
from urllib.parse import urlparse
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import config


def sha256(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def save_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def read_manifest():
    if not config.MANIFEST.is_file():
        raise FileNotFoundError(f'Missing labeled ImageNet manifest: {config.MANIFEST}. See README.md.')
    with config.MANIFEST.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not {'id', 'path', 'label', 'split'} <= set(reader.fieldnames or []):
            raise ValueError('Manifest needs id,path,label,split columns')
        rows = list(reader)
    seen_ids, seen_paths = set(), set()
    for row in rows:
        row['label'] = int(row['label'])
        path = (config.MANIFEST.parent / row['path']).resolve()
        if (not path.is_file() or not 0 <= row['label'] < 1000 or
                row['split'] != config.SPLIT or row['id'] in seen_ids or str(path) in seen_paths):
            raise ValueError(f'Invalid/duplicate image or label/split: {row["id"]}')
        if row.get('image_sha256') and sha256(path) != row['image_sha256']:
            raise ValueError(f'Image bytes changed: {row["id"]}')
        seen_ids.add(row['id']); seen_paths.add(str(path))
        row['path'] = str(path)
    if not rows:
        raise ValueError('Manifest is empty')
    counts = {label: sum(row['label'] == label for row in rows) for label in config.SELECTED_CLASSES}
    if (set(row['label'] for row in rows) != set(config.SELECTED_CLASSES) or
            any(count != config.IMAGES_PER_CLASS for count in counts.values())):
        raise ValueError('Expected exactly 10 selected classes, 50 images each; rerun 0_preparedata.py')
    if config.MAX_IMAGES and len(rows) > config.MAX_IMAGES:
        rows = random.Random(config.SEED).sample(rows, config.MAX_IMAGES)
    return rows


class LabeledImages(Dataset):
    def __init__(self, rows, transform):
        self.rows, self.transform = rows, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row['path']) as image:
            tensor = self.transform(image.convert('RGB'))
        return tensor, row['label']


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def measure(model, inputs, device):
    for _ in range(config.WARMUP):
        model(inputs)
    synchronize(device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    repeats, samples = [], []
    for _ in range(config.REPEATS):
        durations = []
        for _ in range(config.TIMING_REQUESTS):
            synchronize(device)
            start = time.perf_counter()
            model(inputs)
            synchronize(device)
            durations.append(time.perf_counter() - start)
        samples.extend(durations)
        repeats.append(float(np.mean(durations)))
    return dict(batch=len(inputs), requests=len(samples),
                mean_ms=float(np.mean(samples) * 1000),
                median_ms=float(np.median(samples) * 1000),
                p95_ms=float(np.percentile(samples, 95) * 1000),
                request_ms=[value * 1000 for value in samples],
                repeat_mean_ms=[value * 1000 for value in repeats],
                repeat_std_ms=float(np.std(repeats) * 1000),
                images_per_second=float(len(inputs) / np.mean(samples)),
                peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else None,
                peak_reserved_mib=torch.cuda.max_memory_reserved(device) / 2**20 if device.type == 'cuda' else None)


def benchmark(name, folder):
    runtime = json.loads((folder / 'run_settings.json').read_text())
    for key in ('WARMUP', 'REPEATS', 'TIMING_REQUESTS', 'BATCH_SIZE'):
        setattr(config, key, runtime[key])
    torch.manual_seed(config.SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Cache stays local to this task, never inside Git-tracked model artifacts.
    original_cache = Path(torch.hub.get_dir()) / 'checkpoints'
    torch.hub.set_dir(str(config.ROOT / 'data' / 'torch_cache'))
    rows = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    weights = models.get_weight(config.MODELS[name])
    transform = weights.transforms()
    weight_path = Path(torch.hub.get_dir()) / 'checkpoints' / Path(urlparse(weights.url).path).name
    cached = original_cache / weight_path.name
    prefix = weight_path.stem.rsplit('-', 1)[-1]
    if not weight_path.exists() and cached.is_file() and sha256(cached).startswith(prefix):
        weight_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached, weight_path)
    print(f'{name}: loading pretrained {config.MODELS[name]}', flush=True)
    model = models.get_model(name, weights=weights).to(device).eval()
    weight_hash = sha256(weight_path)
    if not weight_hash.startswith(prefix):
        raise ValueError('Cached weight hash does not match official URL hash prefix')
    dataset = LabeledImages(rows, transform)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    correct1 = correct5 = count = 0
    predictions = []
    with torch.inference_mode():
        start = time.perf_counter()
        for images, labels in loader:
            logits = model(images.to(device))
            if logits.shape != (len(labels), 1000):
                raise ValueError('Expected the original 1000-class classifier')
            scores, predicted = logits.softmax(1).topk(5, dim=1)
            predicted, scores = predicted.cpu(), scores.cpu()
            correct1 += int((predicted[:, 0] == labels).sum())
            correct5 += int((predicted == labels[:, None]).any(1).sum())
            for i, label in enumerate(labels):
                predictions.append(dict(id=rows[count + i]['id'], true=int(label),
                    top5=predicted[i].tolist(), scores=scores[i].tolist()))
            count += len(labels)
            print(f'{name}: accuracy pass {count}/{len(dataset)}', flush=True)
        synchronize(device)
        evaluation_seconds = time.perf_counter() - start
        del logits
        images = torch.stack([dataset[i][0] for i in range(min(config.BATCH_SIZE, len(dataset)))])
        # Keep only one image resident for batch-1 memory measurement.
        single = images[:1].clone().to(device)
        print(f'{name}: measuring batch-1 latency', flush=True)
        latency = measure(model, single, device)
        del single
        batch = images.to(device)
        print(f'{name}: measuring batch-{len(batch)} throughput', flush=True)
        throughput = measure(model, batch, device)
    result = dict(evaluated_at=datetime.now(timezone.utc).isoformat(),
        profile=runtime['profile'], weights_url=weights.url, weights_sha256=weight_hash,
        timing_image_ids=[row['id'] for row in rows[:len(images)]],
        runtime=runtime, source_sha256=sha256(Path(__file__)), os=platform.platform(),
        gpu_vram_bytes=torch.cuda.get_device_properties(device).total_memory if device.type == 'cuda' else None,
        tf32=False, model=name, weights=config.MODELS[name], dataset=config.DATASET_NAME,
        split=config.SPLIT, n=count, top1_correct=correct1, top1_wrong=count-correct1,
        top5_correct=correct5, top1_percent=100 * correct1 / count,
        top5_percent=100 * correct5 / count, parameters=sum(p.numel() for p in model.parameters()),
        transform=str(transform), device=str(device), precision='FP32',
        gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        python=platform.python_version(), torch=str(torch.__version__), torchvision=str(torchvision.__version__),
        manifest_sha256=hashlib.sha256((folder / 'manifest.json').read_bytes()).hexdigest(),
        evaluation_seconds=evaluation_seconds, latency=latency, throughput=throughput,
        timing_scope='forward only, preprocessed resident input, synchronized wall time; excludes load/transfer',
        categories=weights.meta['categories'])
    save_json(folder / f'{name}.json', result)
    save_json(folder / f'{name}_predictions.json', predictions)


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == '--model':
        benchmark(sys.argv[2], Path(sys.argv[3]))
    else:
        # Preflight runs before any weight download or new result folder.
        if sys.argv[1:] not in ([], ['--smoke']):
            raise SystemExit('Usage: python 1_benchmark.py [--smoke]')
        smoke = sys.argv[1:] == ['--smoke']
        rows = read_manifest()
        if smoke:
            rows = rows[:16]
            config.WARMUP, config.REPEATS, config.TIMING_REQUESTS = 2, 1, 3
        prefix = 'smoke' if smoke else 'lan'
        for value in (config.BATCH_SIZE, config.WARMUP, config.REPEATS, config.TIMING_REQUESTS):
            if value < 1:
                raise ValueError('Benchmark counts must be positive')
        config.OUTPUT.mkdir(exist_ok=True)
        number = 1
        while (config.OUTPUT / f'{prefix}_{number}').exists():
            number += 1
        folder = config.OUTPUT / f'{prefix}_{number}'
        folder.mkdir()
        save_json(folder / 'manifest.json', rows)
        runtime = {key: getattr(config, key) for key in ('WARMUP', 'REPEATS', 'TIMING_REQUESTS', 'BATCH_SIZE')}
        runtime.update(profile='smoke' if smoke else 'benchmark', seed=config.SEED, models=config.MODELS)
        save_json(folder / 'run_settings.json', runtime)
        statuses, results = {}, []
        order = list(config.MODELS)
        random.Random(config.SEED).shuffle(order)
        save_json(folder / 'model_order.json', order)
        for name in order:
            print(f'Starting {name}', flush=True)
            with (folder / f'{name}.log').open('w', encoding='utf-8') as log:
                completed = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                    '--model', name, str(folder.resolve())], stdout=log, stderr=subprocess.STDOUT)
            statuses[name] = dict(returncode=completed.returncode, status='completed' if completed.returncode == 0 else 'failed; see model log')
            save_json(folder / 'status.json', statuses)
            if completed.returncode == 0:
                result = json.loads((folder / f'{name}.json').read_text())
                results.append(dict(model=name, weights=result['weights'], n=result['n'],
                    top1_correct=result['top1_correct'], top1_wrong=result['top1_wrong'],
                    top5_correct=result['top5_correct'],
                    top1_percent=result['top1_percent'], top5_percent=result['top5_percent'],
                    latency_ms=result['latency']['mean_ms'], p95_ms=result['latency']['p95_ms'],
                    throughput_batch=result['throughput']['batch'], images_per_second=result['throughput']['images_per_second'],
                    peak_allocated_mib=result['throughput']['peak_allocated_mib'],
                    peak_reserved_mib=result['throughput']['peak_reserved_mib'], parameters_m=result['parameters']/1e6))
        if results:
            with (folder / 'results.csv').open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(results[0]))
                writer.writeheader(); writer.writerows(results)
        print('Output:', folder)
        if any(row['returncode'] for row in statuses.values()):
            raise SystemExit('Some models failed; partial results and logs preserved.')
