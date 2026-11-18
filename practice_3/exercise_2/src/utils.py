"""Small, dependency-light, auditable artifact utilities."""
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as src:
        for chunk in iter(lambda: src.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def sha256_records(dataset):
    """Order-sensitive hash of text and label, independent of Arrow cache paths."""
    h = hashlib.sha256()
    for record in dataset:
        h.update(json.dumps({'text': record['text'], 'label': int(record['label'])},
                            ensure_ascii=False, sort_keys=True).encode('utf-8'))
        h.update(b'\n')
    return h.hexdigest()

def environment():
    import torch
    import transformers
    import datasets
    return {
        'recorded_utc': datetime.now(timezone.utc).isoformat(),
        'python': sys.version.split()[0], 'platform': platform.platform(),
        'torch': torch.__version__, 'transformers': transformers.__version__,
        'datasets': datasets.__version__, 'device':
            'cuda' if torch.cuda.is_available() else 'cpu',
        'cuda_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }

def next_run(root):
    root.mkdir(parents=True, exist_ok=True)
    i = 1
    while (root / f'lan_{i}').exists():
        i += 1
    result = root / f'lan_{i}'
    result.mkdir()
    return result
