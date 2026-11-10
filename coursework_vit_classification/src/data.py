"""What: Tải CIFAR-100 và quản lý split. Why: Các thí nghiệm dùng chung dữ liệu.
Who: EDA và train/evaluate sau này. When: Khi gọi hàm, không tải lúc import.
Where: data/ và outputs/cifar100_split.json. How: torchvision và stratified split.
"""
import json

import numpy as np
from sklearn.model_selection import train_test_split
from torchvision.datasets import CIFAR100

from config import DATA_DIR, OUTPUT_DIR, SEED, SPLIT_FILE, VALIDATION_SIZE


def load_datasets(download=False):
    original = CIFAR100(root=str(DATA_DIR), train=True, download=download)
    test = CIFAR100(root=str(DATA_DIR), train=False, download=download)
    return original, test


def load_split(targets):
    if SPLIT_FILE.exists():
        split = json.loads(SPLIT_FILE.read_text(encoding='utf-8'))
        if split['seed'] != SEED:
            raise ValueError('Seed khác split đã lưu; không tự ghi đè split.')
    else:
        train_ids, val_ids = train_test_split(
            np.arange(len(targets)), test_size=VALIDATION_SIZE,
            random_state=SEED, stratify=targets)
        split = {'seed': SEED, 'train': train_ids.tolist(), 'validation': val_ids.tolist()}
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        SPLIT_FILE.write_text(json.dumps(split), encoding='utf-8')
    train_ids, val_ids = split['train'], split['validation']
    if (len(train_ids) != len(targets) - VALIDATION_SIZE
            or len(val_ids) != VALIDATION_SIZE
            or sorted(train_ids + val_ids) != list(range(len(targets)))):
        raise ValueError('Split thiếu ảnh, trùng ảnh hoặc không đúng kích thước.')
    for ids, expected in ((train_ids, 450), (val_ids, 50)):
        if not np.all(np.bincount(np.asarray(targets)[ids], minlength=100) == expected):
            raise ValueError('Split không cân bằng theo lớp.')
    return train_ids, val_ids
