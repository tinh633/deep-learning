import json
import random

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from config import (DATA, DATA_FOLDER, IMAGENET_MEAN, IMAGENET_STD,
                    SPLIT_FILE)


def load_datasets(download=False):
    """Return original PIL images; no transforms so EDA can inspect raw data."""
    train = datasets.CIFAR10(root=str(DATA_FOLDER), train=True, download=download)
    test = datasets.CIFAR10(root=str(DATA_FOLDER), train=False, download=download)
    return train, test


def load_split(targets, settings=None, split_file=SPLIT_FILE):
    """Reuse saved stratified indices; reject stale or inconsistent splits."""
    settings = DATA if settings is None else settings
    labels = np.asarray(targets)
    if split_file.exists():
        split = json.loads(split_file.read_text(encoding='utf-8'))
        if (split['seed'] != settings['seed'] or
                split['val_size'] != settings['val_size'] or
                split['targets'] != labels.tolist()):
            raise ValueError('Saved split does not match data/settings; do not overwrite it.')
    else:
        train_ids, val_ids = train_test_split(
            np.arange(len(labels)), test_size=settings['val_size'],
            random_state=settings['seed'], stratify=labels)
        split = dict(seed=settings['seed'], val_size=settings['val_size'],
                     targets=labels.tolist(), train_indices=train_ids.tolist(),
                     val_indices=val_ids.tolist())

    train_ids = split['train_indices']
    val_ids = split['val_indices']
    combined = train_ids + val_ids
    if (len(val_ids) != settings['val_size'] or
            len(combined) != len(labels) or sorted(combined) != list(range(len(labels)))):
        raise ValueError('Split indices must cover the dataset exactly once.')
    classes, counts = np.unique(labels, return_counts=True)
    for ids in (train_ids, val_ids):
        actual = np.array([(labels[ids] == label).sum() for label in classes])
        if np.any(np.abs(actual - counts * len(ids) / len(labels)) > 1):
            raise ValueError('Saved split does not preserve class proportions.')
    if not split_file.exists():
        split_file.parent.mkdir(parents=True, exist_ok=True)
        split_file.write_text(json.dumps(split, indent=2), encoding='utf-8')
    return train_ids, val_ids


def seed_worker(worker_id):
    # DataLoader gives each worker a reproducible torch seed.
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def build_loaders(settings=None, download=False, training=True):
    """Use augmentation/shuffling only when preparing a training run."""
    settings = DATA if settings is None else settings
    train, test = load_datasets(download=download)
    train_ids, val_ids = load_split(train.targets, settings)
    validation = datasets.CIFAR10(root=str(DATA_FOLDER), train=True, download=False)
    resize = transforms.Resize(
        (settings['image_size'], settings['image_size']),
        interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    train.transform = transforms.Compose([
        resize, transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
    evaluation_transform = transforms.Compose([resize, transforms.ToTensor(), normalize])
    if not training:
        train.transform = evaluation_transform
    validation.transform = evaluation_transform
    test.transform = evaluation_transform

    torch.manual_seed(settings['seed'])
    options = dict(batch_size=settings['batch_size'],
                   num_workers=settings['num_workers'],
                   pin_memory=settings['pin_memory'], worker_init_fn=seed_worker)
    train_loader = DataLoader(
        Subset(train, train_ids), shuffle=training,
        generator=torch.Generator().manual_seed(settings['seed']), **options)
    val_loader = DataLoader(Subset(validation, val_ids), shuffle=False, **options)
    test_loader = DataLoader(test, shuffle=False, **options)
    return train_loader, val_loader, test_loader
