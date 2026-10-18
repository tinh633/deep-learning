import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from config import DATA, DATA_FOLDER


class FashionMNIST(datasets.FashionMNIST):
    @property
    def raw_folder(self):
        return str(DATA_FOLDER)


def build_loaders(settings=None):
    settings = DATA if settings is None else settings
    transform = transforms.ToTensor()
    original = FashionMNIST(root=str(DATA_FOLDER.parent), train=True,
                           download=False, transform=transform)
    test = FashionMNIST(root=str(DATA_FOLDER.parent), train=False,
                       download=False, transform=transform)
    train_indices, val_indices = train_test_split(
        range(len(original)), test_size=settings['val_size'],
        random_state=settings['seed'], stratify=original.targets)
    train, val = Subset(original, train_indices), Subset(original, val_indices)
    batch = settings['batch_size']
    return (DataLoader(train, batch_size=batch, shuffle=True,
                       generator=torch.Generator().manual_seed(settings['seed'])),
            DataLoader(val, batch_size=batch, shuffle=False),
            DataLoader(test, batch_size=batch, shuffle=False))
