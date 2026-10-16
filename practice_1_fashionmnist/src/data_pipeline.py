
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

data_folder = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "FashionMNIST_raw"
)


class FashionMNIST(datasets.FashionMNIST):
    @property
    def raw_folder(self):
        return str(data_folder)


# Augmentation chỉ dành cho training
train_transform = transforms.Compose([
    transforms.RandomAffine(
        degrees=5,
        translate=(0.05, 0.05)
    ),
    transforms.ToTensor()
])

# Validation và test không augmentation
eval_transform = transforms.ToTensor()


train_full = FashionMNIST(
    root=str(data_folder.parent),
    train=True,
    download=False,
    transform=train_transform
)

val_full = FashionMNIST(
    root=str(data_folder.parent),
    train=True,
    download=False,
    transform=eval_transform
)

test_dataset = FashionMNIST(
    root=str(data_folder.parent),
    train=False,
    download=False,
    transform=eval_transform
)

train_indices, val_indices = train_test_split(
    range(len(train_full)),
    test_size=0.1,
    random_state=42,
    stratify=train_full.targets
)

train_dataset = Subset(train_full, train_indices)
val_dataset = Subset(val_full, val_indices)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    generator=torch.Generator().manual_seed(42)
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

if __name__ == "__main__":
    print("Train:", len(train_dataset))
    print("Validation:", len(val_dataset))
    print("Test:", len(test_dataset))

    images, labels = next(iter(train_loader))
    print("Batch:", images.shape, labels.shape)
