from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

data_folder = Path(__file__).resolve().parents[1] / 'data' / 'FashionMNIST_raw'


# Giữ thư mục dữ liệu ngắn thay vì cache mặc định FashionMNIST/raw.
class FashionMNIST(datasets.FashionMNIST):
    @property
    def raw_folder(self):
        return str(data_folder)


transform = transforms.ToTensor()  # Ảnh float32, miền [0, 1].

train_dataset = FashionMNIST(root=str(data_folder.parent), train=True,
                            download=False, transform=transform)
test_dataset = FashionMNIST(root=str(data_folder.parent), train=False,
                           download=False, transform=transform)

train_indices, val_indices = train_test_split(
    range(len(train_dataset)), test_size=0.1, random_state=42,
    stratify=train_dataset.targets)
val_dataset = Subset(train_dataset, val_indices)
train_dataset = Subset(train_dataset, train_indices)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,
                          generator=torch.Generator().manual_seed(42))
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

if __name__ == '__main__':
    print('Train:', len(train_dataset))
    print('Validation:', len(val_dataset))
    print('Test:', len(test_dataset))
    images, labels = next(iter(train_loader))
    print('Batch:', images.shape, labels.shape)
