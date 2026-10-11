import torch
from torch import nn

model = nn.Sequential(
            nn.Flatten(),

            nn.Linear(784, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 10)

)

if __name__ == '__main__':
    from data_pipeline import train_loader

    images, labels = next(iter(train_loader))
    with torch.no_grad():
        outputs = model(images)
    assert outputs.shape == (images.shape[0], 10)
    assert torch.isfinite(outputs).all()
    print(model)
    print('Input:', images.shape)
    print('Output:', outputs.shape)
