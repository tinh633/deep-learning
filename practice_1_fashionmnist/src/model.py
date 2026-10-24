from torch import nn


class FashionMLP(nn.Module):
    def __init__(self, hidden1=512, hidden2=256, hidden3=128, dropout=0.2, hidden4=None):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, hidden3)
        # Without hidden4, keep the original three-hidden-layer checkpoint layout.
        self.fc4 = nn.Linear(hidden3, hidden4 if hidden4 is not None else 10)
        self.fc5 = nn.Linear(hidden4, 10) if hidden4 is not None else None
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

    def forward(self, images):
        # Chuyển mỗi ảnh 28 x 28 thành vector 784 giá trị.
        x = self.flatten(images)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout3(x)
        x = self.fc4(x)

        if self.fc5 is not None:
            x = self.relu(x)
            x = self.dropout4(x)
            x = self.fc5(x)

        return x
