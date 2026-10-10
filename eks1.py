import torch
import torch.nn as nn
from torchinfo import summary

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(3, 16, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(16 * 15 * 15, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv(x)))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

model = CNN()

summary(model, input_size=(1, 3, 32, 32))

