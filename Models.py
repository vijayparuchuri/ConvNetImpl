import torch
from torch import nn


class LeNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.LazyConv2d(6, 5, padding=2), nn.Sigmoid(),
            nn.AvgPool2d(2, 2),
            nn.LazyConv2d(16, 5), nn.Sigmoid(),
            nn.AvgPool2d(2, 2),
            nn.Flatten(),
            nn.LazyLinear(120),nn.Sigmoid(),
            nn.LazyLinear(84), nn.Sigmoid(),
            nn.LazyLinear(num_classes)
        )
        
    def forward(self, X):
        logits = self.net(X)
        return logits