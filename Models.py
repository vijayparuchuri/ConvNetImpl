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
    
    
def nin_block(out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.LazyConv2d(out_channels, kernel_size=kernel_size, stride=stride, padding=padding), nn.ReLU(),
        nn.LazyConv2d(out_channels, kernel_size=1), nn.ReLU(),
        nn.LazyConv2d(out_channels, kernel_size=1), nn.ReLU()
    )
    
class NiN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nin_block(96, kernel_size=11, stride=4, padding=0),
            nn.MaxPool2d(3, stride=2),
            nin_block(256, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(3, stride=2),
            nin_block(384, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout(0.5),
            nin_block(num_classes, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()            
        )
    def forward(self, X):
        logits = self.net(X)
        return logits