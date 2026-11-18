from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(cin,cout,3,padding=1,bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
        )

    def forward(self,x):
        return self.net(x)


class LaundryShapeCNN(nn.Module):
    """3-layer silhouette CNN + 4 geometry features."""
    def __init__(self,num_classes: int=5):
        super().__init__()
        self.features=nn.Sequential(ConvBlock(1,8),ConvBlock(8,16),ConvBlock(16,32))
        self.gap=nn.AdaptiveAvgPool2d((1,1))
        self.classifier=nn.Sequential(
            nn.Linear(36,64),nn.ReLU(inplace=True),nn.Dropout(.25),nn.Linear(64,num_classes)
        )

    def forward(self,mask: torch.Tensor,geometry: torch.Tensor):
        spatial=self.gap(self.features(mask)).flatten(1)
        return self.classifier(torch.cat((spatial,geometry),dim=1))
