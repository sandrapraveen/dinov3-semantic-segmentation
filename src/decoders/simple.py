import torch
import torch.nn as nn

# 1024 channels → 256 → 7 classes

class SimpleDecoder(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x):
        return self.decoder(x)