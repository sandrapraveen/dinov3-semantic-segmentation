import torch
import torch.nn as nn
import torch.nn.functional as F


class DPTHead(nn.Module):
    """
    Simplified DPT-style decoder for ViT/DINO features.

    Inputs:
        features: list of 4 tensors
                  each of shape (B, C, H, W)

    Output:
        segmentation map (B, num_classes, H, W)
    """

    def __init__(self, in_channels=1024, num_classes=7, features=256):
        super().__init__()

        # 1. Project each feature → same channel size
        self.proj = nn.ModuleList([
            nn.Conv2d(in_channels, features, kernel_size=1)
            for _ in range(4)
        ])

        # 2. Fusion block (after concatenation)
        self.fusion = nn.Sequential(
            nn.Conv2d(features * 4, features, kernel_size=3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),

            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # 3. Output head
        self.head = nn.Conv2d(features, num_classes, kernel_size=1)

    def forward(self, features):
        """
        features = [f1, f2, f3, f4]
        each: (B, C, H, W)
        """

        # Project features
        proj_feats = []
        for i, f in enumerate(features):
            proj_feats.append(self.proj[i](f))

        # Ensure same spatial size (safety)
        target_size = proj_feats[0].shape[-2:]
        proj_feats = [
            F.interpolate(f, size=target_size, mode="bilinear", align_corners=False)
            for f in proj_feats
        ]

        # Concatenate along channel dimension
        x = torch.cat(proj_feats, dim=1)  # (B, 4*features, H, W)

        # Fuse
        x = self.fusion(x)

        # Predict
        out = self.head(x)

        return out