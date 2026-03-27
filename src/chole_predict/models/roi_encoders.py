from __future__ import annotations

import torch
import torch.nn as nn


class Small3DCNN(nn.Module):
    def __init__(self, in_channels: int = 1, emb_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, 16, 3, padding=1), nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3, padding=1), nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(32, 64, 3, padding=1), nn.ReLU(True), nn.AdaptiveAvgPool3d(1),
        )
        self.fc = nn.Linear(64, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x).flatten(1)
        return self.fc(h)
