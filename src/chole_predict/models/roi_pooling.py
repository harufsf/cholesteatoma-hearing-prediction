from __future__ import annotations

import torch
import torch.nn as nn

from chole_predict.models.roi_encoders import Small3DCNN


class MultiROIPool(nn.Module):
    def __init__(self, n_rois: int, emb_dim: int = 64, mode: str = "concat"):
        super().__init__()
        self.mode = mode
        self.encoders = nn.ModuleList([Small3DCNN(1, emb_dim) for _ in range(n_rois)])

    def forward(self, vols: torch.Tensor) -> torch.Tensor:
        feats = []
        for i, enc in enumerate(self.encoders):
            feats.append(enc(vols[:, i:i+1]))
        if self.mode == "mean":
            return torch.stack(feats, dim=1).mean(dim=1)
        return torch.cat(feats, dim=1)
