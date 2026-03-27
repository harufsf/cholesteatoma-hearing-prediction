from __future__ import annotations

import torch
import torch.nn as nn

from chole_predict.models.roi_pooling import MultiROIPool


class ResidualROIModel(nn.Module):
    def __init__(self, num_dim: int, n_sex: int, n_dis: int, n_pr: int, out_dim: int, n_rois: int, roi_pool: str = "concat"):
        super().__init__()
        self.sex_emb = nn.Embedding(n_sex, 8)
        self.dis_emb = nn.Embedding(n_dis, 8)
        self.pr_emb = nn.Embedding(n_pr, 8)
        self.roi = MultiROIPool(n_rois=n_rois, emb_dim=64, mode=roi_pool)
        roi_dim = 64 if roi_pool == "mean" else 64 * n_rois
        in_dim = num_dim + 8 + 8 + 8 + roi_dim
        self.head = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(True), nn.Dropout(0.2),
            nn.Linear(128, out_dim),
        )

    def forward(self, vols: torch.Tensor, num: torch.Tensor, cats: dict[str, torch.Tensor]) -> torch.Tensor:
        img = self.roi(vols)
        x = torch.cat([num, self.sex_emb(cats["sex"]), self.dis_emb(cats["disease"]), self.pr_emb(cats["primary_or_recur"]), img], dim=1)
        return self.head(x)
