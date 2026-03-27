from __future__ import annotations

import torch
import torch.nn as nn


class TabularMLP(nn.Module):
    def __init__(self, num_dim: int, n_sex: int, n_dis: int, n_pr: int, out_dim: int, sex_emb: int = 8, dis_emb: int = 8, pr_emb: int = 8, hidden: int = 128):
        super().__init__()
        self.sex_emb = nn.Embedding(n_sex, sex_emb)
        self.dis_emb = nn.Embedding(n_dis, dis_emb)
        self.pr_emb = nn.Embedding(n_pr, pr_emb)
        in_dim = num_dim + sex_emb + dis_emb + pr_emb
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(True), nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.ReLU(True), nn.Dropout(0.1),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, num: torch.Tensor, cats: dict[str, torch.Tensor]) -> torch.Tensor:
        x = torch.cat([
            num,
            self.sex_emb(cats["sex"]),
            self.dis_emb(cats["disease"]),
            self.pr_emb(cats["primary_or_recur"]),
        ], dim=1)
        return self.net(x)
