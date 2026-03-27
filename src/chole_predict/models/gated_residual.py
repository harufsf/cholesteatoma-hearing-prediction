from __future__ import annotations

import torch
import torch.nn as nn

from chole_predict.models.roi_pooling import MultiROIPool


class ResidualGatedROIModel(nn.Module):
    def __init__(self, num_dim: int, n_sex: int, n_dis: int, n_pr: int, out_dim: int, n_rois: int, roi_pool: str = "concat", gate_use_ytab: bool = True):
        super().__init__()
        self.gate_use_ytab = gate_use_ytab
        self.sex_emb = nn.Embedding(n_sex, 8)
        self.dis_emb = nn.Embedding(n_dis, 8)
        self.pr_emb = nn.Embedding(n_pr, 8)
        self.roi = MultiROIPool(n_rois=n_rois, emb_dim=64, mode=roi_pool)
        roi_dim = 64 if roi_pool == "mean" else 64 * n_rois
        tab_dim = num_dim + 8 + 8 + 8
        base_dim = tab_dim + roi_dim
        self.delta_head = nn.Sequential(nn.Linear(base_dim, 128), nn.ReLU(True), nn.Dropout(0.2), nn.Linear(128, out_dim))
        gate_in = base_dim + (out_dim if gate_use_ytab else 0)
        self.gate_head = nn.Sequential(nn.Linear(gate_in, 64), nn.ReLU(True), nn.Linear(64, out_dim))

    def _tab_embed(self, num: torch.Tensor, cats: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([num, self.sex_emb(cats["sex"]), self.dis_emb(cats["disease"]), self.pr_emb(cats["primary_or_recur"] )], dim=1)

    def forward(self, vols: torch.Tensor, num: torch.Tensor, cats: dict[str, torch.Tensor], y_tab: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        base = torch.cat([self.roi(vols), self._tab_embed(num, cats)], dim=1)
        delta = self.delta_head(base)
        gate_input = torch.cat([base, y_tab], dim=1) if self.gate_use_ytab and y_tab is not None else base
        gate = torch.sigmoid(self.gate_head(gate_input))
        return delta, gate
