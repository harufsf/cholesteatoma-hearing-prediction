from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from chole_predict.data.categorical import map_category


@dataclass
class TabularBatchSpec:
    id_col: str
    num_cols: list[str]
    target_cols: list[str]


class TabularDataset(Dataset):
    def __init__(self, df: pd.DataFrame, spec: TabularBatchSpec, scaler, cat_maps: dict[str, dict[str, int]]):
        self.df = df.reset_index(drop=True)
        self.spec = spec
        self.scaler = scaler
        self.cat_maps = cat_maps

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        pid = str(row[self.spec.id_col])
        num = row[self.spec.num_cols].to_numpy(dtype=np.float32).reshape(1, -1)
        num = self.scaler.transform(num).astype(np.float32).reshape(-1)
        num_t = torch.from_numpy(num)
        cats = {
            "sex": torch.tensor(map_category(row["sex"], self.cat_maps["sex"]), dtype=torch.long),
            "disease": torch.tensor(map_category(row["disease"], self.cat_maps["disease"]), dtype=torch.long),
            "primary_or_recur": torch.tensor(map_category(row["primary_or_recur"], self.cat_maps["primary_or_recur"]), dtype=torch.long),
        }
        y = torch.from_numpy(row[self.spec.target_cols].to_numpy(dtype=np.float32))
        return pid, num_t, cats, y
