from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from chole_predict.data.tabular_dataset import TabularBatchSpec
from chole_predict.data.categorical import map_category
from chole_predict.roi.canonicalize import unify_to_right
from chole_predict.roi.preprocess import hu_preprocess, resize_3d_torch


class ResidualROIDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        spec: TabularBatchSpec,
        scaler,
        cat_maps: dict[str, dict[str, int]],
        roi_cols: list[str],
        y_tab_map: dict[str, np.ndarray],
        out_dhw: tuple[int, int, int] = (160, 192, 192),
        side_col: str = "side",
    ):
        self.df = df.reset_index(drop=True)
        self.spec = spec
        self.scaler = scaler
        self.cat_maps = cat_maps
        self.roi_cols = roi_cols
        self.y_tab_map = y_tab_map
        self.out_dhw = out_dhw
        self.side_col = side_col

    def __len__(self) -> int:
        return len(self.df)

    def _load_vol(self, path: str, side: str) -> torch.Tensor:
        arr = np.load(path).astype(np.float32)
        arr = unify_to_right(arr, side)
        arr = hu_preprocess(arr)
        ten = torch.from_numpy(arr)[None, None, ...]
        ten = resize_3d_torch(ten, self.out_dhw).squeeze(0)
        return ten.squeeze(0)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        pid = str(row[self.spec.id_col])
        side = str(row.get(self.side_col, "R"))
        num = row[self.spec.num_cols].to_numpy(dtype=np.float32).reshape(1, -1)
        num = self.scaler.transform(num).astype(np.float32).reshape(-1)
        num_t = torch.from_numpy(num)
        cats = {
            "sex": torch.tensor(map_category(row["sex"], self.cat_maps["sex"]), dtype=torch.long),
            "disease": torch.tensor(map_category(row["disease"], self.cat_maps["disease"]), dtype=torch.long),
            "primary_or_recur": torch.tensor(map_category(row["primary_or_recur"], self.cat_maps["primary_or_recur"]), dtype=torch.long),
        }
        vols = []
        for c in self.roi_cols:
            p = row[c]
            if not isinstance(p, str) or not Path(p).exists():
                raise FileNotFoundError(f"Missing ROI npy: {p}")
            vols.append(self._load_vol(p, side))
        vols_t = torch.stack(vols, dim=0)
        y_tab = torch.from_numpy(np.asarray(self.y_tab_map[pid], dtype=np.float32))
        y = torch.from_numpy(row[self.spec.target_cols].to_numpy(dtype=np.float32))
        return pid, num_t, cats, vols_t, y_tab, y
