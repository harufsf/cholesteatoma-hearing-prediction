from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def hu_preprocess(vol: np.ndarray, hu_min: float = -1024.0, hu_max: float = 3000.0) -> np.ndarray:
    vol = np.asarray(vol, dtype=np.float32)
    vol = np.clip(vol, hu_min, hu_max)
    vol = (vol - hu_min) / (hu_max - hu_min + 1e-12)
    return vol.astype(np.float32)


def resize_3d_torch(x: torch.Tensor, out_dhw: tuple[int, int, int]) -> torch.Tensor:
    return F.interpolate(x, size=out_dhw, mode="trilinear", align_corners=False)
