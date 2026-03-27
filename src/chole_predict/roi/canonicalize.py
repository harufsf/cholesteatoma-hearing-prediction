from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

def normalize_side_value(side: Any) -> Optional[str]:
    if side is None:
        return None
    s = str(side).strip().upper()
    if s in ["R", "RIGHT", "RT"]:
        return "R"
    if s in ["L", "LEFT", "LT"]:
        return "L"
    return None

def flip_x_point(p_zyx: np.ndarray, X: int) -> np.ndarray:
    p = np.asarray(p_zyx, dtype=np.float32).copy()
    p[2] = (float(X) - 1.0) - p[2]
    return p

def canonicalize_lr(vol_zyx: np.ndarray, side_rl: Optional[str], pts_zyx: Optional[Dict[str, Any]] = None
                   ) -> Tuple[np.ndarray, Dict[str, np.ndarray], bool]:
    if pts_zyx is None:
        pts_zyx = {}
    pts_out: Dict[str, np.ndarray] = {}
    for k, v in pts_zyx.items():
        arr = as_vec3(v)
        if arr is not None:
            pts_out[k] = arr

    if side_rl not in ("L", "R") or side_rl == "R":
        return vol_zyx, pts_out, False

    X = int(vol_zyx.shape[2])
    vol_flip = vol_zyx[:, :, ::-1].copy()
    for k, p in pts_out.items():
        pts_out[k] = np.array([p[0], p[1], (X - 1) - p[2]], dtype=np.float32)
    return vol_flip, pts_out, True



def unify_to_right(vol: np.ndarray, side: Any) -> np.ndarray:
    side_norm = normalize_side_value(side)
    if side_norm == "L":
        return np.ascontiguousarray(vol[..., ::-1])
    return vol
