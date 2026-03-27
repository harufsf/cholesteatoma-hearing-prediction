from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.ndimage import zoom


def extract_spherical_roi(vol_zyx: np.ndarray, center_zyx: Tuple[int, int, int], cube_mm: float, spacing_zyx_mm: Tuple[float, float, float], out_shape: Tuple[int, int, int] = (64, 64, 64), fill_hu: float = -1000.0) -> np.ndarray:
    half_mm = cube_mm / 2.0
    rz = int(round(half_mm / spacing_zyx_mm[0]))
    ry = int(round(half_mm / spacing_zyx_mm[1]))
    rx = int(round(half_mm / spacing_zyx_mm[2]))
    cz, cy, cx = center_zyx
    z0, z1 = max(0, cz - rz), min(vol_zyx.shape[0], cz + rz)
    y0, y1 = max(0, cy - ry), min(vol_zyx.shape[1], cy + ry)
    x0, x1 = max(0, cx - rx), min(vol_zyx.shape[2], cx + rx)
    crop = np.full((2 * rz, 2 * ry, 2 * rx), fill_hu, dtype=np.float32)
    src = vol_zyx[z0:z1, y0:y1, x0:x1]
    dz0 = max(0, rz - cz)
    dy0 = max(0, ry - cy)
    dx0 = max(0, rx - cx)
    crop[dz0:dz0 + src.shape[0], dy0:dy0 + src.shape[1], dx0:dx0 + src.shape[2]] = src
    scale = np.array(out_shape, dtype=np.float32) / np.array(crop.shape, dtype=np.float32)
    roi_resized = zoom(crop, scale, order=1)
    cz2, cy2, cx2 = np.array(out_shape, dtype=np.float32) / 2.0
    zz, yy, xx = np.ogrid[:out_shape[0], :out_shape[1], :out_shape[2]]
    dist_sq = (zz - cz2) ** 2 + (yy - cy2) ** 2 + (xx - cx2) ** 2
    mask_radius_sq = (min(out_shape) / 2.0) ** 2
    roi = roi_resized.astype(np.float32, copy=True)
    roi[dist_sq > mask_radius_sq] = float(fill_hu)
    return roi
