from __future__ import annotations

from typing import Tuple

import numpy as np

def scale_point_to_iso(p_zyx: np.ndarray, spacing_zyx: Tuple[float,float,float], iso_mm: float) -> np.ndarray:
    sz, sy, sx = spacing_zyx
    scale = np.array([sz/iso_mm, sy/iso_mm, sx/iso_mm], dtype=np.float32)
    return p_zyx.astype(np.float32) * scale

def map_point_crop_to_input(p_crop: np.ndarray, scale: np.ndarray, shift: np.ndarray, input_shape: Tuple[int,int,int]) -> np.ndarray:
    p = p_crop.astype(np.float32) * scale.astype(np.float32) + shift.astype(np.float32)
    # clamp
    p = np.array([
        max(0.0, min(input_shape[0]-1.0, float(p[0]))),
        max(0.0, min(input_shape[1]-1.0, float(p[1]))),
        max(0.0, min(input_shape[2]-1.0, float(p[2]))),
    ], dtype=np.float32)
    return p

def map_point_input_to_crop(p_in: np.ndarray, scale: np.ndarray, shift: np.ndarray) -> np.ndarray:
    inv = 1.0 / np.maximum(scale.astype(np.float32), 1e-8)
    return (p_in.astype(np.float32) - shift.astype(np.float32)) * inv
