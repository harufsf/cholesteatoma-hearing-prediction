from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

def estimate_anchor_center_ctonly(vol_canon_zyx: np.ndarray,
                                  iso_mm: float,
                                  method: str = "bbox_frac",
                                  x_frac: float = 0.70,
                                  y_shift_mm: float = 0.0,
                                  z_shift_mm: float = 0.0) -> np.ndarray:
    """Estimate a deterministic crop center from CT only (no GT/click).

    method:
      - bbox_frac: threshold-based body/skull mask -> bounding box -> take (z,y) center and x at a fraction toward the right.
      - volume_center: fallback to volume center.

    Notes:
      - Assumes vol is already LR-canonical (target ear on RIGHT side after optional flip).
      - x_frac in [0,1], where 0.5 is midline and >0.5 biases to right.
    """
    Z, Y, X = vol_canon_zyx.shape
    if method == "volume_center":
        c = np.array([(Z-1)/2.0, (Y-1)/2.0, (X-1)/2.0], dtype=np.float32)
    else:
        # robust threshold; exclude air/background
        thr = -300.0
        step = 2 if iso_mm <= 0.6 else 1
        v = vol_canon_zyx[::step, ::step, ::step]
        m = v > thr
        if not np.any(m):
            c = np.array([(Z-1)/2.0, (Y-1)/2.0, (X-1)/2.0], dtype=np.float32)
        else:
            zz, yy, xx = np.where(m)
            z0, z1 = int(zz.min())*step, int(zz.max())*step
            y0, y1 = int(yy.min())*step, int(yy.max())*step
            x0, x1 = int(xx.min())*step, int(xx.max())*step
            cz = (z0 + z1) / 2.0
            cy = (y0 + y1) / 2.0
            xf = float(np.clip(x_frac, 0.0, 1.0))
            cx = x0 + xf * max(1.0, (x1 - x0))
            c = np.array([cz, cy, cx], dtype=np.float32)

    # apply optional shifts (mm -> vox)
    c[0] += float(z_shift_mm) / float(iso_mm)
    c[1] += float(y_shift_mm) / float(iso_mm)
    # clamp to volume bounds
    c[0] = float(np.clip(c[0], 0.0, float(Z-1)))
    c[1] = float(np.clip(c[1], 0.0, float(Y-1)))
    c[2] = float(np.clip(c[2], 0.0, float(X-1)))
    return c.astype(np.float32)
