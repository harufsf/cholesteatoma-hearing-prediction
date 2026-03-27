from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import zoom

DEFAULT_HU_PAD = -1024.0

def crop_around_center(vol_zyx: np.ndarray,
                       center_zyx: np.ndarray,
                       iso_mm: float,
                       crop_size_mm: Tuple[float,float,float],
                       pad_hu: float = DEFAULT_HU_PAD,
                       crop_y_front_mm: Optional[float] = None,
                       crop_y_back_mm: Optional[float] = None
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Crop a box of fixed voxel size (exclusive end) centered at center_zyx.

    v9.1: allow optional *asymmetric* crop on Y to reduce anterior (front) field of view:
      - crop_y_front_mm: mm to keep to the +Y (front) direction from center
      - crop_y_back_mm : mm to keep to the -Y (back) direction from center
    If both are None, falls back to symmetric crop using crop_size_mm[1].

    Returns:
      vol_crop (float32)
      origin_full_zyx (int32) : full coord corresponding to crop index (0,0,0) (can be negative)
      pad_before_zyx (int32)  : amount of padding added before in each axis

    Mapping:
      full = crop + origin_full
      crop = full - origin_full
    """
    cz, cy, cx = [float(v) for v in center_zyx]
    Z, Y, X = vol_zyx.shape
    sz_mm, sy_mm, sx_mm = crop_size_mm

    sz = int(round(sz_mm / iso_mm))
    sx = int(round(sx_mm / iso_mm))
    sz = max(8, sz); sx = max(8, sx)

    # Y: symmetric or asymmetric
    if (crop_y_front_mm is None) and (crop_y_back_mm is None):
        sy = int(round(sy_mm / iso_mm))
        sy = max(8, sy)
        sy_front = None
        sy_back = None
    else:
        front_mm = float(crop_y_front_mm) if crop_y_front_mm is not None else float(sy_mm) * 0.5
        back_mm  = float(crop_y_back_mm)  if crop_y_back_mm  is not None else float(sy_mm) * 0.5
        front_mm = max(1.0, front_mm)
        back_mm  = max(1.0, back_mm)
        sy_front = max(1, int(round(front_mm / iso_mm)))
        sy_back  = max(1, int(round(back_mm / iso_mm)))
        sy = max(8, sy_front + sy_back)

    hz = sz // 2
    hx = sx // 2

    z0 = int(round(cz)) - hz
    x0 = int(round(cx)) - hx
    z1 = z0 + sz  # exclusive
    x1 = x0 + sx

    if sy_front is None:
        hy = sy // 2
        y0 = int(round(cy)) - hy
        y1 = y0 + sy  # exclusive
    else:
        y0 = int(round(cy)) - sy_back
        y1 = int(round(cy)) + sy_front  # exclusive

    # Padding if outside
    pad_before = np.array([0,0,0], dtype=np.int32)
    pad_after  = np.array([0,0,0], dtype=np.int32)

    if z0 < 0:
        pad_before[0] = -z0
        z0 = 0
    if y0 < 0:
        pad_before[1] = -y0
        y0 = 0
    if x0 < 0:
        pad_before[2] = -x0
        x0 = 0

    if z1 > Z:
        pad_after[0] = z1 - Z
        z1 = Z
    if y1 > Y:
        pad_after[1] = y1 - Y
        y1 = Y
    if x1 > X:
        pad_after[2] = x1 - X
        x1 = X

    vol_crop = vol_zyx[z0:z1, y0:y1, x0:x1].astype(np.float32)

    if (pad_before.sum() + pad_after.sum()) > 0:
        vol_crop = np.pad(vol_crop,
                          ((pad_before[0], pad_after[0]),
                           (pad_before[1], pad_after[1]),
                           (pad_before[2], pad_after[2])),
                          mode="constant",
                          constant_values=float(pad_hu)).astype(np.float32)

    origin_full = np.array([z0, y0, x0], dtype=np.int32) - pad_before.astype(np.int32)
    return vol_crop, origin_full.astype(np.int32), pad_before.astype(np.int32)

def force_shape(vol: np.ndarray, shape: Tuple[int,int,int], pad_value: float) -> np.ndarray:
    tz, ty, tx = shape
    z, y, x = vol.shape
    # pad
    pz0 = max(0, (tz - z)//2); pz1 = max(0, tz - z - pz0)
    py0 = max(0, (ty - y)//2); py1 = max(0, ty - y - py0)
    px0 = max(0, (tx - x)//2); px1 = max(0, tx - x - px0)
    if any([pz0,pz1,py0,py1,px0,px1]):
        vol = np.pad(vol, ((pz0,pz1),(py0,py1),(px0,px1)), mode="constant", constant_values=pad_value)
    # crop
    z, y, x = vol.shape
    z0 = max(0, (z - tz)//2); y0 = max(0, (y - ty)//2); x0 = max(0, (x - tx)//2)
    return vol[z0:z0+tz, y0:y0+ty, x0:x0+tx]

def resize_vol_to_input(vol_zyx: np.ndarray, target_shape: Tuple[int,int,int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Resize crop volume to model input shape using zoom (order=1).
    Returns:
      vol_resized (float32)
      scale_crop_to_in (float32 vec3) : in = crop * scale + shift
      shift_in (float32 vec3) : due to symmetric pad/crop in force_shape
    """
    z, y, x = vol_zyx.shape
    tz, ty, tx = target_shape
    scale = np.array([tz / max(1,z), ty / max(1,y), tx / max(1,x)], dtype=np.float32)
    vol_zoom = zoom(vol_zyx.astype(np.float32), tuple(scale.tolist()), order=1)

    # compute shift applied by force_shape (symmetric)
    zz, yy, xx = vol_zoom.shape
    shift = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    if zz < tz:
        shift[0] = (tz - zz) / 2.0
    if yy < ty:
        shift[1] = (ty - yy) / 2.0
    if xx < tx:
        shift[2] = (tx - xx) / 2.0
    if zz > tz:
        shift[0] = -(zz - tz) / 2.0
    if yy > ty:
        shift[1] = -(yy - ty) / 2.0
    if xx > tx:
        shift[2] = -(xx - tx) / 2.0

    vol_r = force_shape(vol_zoom, target_shape, pad_value=float(DEFAULT_HU_PAD)).astype(np.float32)
    return vol_r, scale, shift

def resize_crop_to_input(vol_crop_zyx: np.ndarray, target_shape: Tuple[int,int,int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compatibility wrapper."""
    return resize_vol_to_input(vol_crop_zyx, target_shape)

def apply_conditional_anterior_mask(vol_crop_zyx: np.ndarray,
                                    iso_mm: float,
                                    y_from_center_mm: float,
                                    alpha: float = 0.2,
                                    ramp_mm: float = 10.0,
                                    pad_hu: float = DEFAULT_HU_PAD,
                                    mode: str = "attenuate") -> np.ndarray:
    """Suppress anterior region in canonical crop cube (anterior assumed to be -Y).

    y_from_center_mm:
      distance from crop center toward anterior (-Y) at which suppression begins.
      Region with y <= (center_y - y_from_center_mm) will be suppressed.

    mode:
      - attenuate: blend HU towards pad_hu using weight w(y) in [alpha, 1]
      - zero: set suppressed region to pad_hu

    ramp_mm:
      smooth transition width (mm) to avoid hard edges.
    """
    vol = vol_crop_zyx.astype(np.float32, copy=True)
    Z, Y, X = vol.shape
    cy = (Y - 1) / 2.0
    start_y = cy - float(y_from_center_mm) / float(iso_mm)
    ramp = max(0.0, float(ramp_mm) / float(iso_mm))
    y = np.arange(Y, dtype=np.float32)

    if ramp <= 1e-6:
        w = np.where(y <= start_y, float(alpha), 1.0).astype(np.float32)
    else:
        # y <= start_y - ramp -> alpha; y >= start_y -> 1; linear in between
        w = np.ones((Y,), dtype=np.float32)
        w[y <= (start_y - ramp)] = float(alpha)
        mid = (y > (start_y - ramp)) & (y < start_y)
        if np.any(mid):
            t = (y[mid] - (start_y - ramp)) / max(1e-6, ramp)
            w[mid] = float(alpha) + (1.0 - float(alpha)) * t

    w3 = w.reshape(1, Y, 1)  # broadcast
    if str(mode).lower() == "zero":
        vol[:, y <= start_y, :] = float(pad_hu)
    else:
        vol = (w3 * vol) + ((1.0 - w3) * float(pad_hu))
    return vol.astype(np.float32)
