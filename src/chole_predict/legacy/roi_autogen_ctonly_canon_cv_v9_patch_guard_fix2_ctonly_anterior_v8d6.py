#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ROI auto-generation (LR canonical / click-centered crop / GT=vw_point.json / CV train+predict + eval)

v9: **NO-LEAK** CV. When predicting fold k, training uses ONLY folds != k, and
    the fold-k cases are NOT used even for validation/early-stopping (dev split is drawn from training folds).
    Adds `reqa` mode to regenerate missing QA images from existing *_pred_center.json files.

v9 fixes "not learning / always corner prediction" by:
- Training heatmap as a *probability distribution* with log_softmax + KL divergence against a normalized Gaussian target.
  -> prevents trivial "all zeros" solution that MSE+sigmoid tends to converge to.
- Inference uses soft-argmax (expected coordinate) for stability; falls back to argmax if needed.
- Crop uses *exclusive end* indexing with exact voxel size (no +1 ambiguity).
- Robust dicom_dir inference when df_final_fixed.csv lacks dicom_dir (uses manifest or root/dicom_root/<id>).

Notes:
- Coordinates are always voxel index in ZYX.
- LR canonical: if side==L, flip X axis and apply to volume + all points synchronously.

"""

from __future__ import annotations

import os, re, json, math, glob, time, csv, argparse, random
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

# optional
HAS_MPL = False
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

try:
    import SimpleITK as sitk
except Exception as e:
    raise SystemExit("SimpleITK is required to load DICOM. Install SimpleITK first.") from e

try:
    from scipy.ndimage import zoom
except Exception as e:
    raise SystemExit("scipy is required (scipy.ndimage.zoom). Install scipy first.") from e

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as e:
    raise SystemExit("PyTorch is required. Install torch first.") from e



# -------------------------------------------------------------------------
# Reproducibility utilities
# -------------------------------------------------------------------------
def enable_determinism(seed: int, verbose: bool = True, strict: bool = False) -> None:
    """Enable deterministic/reproducible behavior as much as possible.

    Notes:
    - For best effect, also set env vars BEFORE launching python:
        CUBLAS_WORKSPACE_CONFIG=:4096:8
        PYTHONHASHSEED=<seed>
    """
    import os as _os
    import random as _random
    import numpy as _np

    seed = int(seed)

    # cuBLAS deterministic workspace (best set before CUDA context is created)
    _os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    _os.environ.setdefault("PYTHONHASHSEED", str(seed))

    _random.seed(seed)
    _np.random.seed(seed)

    try:
        import torch as _torch

        _torch.manual_seed(seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed(seed)
            _torch.cuda.manual_seed_all(seed)

        # cuDNN / TF32
        _torch.backends.cudnn.benchmark = False
        _torch.backends.cudnn.deterministic = True
        _torch.backends.cuda.matmul.allow_tf32 = False
        _torch.backends.cudnn.allow_tf32 = False

        # Force deterministic algorithms where possible
        _torch.use_deterministic_algorithms(True, warn_only=True)

        # Reduce nondeterminism from CPU threading (optional)
        try:
            _torch.set_num_threads(1)
        except Exception:
            pass

        if verbose:
            print(f"[deterministic] enabled (seed={seed})")
            if _torch.cuda.is_available():
                print(f"[deterministic] CUBLAS_WORKSPACE_CONFIG={_os.environ.get('CUBLAS_WORKSPACE_CONFIG')}")
    except Exception as e:
        if verbose:
            print(f"[deterministic][WARN] torch deterministic setup skipped: {e}")




DEFAULT_ISO_MM = 0.5
DEFAULT_HU_PAD = -1000.0
DEFAULT_INPUT_SHAPE = (64, 64, 64)
DEFAULT_SIGMA_VOX = 2.5
DEFAULT_CROP_MM = (200.0, 160.0, 160.0)  # (Z,Y,X) mm around click in canonical space
DEFAULT_ROI_SIZES_MM = [25, 40, 60]


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def load_json(p: str) -> Any:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(d: Any, p: str) -> None:
    ensure_dir(os.path.dirname(p))
    with open(p, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)

def normalize_pid(pid: Any) -> Optional[str]:
    if pid is None:
        return None
    s = str(pid).strip()
    s = s.replace("\u200b", "").replace("\ufeff", "")
    if s.endswith(".0"):
        s = s[:-2]
    try:
        s = str(int(float(s)))
    except Exception:
        pass
    return s

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

def infer_points_space_auto(p: np.ndarray,
                            vol_raw_shape: Tuple[int,int,int],
                            vol_iso_shape: Tuple[int,int,int]) -> str:
    """Heuristic: decide whether point indices are in raw-DICOM grid or already in iso grid."""
    p = np.asarray(p, dtype=np.float32)
    in_raw = (0 <= p[0] < vol_raw_shape[0]) and (0 <= p[1] < vol_raw_shape[1]) and (0 <= p[2] < vol_raw_shape[2])
    in_iso = (0 <= p[0] < vol_iso_shape[0]) and (0 <= p[1] < vol_iso_shape[1]) and (0 <= p[2] < vol_iso_shape[2])
    if in_iso and not in_raw:
        return "iso"
    if in_raw and not in_iso:
        return "raw"
    # ambiguous: compare distance to the respective max dimension
    raw_max = float(max(vol_raw_shape))
    iso_max = float(max(vol_iso_shape))
    pm = float(np.max(p))
    if abs(pm - iso_max) < abs(pm - raw_max):
        return "iso"
    return "raw"

def detect_points_already_canonical(raw_json: Dict[str, Any]) -> bool:
    """Best-effort: detect if vw_point.json was created on LR-canonical (right-aligned) volume."""
    for k in ["flipped_lr", "lr_canonical", "is_canonical", "is_canon", "canonical", "canon"]:
        if k in raw_json:
            v = raw_json.get(k)
            if isinstance(v, bool) and v:
                return True
            if isinstance(v, (int, float)) and v != 0:
                return True
            if isinstance(v, str) and v.strip().lower() in ["1", "true", "yes", "y"]:
                return True
    for k in raw_json.keys():
        if "canon" in str(k).lower():
            return True
    return False


def as_vec3(v: Any) -> Optional[np.ndarray]:
    if v is None:
        return None
    if isinstance(v, np.ndarray) and v.size == 3:
        return v.astype(np.float32).reshape(3)
    if isinstance(v, (list, tuple)) and len(v) == 3:
        try:
            return np.asarray([float(x) for x in v], dtype=np.float32).reshape(3)
        except Exception:
            return None
    return None


# =========================
# DICOM loading / resample
# =========================

def load_dicom_series_to_hu_zyx(dicom_dir: str) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    if not os.path.isdir(dicom_dir):
        raise FileNotFoundError(f"DICOM dir not found: {dicom_dir}")
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dicom_dir)
    if not series_ids:
        raise RuntimeError(f"No DICOM series found in: {dicom_dir}")
    file_names = reader.GetGDCMSeriesFileNames(dicom_dir, series_ids[0])
    reader.SetFileNames(file_names)
    img = reader.Execute()
    sx, sy, sz = img.GetSpacing()  # SITK: (x,y,z)
    vol_zyx = sitk.GetArrayFromImage(img).astype(np.int16, copy=False)  # numpy: ZYX
    spacing_zyx = (float(sz), float(sy), float(sx))
    return vol_zyx, spacing_zyx

def resample_to_iso(vol_zyx: np.ndarray, spacing_zyx: Tuple[float, float, float], iso_mm: float) -> np.ndarray:
    sz, sy, sx = spacing_zyx
    zf = (sz / iso_mm, sy / iso_mm, sx / iso_mm)
    if all(abs(a - 1.0) < 1e-6 for a in zf):
        return vol_zyx.astype(np.float32)
    return zoom(vol_zyx.astype(np.float32), zf, order=1).astype(np.float32)

def scale_point_to_iso(p_zyx: np.ndarray, spacing_zyx: Tuple[float,float,float], iso_mm: float) -> np.ndarray:
    sz, sy, sx = spacing_zyx
    scale = np.array([sz/iso_mm, sy/iso_mm, sx/iso_mm], dtype=np.float32)
    return p_zyx.astype(np.float32) * scale


# =========================
# LR canonicalization
# =========================

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


# =========================
# vw_point.json loading
# =========================

def pick_first_present_vec3(d: Dict[str, Any], keys: List[str]) -> Tuple[Optional[np.ndarray], Optional[str]]:
    for k in keys:
        if k in d:
            v = as_vec3(d.get(k))
            if v is not None:
                return v, k
    return None, None


def find_first_vec3_in_json(raw: Any) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Best-effort fallback: find a plausible 3-vector in arbitrary vw_point.json schemas.
    We only accept candidates whose *key/path* suggests a landmark/point/center.
    Returns (vec3_zyx_or_xyz_as_numpy, path_str)
    """
    KEY_OK = re.compile(r"(vw|click|snap|point|center|coord|landmark|zyx|xyz|eac|coch|vestib|stapes|oval|round)", re.IGNORECASE)

    def is_num(x: Any) -> bool:
        return isinstance(x, (int, float, np.integer, np.floating)) and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))

    def from_dict_xyz(d: Dict[str, Any]) -> Optional[np.ndarray]:
        # Accept {"x":..,"y":..,"z":..} or {"z":..,"y":..,"x":..}
        if all(k in d for k in ("x","y","z")) and all(is_num(d[k]) for k in ("x","y","z")):
            return np.array([d["z"], d["y"], d["x"]], dtype=np.float32)  # to zyx
        if all(k in d for k in ("z","y","x")) and all(is_num(d[k]) for k in ("z","y","x")):
            return np.array([d["z"], d["y"], d["x"]], dtype=np.float32)
        return None

    def from_obj(v: Any) -> Optional[np.ndarray]:
        # Accept list/tuple len3
        if isinstance(v, (list, tuple)) and len(v) == 3 and all(is_num(x) for x in v):
            return np.array(v, dtype=np.float32)
        # Accept dict forms
        if isinstance(v, dict):
            # common nested fields
            for kk in ("zyx","xyz","vec3","point","center","coord","coords","p"):
                if kk in v:
                    vv = from_obj(v[kk])
                    if vv is not None:
                        return vv
            vv = from_dict_xyz(v)
            if vv is not None:
                return vv
        return None

    def walk(obj: Any, path: str = ""):
        # Yield (vec, path) candidates
        if isinstance(obj, dict):
            for k, v in obj.items():
                p = f"{path}.{k}" if path else str(k)
                vec = from_obj(v)
                if vec is not None and KEY_OK.search(p):
                    yield vec, p
                # recurse
                yield from walk(v, p)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                p = f"{path}[{i}]"
                vec = from_obj(v)
                if vec is not None and KEY_OK.search(p):
                    yield vec, p
                yield from walk(v, p)

    for vec, p in walk(raw, ""):
        # vec may be xyz or zyx; we cannot always know.
        # We keep as-is; downstream treats it as zyx unless json coord_space indicates otherwise.
        return vec, p
    return None, None

def load_points_from_vw_json(point_json_path: str,
                             click_keys: Optional[List[str]] = None,
                             gt_keys: Optional[List[str]] = None
                             ) -> Dict[str, Any]:
    raw = load_json(point_json_path)
    if not isinstance(raw, dict):
        raise ValueError(f"vw_point.json must be dict: {point_json_path}")

    if click_keys is None:
        click_keys = [
            "vw_click_canon_full_zyx", "vw_snapped_canon_full_zyx",
            "vw_snapped_full_zyx", "vw_click_full_zyx",
            "vw_snapped_full_zyx_f", "vw_click_full_zyx_f",
            "vw_snapped_zyx", "vw_click_zyx",
            "vw_snapped_zyx_f", "vw_click_zyx_f",
        ]
    if gt_keys is None:
        gt_keys = [
            "gt_center_full_zyx", "center_full_zyx",
            "gt_center_full_zyx_f", "center_full_zyx_f",
            # fallback to click (your current convention)
            *click_keys,
        ]

    click, ck = pick_first_present_vec3(raw, click_keys)
    gt, gk = pick_first_present_vec3(raw, gt_keys)

    # fallback: schema-agnostic vec3 search
    if click is None and gt is None:
        auto, apath = find_first_vec3_in_json(raw)
        if auto is not None:
            click, gt = auto, auto
            ck, gk = apath, apath
        else:
            raise KeyError(f"No usable vec3 found in vw_point.json: {point_json_path}")
    if click is None:
        click = gt.copy(); ck = gk
    if gt is None:
        gt = click.copy(); gk = ck

    return {
        "click_full_zyx": click.astype(np.float32),
        "gt_full_zyx": gt.astype(np.float32),
        "click_key_used": ck,
        "gt_key_used": gk,
        # Whether the selected key is already LR-canonical (right-aligned)
        "click_is_canon": (ck is not None and "canon" in str(ck).lower()),
        "gt_is_canon": (gk is not None and "canon" in str(gk).lower()),
        # Helpful hints from json (may be absent)
        "coord_space": raw.get("coord_space") if isinstance(raw, dict) else None,
        "canon_flipped_lr": raw.get("canon_flipped_lr") if isinstance(raw, dict) else None,
        "raw": raw,
    }


# =========================
# Crop (exclusive end), resize, coordinate maps
# =========================

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


# =========================
# Model / target / softargmax
# =========================

class Tiny3DUNet(nn.Module):
    def __init__(self, base=16):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv3d(1, base, 3, padding=1), nn.ReLU(inplace=True),
                                  nn.Conv3d(base, base, 3, padding=1), nn.ReLU(inplace=True))
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = nn.Sequential(nn.Conv3d(base, base*2, 3, padding=1), nn.ReLU(inplace=True),
                                  nn.Conv3d(base*2, base*2, 3, padding=1), nn.ReLU(inplace=True))
        self.pool2 = nn.MaxPool3d(2)
        self.bott = nn.Sequential(nn.Conv3d(base*2, base*4, 3, padding=1), nn.ReLU(inplace=True),
                                  nn.Conv3d(base*4, base*4, 3, padding=1), nn.ReLU(inplace=True))
        self.up2 = nn.ConvTranspose3d(base*4, base*2, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv3d(base*4, base*2, 3, padding=1), nn.ReLU(inplace=True),
                                  nn.Conv3d(base*2, base*2, 3, padding=1), nn.ReLU(inplace=True))
        self.up1 = nn.ConvTranspose3d(base*2, base, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv3d(base*2, base, 3, padding=1), nn.ReLU(inplace=True),
                                  nn.Conv3d(base, base, 3, padding=1), nn.ReLU(inplace=True))
        self.out = nn.Conv3d(base, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b  = self.bott(self.pool2(e2))
        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        return self.out(d1)  # logits

def gaussian_heatmap(shape: Tuple[int,int,int], center_zyx: np.ndarray, sigma: float) -> np.ndarray:
    z, y, x = shape
    cz, cy, cx = [float(v) for v in center_zyx]
    zz = np.arange(z, dtype=np.float32)[:, None, None]
    yy = np.arange(y, dtype=np.float32)[None, :, None]
    xx = np.arange(x, dtype=np.float32)[None, None, :]
    d2 = (zz - cz)**2 + (yy - cy)**2 + (xx - cx)**2
    hm = np.exp(-0.5 * d2 / max(1e-6, sigma**2)).astype(np.float32)
    s = float(hm.sum())
    if s > 0:
        hm /= s
    return hm

def soft_argmax_zyx(p: np.ndarray) -> np.ndarray:
    """p is probability distribution (Z,Y,X) sum=1."""
    z, y, x = p.shape
    zz = np.arange(z, dtype=np.float32)[:, None, None]
    yy = np.arange(y, dtype=np.float32)[None, :, None]
    xx = np.arange(x, dtype=np.float32)[None, None, :]
    cz = float((p * zz).sum())
    cy = float((p * yy).sum())
    cx = float((p * xx).sum())
    return np.array([cz, cy, cx], dtype=np.float32)

def confidence_from_logits(logits: np.ndarray) -> Dict[str, Any]:
    a = logits.reshape(-1).astype(np.float64)
    a = a - a.max()
    expa = np.exp(a)
    p = expa / (expa.sum() + 1e-12)
    ent = -float(np.sum(p * np.log(p + 1e-12)))
    ent_norm = ent / math.log(len(p) + 1e-12)
    part = np.partition(p, -2)
    top2v = float(part[-2])
    top1v = float(part[-1])
    ratio = float(top1v / (top2v + 1e-12))
    return {
        "top1_value": top1v,
        "top2_value": top2v,
        "top1_top2_ratio": ratio,
        "softmax_entropy_norm": float(ent_norm),
    }


# =========================
# QA (optional)
# =========================

def save_qa_montage(vol_zyx: np.ndarray, gt_zyx: np.ndarray, pr_zyx: np.ndarray, click_zyx: Optional[np.ndarray], out_png: str, title: str = "") -> None:
    if not HAS_MPL:
        return
    vol = vol_zyx
    gt = np.asarray(gt_zyx, dtype=np.float32)
    pr = np.asarray(pr_zyx, dtype=np.float32)

    z = int(round(gt[0])); y = int(round(gt[1])); x = int(round(gt[2]))
    z = max(0, min(vol.shape[0]-1, z))
    y = max(0, min(vol.shape[1]-1, y))
    x = max(0, min(vol.shape[2]-1, x))

    fig = plt.figure(figsize=(10, 6))
    fig.suptitle(title, fontsize=10)

    ax1 = fig.add_subplot(2,3,1); ax1.set_title("Axial@GT(z)")
    ax1.imshow(vol[z,:,:], cmap="gray"); ax1.scatter([gt[2]],[gt[1]], c="cyan", s=30, marker="x")
    if click_zyx is not None:
        ck=np.asarray(click_zyx,dtype=np.float32); ax1.scatter([ck[2]],[ck[1]], c="lime", s=20, marker="+")
    ax1.scatter([pr[2]],[pr[1]], c="orange", s=30, marker="o"); ax1.axis("off")

    ax2 = fig.add_subplot(2,3,2); ax2.set_title("Coronal@GT(y)")
    ax2.imshow(vol[:,y,:], cmap="gray"); ax2.scatter([gt[2]],[gt[0]], c="cyan", s=30, marker="x")
    if click_zyx is not None:
        ck=np.asarray(click_zyx,dtype=np.float32); ax2.scatter([ck[2]],[ck[0]], c="lime", s=20, marker="+")
    ax2.scatter([pr[2]],[pr[0]], c="orange", s=30, marker="o"); ax2.axis("off")

    ax3 = fig.add_subplot(2,3,3); ax3.set_title("Sagittal@GT(x)")
    ax3.imshow(vol[:,:,x], cmap="gray"); ax3.scatter([gt[1]],[gt[0]], c="cyan", s=30, marker="x")
    if click_zyx is not None:
        ck=np.asarray(click_zyx,dtype=np.float32); ax3.scatter([ck[1]],[ck[0]], c="lime", s=20, marker="+")
    ax3.scatter([pr[1]],[pr[0]], c="orange", s=30, marker="o"); ax3.axis("off")

    z2 = int(round(pr[0])); y2 = int(round(pr[1])); x2 = int(round(pr[2]))
    z2 = max(0, min(vol.shape[0]-1, z2))
    y2 = max(0, min(vol.shape[1]-1, y2))
    x2 = max(0, min(vol.shape[2]-1, x2))

    ax4 = fig.add_subplot(2,3,4); ax4.set_title("Axial@PR(z)")
    ax4.imshow(vol[z2,:,:], cmap="gray"); ax4.scatter([gt[2]],[gt[1]], c="cyan", s=30, marker="x")
    if click_zyx is not None:
        ck=np.asarray(click_zyx,dtype=np.float32); ax4.scatter([ck[2]],[ck[1]], c="lime", s=20, marker="+")
    ax4.scatter([pr[2]],[pr[1]], c="orange", s=30, marker="o"); ax4.axis("off")

    ax5 = fig.add_subplot(2,3,5); ax5.set_title("Coronal@PR(y)")
    ax5.imshow(vol[:,y2,:], cmap="gray"); ax5.scatter([gt[2]],[gt[0]], c="cyan", s=30, marker="x")
    if click_zyx is not None:
        ck=np.asarray(click_zyx,dtype=np.float32); ax5.scatter([ck[2]],[ck[0]], c="lime", s=20, marker="+")
    ax5.scatter([pr[2]],[pr[0]], c="orange", s=30, marker="o"); ax5.axis("off")

    ax6 = fig.add_subplot(2,3,6); ax6.set_title("Sagittal@PR(x)")
    ax6.imshow(vol[:,:,x2], cmap="gray"); ax6.scatter([gt[1]],[gt[0]], c="cyan", s=30, marker="x")
    if click_zyx is not None:
        ck=np.asarray(click_zyx,dtype=np.float32); ax6.scatter([ck[1]],[ck[0]], c="lime", s=20, marker="+")
    ax6.scatter([pr[1]],[pr[0]], c="orange", s=30, marker="o"); ax6.axis("off")

    ensure_dir(os.path.dirname(out_png))
    fig.tight_layout(rect=[0,0,1,0.95])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# =========================
# ROI extraction
# =========================

def extract_spherical_roi(vol_zyx: np.ndarray, center_zyx: np.ndarray, radius_mm: float, iso_mm: float,
                          pad_hu: float = DEFAULT_HU_PAD) -> np.ndarray:
    r = int(round(radius_mm / iso_mm))
    cz, cy, cx = [int(round(v)) for v in center_zyx]
    Z, Y, X = vol_zyx.shape
    z0, z1 = cz - r, cz + r + 1
    y0, y1 = cy - r, cy + r + 1
    x0, x1 = cx - r, cx + r + 1

    pad_before = [max(0, -z0), max(0, -y0), max(0, -x0)]
    pad_after  = [max(0, z1 - Z), max(0, y1 - Y), max(0, x1 - X)]

    z0c, y0c, x0c = max(0, z0), max(0, y0), max(0, x0)
    z1c, y1c, x1c = min(Z, z1), min(Y, y1), min(X, x1)

    crop = vol_zyx[z0c:z1c, y0c:y1c, x0c:x1c].astype(np.float32, copy=False)
    if any(pad_before) or any(pad_after):
        crop = np.pad(
            crop,
            ((pad_before[0], pad_after[0]), (pad_before[1], pad_after[1]), (pad_before[2], pad_after[2])),
            mode="constant",
            constant_values=float(pad_hu),
        )
    return crop.astype(np.float32)


# =========================
# Dataset cases
# =========================

@dataclass
class CaseInfo:
    pid: str
    dicom_dir: str
    side_rl: Optional[str]
    fold: int
    point_json: str

def load_cases_from_csv(gt_csv: str, point_json_dir: str, root: str, dicom_root: str) -> List[CaseInfo]:
    df = pd.read_csv(gt_csv)
    cols_l = {c.lower(): c for c in df.columns}

    def col_optional(*names: str) -> Optional[str]:
        for n in names:
            if n.lower() in cols_l:
                return cols_l[n.lower()]
        return None

    def col_required(name: str) -> str:
        c = col_optional(name)
        if c is None:
            raise KeyError(f"Missing column '{name}' in {gt_csv}. Found: {list(df.columns)}")
        return c

    id_col = col_required("id")
    fold_col = col_required("fold")
    side_col = col_optional("side")

    dicom_col = col_optional("dicom_dir", "dicom_path", "ct_dir", "ct_path", "series_dir")
    manifest_col = col_optional("roi_manifest_path", "manifest_path")

    def infer_dicom_dir_from_row(row: pd.Series, pid: str) -> str:
        if dicom_col is not None:
            v = str(row[dicom_col]).strip()
            if v and v.lower() != "nan" and os.path.isdir(v):
                return v
        if manifest_col is not None:
            mpath = str(row[manifest_col]).strip()
            if mpath and mpath.lower() != "nan" and os.path.exists(mpath):
                try:
                    mj = load_json(mpath)
                    if isinstance(mj, dict):
                        for k in ["dicom_dir", "dicom_path", "ct_dir", "ct_path", "series_dir", "dicomFolder", "dicom"]:
                            if k in mj and isinstance(mj[k], str) and os.path.isdir(mj[k]):
                                return mj[k]
                except Exception:
                    pass
        cand = os.path.join(root, dicom_root, pid)
        if os.path.isdir(cand):
            return cand
        raise FileNotFoundError(
            f"Cannot infer dicom_dir for id={pid}. Tried manifest and {cand}. "
            f"Fix: ensure {dicom_root}/<id> exists or add dicom_dir column or write dicom_dir into roi_manifest_path json."
        )

    cases: List[CaseInfo] = []
    for _, r in df.iterrows():
        pid = normalize_pid(r[id_col])
        if not pid:
            continue
        try:
            fold = int(r[fold_col])
        except Exception:
            continue
        side_rl = normalize_side_value(r[side_col]) if side_col else None
        dicom_dir = infer_dicom_dir_from_row(r, pid)
        pj = os.path.join(point_json_dir, f"{pid}_vw_point.json")
        cases.append(CaseInfo(pid=pid, dicom_dir=dicom_dir, side_rl=side_rl, fold=fold, point_json=pj))
    return cases


# =========================
# Preprocess one case
# =========================

def prepare_case_inputs(case: CaseInfo,
                        iso_mm: float,
                        crop_size_mm: Tuple[float,float,float],
                        input_shape: Tuple[int,int,int],
                        click_keys: Optional[List[str]],
                        gt_keys: Optional[List[str]],
                        vw_points_space: str,
                        crop_y_front_mm: Optional[float] = None,
                        crop_y_back_mm: Optional[float] = None,
                        anchor_method: str = "bbox_frac",
                        anchor_x_frac: float = 0.70,
                        anchor_y_shift_mm: float = 0.0,
                        anchor_z_shift_mm: float = 0.0,
                        anterior_mask_y_mm: float = 0.0,
                        anterior_mask_alpha: float = 0.2,
                        anterior_mask_ramp_mm: float = 10.0,
                        anterior_mask_mode: str = "attenuate",
                        ) -> Dict[str, Any]:
    """Preprocess one case in a **CT-only (no click)** manner.

    - GT can be loaded from vw_point.json for training/eval, but MUST NOT affect crop/mask.
    - Crop center is estimated from CT (after LR canonicalization) using a deterministic anchor.
    - Optional conditional anterior mask is applied in the canonical crop cube (anterior assumed to be -Y).
    """
    if not os.path.exists(case.point_json):
        raise FileNotFoundError(f"Point JSON not found: {case.point_json}")

    # Load + resample to ISO
    vol_raw_zyx, spacing_zyx = load_dicom_series_to_hu_zyx(case.dicom_dir)
    vol_iso_zyx = resample_to_iso(vol_raw_zyx, spacing_zyx, iso_mm)

    # ---- Load GT (ONLY for supervision / evaluation; NOT for preprocessing decisions) ----
    pts = load_points_from_vw_json(case.point_json, click_keys=click_keys, gt_keys=gt_keys)
    rawj = pts.get("raw", {})

    # Detect whether vw_point.json contains coordinates already in LR-canonical space
    already_canon = bool(pts.get("click_is_canon", False))
    if not already_canon:
        try:
            already_canon = bool(detect_points_already_canonical(rawj))
        except Exception:
            already_canon = False

    gt_raw = pts["gt_full_zyx"]  # as stored in vw_point.json (unknown space)

    # Decide coordinate space for vw_point.json
    points_space = vw_points_space

    # Determine whether the stored point is already on ISO grid
    coord_space = pts.get("coord_space")
    gt_is_canon = bool(pts.get("click_is_canon", False))

    if points_space == "auto":
        if isinstance(coord_space, str) and ("iso" in coord_space.lower() or "norm" in coord_space.lower()):
            base = "iso"
        else:
            base = infer_points_space_auto(gt_raw, vol_raw_zyx.shape, vol_iso_zyx.shape)

        if base == "iso":
            points_space = "iso_canon" if gt_is_canon else "iso"
        else:
            points_space = "raw"

    # Convert GT to ISO grid (voxel indices of vol_iso_zyx)
    if points_space == "raw":
        gt_iso = scale_point_to_iso(gt_raw, spacing_zyx, iso_mm)
    else:
        gt_iso = np.asarray(gt_raw, dtype=np.float32)

    # ---- LR canonicalize volume (OK to use side from df; clinical covariate) ----
    flipped = False
    if case.side_rl == "L":
        flipped = True
        vol_canon = vol_iso_zyx[:, :, ::-1].copy()
        X = vol_iso_zyx.shape[2]
        gt_c = gt_iso if (points_space == "iso_canon") else flip_x_point(gt_iso, X)
    else:
        vol_canon = vol_iso_zyx
        gt_c = gt_iso

    gt_c = np.asarray(gt_c, dtype=np.float32)

    # ---- CT-only anchor estimation (deterministic) ----
    anchor_c = estimate_anchor_center_ctonly(
        vol_canon_zyx=vol_canon,
        iso_mm=iso_mm,
        method=anchor_method,
        x_frac=float(anchor_x_frac),
        y_shift_mm=float(anchor_y_shift_mm),
        z_shift_mm=float(anchor_z_shift_mm),
    )

    # ---- Crop around anchor (NOT around GT/click) ----
    vol_crop, origin_full, pad_before = crop_around_center(
        vol_canon, anchor_c, iso_mm, crop_size_mm, pad_hu=DEFAULT_HU_PAD,
        crop_y_front_mm=crop_y_front_mm, crop_y_back_mm=crop_y_back_mm)

    # ---- Conditional anterior mask in canonical crop cube (anterior assumed to be -Y) ----
    if float(anterior_mask_y_mm) > 0.0:
        vol_crop = apply_conditional_anterior_mask(
            vol_crop_zyx=vol_crop,
            iso_mm=float(iso_mm),
            y_from_center_mm=float(anterior_mask_y_mm),
            alpha=float(anterior_mask_alpha),
            ramp_mm=float(anterior_mask_ramp_mm),
            pad_hu=float(DEFAULT_HU_PAD),
            mode=str(anterior_mask_mode),
        )

    # ---- Resize crop to model input ----
    vol_in, scale_crop_to_in, shift_in = resize_crop_to_input(vol_crop, input_shape)

    # Map GT into crop space (float) then into input space (float)
    gt_crop = gt_c - origin_full.astype(np.float32)
    gt_in = map_point_crop_to_input(gt_crop, scale_crop_to_in, shift_in, input_shape)

    return {
        "vol_full_canon_zyx": vol_canon.astype(np.float32),
        "vol_crop_zyx": vol_crop.astype(np.float32),
        "vol_in_zyx": vol_in.astype(np.float32),  # (Z,Y,X) on input grid, before clip/normalize
        "origin_full_zyx": origin_full.astype(np.int32),
        "pad_before_zyx": pad_before.astype(np.int32),
        "scale_crop_to_in": scale_crop_to_in.astype(np.float32),
        "shift_in": shift_in.astype(np.float32),
        "gt_canon_zyx": gt_c.astype(np.float32),
        "gt_crop_zyx": gt_crop.astype(np.float32),
        "gt_in_zyx": gt_in.astype(np.float32),
        "anchor_canon_zyx": anchor_c.astype(np.float32),
        "flipped_lr": bool(flipped),
        "point_keys": {
            "gt_key_used": pts.get("gt_key_used"),
            "points_space_used": points_space,
            "points_space_requested": vw_points_space,
        },
        "ctonly": {
            "anchor_method": str(anchor_method),
            "anchor_x_frac": float(anchor_x_frac),
            "anchor_y_shift_mm": float(anchor_y_shift_mm),
            "anchor_z_shift_mm": float(anchor_z_shift_mm),
            "anterior_mask_y_mm": float(anterior_mask_y_mm),
            "anterior_mask_alpha": float(anterior_mask_alpha),
            "anterior_mask_ramp_mm": float(anterior_mask_ramp_mm),
            "anterior_mask_mode": str(anterior_mask_mode),
        }
    }

def train_one_fold(train_cases: List[CaseInfo],
                   dev_cases: List[CaseInfo],
                   iso_mm: float,
                   crop_size_mm: Tuple[float,float,float],
                   input_shape: Tuple[int,int,int],
                   click_keys: Optional[List[str]],
                   sigma_vox: float,
                   clip_high: float,
                   crop_y_front_mm: Optional[float],
                   crop_y_back_mm: Optional[float],
                   gt_keys: Optional[List[str]],
                   device: str,
                   vw_points_space: str,
                   epochs: int = 10,
                   batch_size: int = 2,
                   lr: float = 1e-3,
                   seed: int = 1337,
                   # CT-only anchor
                   anchor_method: str = "bbox_frac",
                   anchor_x_frac: float = 0.70,
                   anchor_y_shift_mm: float = 0.0,
                   anchor_z_shift_mm: float = 0.0,
                   # Anterior mask (anterior=-Y)
                   anterior_mask_y_mm: float = 0.0,
                   anterior_mask_alpha: float = 0.2,
                   anterior_mask_ramp_mm: float = 10.0,
                   anterior_mask_mode: str = "attenuate",
                   
                   best_epoch_metric: str = "val_kl",
                   best_epoch_patience: int = 0,
                   best_epoch_min_delta: float = 0.0,
                   best_epoch_ema: float = 0.0,
                   ) -> Tuple[nn.Module, int]:
    """Train on one CV fold with **CT-only preprocessing** (no click-based crop/mask)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = Tiny3DUNet(base=16).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def make_batch(cases_batch: List[CaseInfo], return_gt_in: bool = False):
        xs = []
        ys = []
        gtins = []
        for c in cases_batch:
            inp = prepare_case_inputs(
                case=c, iso_mm=iso_mm, crop_size_mm=crop_size_mm, input_shape=input_shape,
                click_keys=click_keys,
                gt_keys=gt_keys, vw_points_space=vw_points_space,
                crop_y_front_mm=crop_y_front_mm, crop_y_back_mm=crop_y_back_mm,
                anchor_method=anchor_method, anchor_x_frac=anchor_x_frac, anchor_y_shift_mm=anchor_y_shift_mm, anchor_z_shift_mm=anchor_z_shift_mm,
                anterior_mask_y_mm=anterior_mask_y_mm, anterior_mask_alpha=anterior_mask_alpha,
                anterior_mask_ramp_mm=anterior_mask_ramp_mm, anterior_mask_mode=anterior_mask_mode,
            )
            x = inp["vol_in_zyx"][None, ...]  # (1,Z,Y,X)
            x = np.clip(x, -1024.0, float(clip_high))
            x = (x - (-1024.0)) / (float(clip_high) - (-1024.0) + 1e-6)
            x = (x * 2.0) - 1.0
            xs.append(x.astype(np.float32))

            # Target distribution in input grid (Gaussian around gt_in)
            gt_in = inp["gt_in_zyx"].astype(np.float32)
            if return_gt_in:
                gtins.append(gt_in)
            y = gaussian_heatmap(input_shape, gt_in, float(sigma_vox)).astype(np.float32)
            ys.append(y[None, ...])  # (1,Z,Y,X)

        X = torch.from_numpy(np.stack(xs, axis=0)).to(device)  # (B,1,Z,Y,X)
        Y = torch.from_numpy(np.stack(ys, axis=0)).to(device)  # (B,1,Z,Y,X)
        return X, Y, (np.stack(gtins, axis=0) if return_gt_in else None)

    best_val = float("inf")
    best_state = None
    best_epoch = -1
    bad_epochs = 0

    metric_s = None  # optional smoothed metric for best-epoch selection

    for ep in range(int(epochs)):
        model.train()
        random.shuffle(train_cases)
        losses = []
        for i in range(0, len(train_cases), int(batch_size)):
            batch = train_cases[i:i+int(batch_size)]
            Xb, Yb, _ = make_batch(batch)
            opt.zero_grad(set_to_none=True)
            logits = model(Xb)  # (B,1,Z,Y,X)

            # KL( target || pred )
            logp = torch.log_softmax(logits.flatten(2), dim=-1).view_as(logits)
            # normalize target
            yt = Yb / (Yb.sum(dim=(2,3,4), keepdim=True) + 1e-12)
            loss = (yt * (torch.log(yt + 1e-12) - logp)).sum(dim=(2,3,4)).mean()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))

        # dev monitoring (NO-LEAK: dev is drawn only from non-val folds by caller)
        model.eval()
        vlosses = []
        vdist = []
        if dev_cases:
            want_dist = (best_epoch_metric == "dev_dist")
            vz = float(crop_size_mm[0]) / float(input_shape[0])
            vy = float(crop_size_mm[1]) / float(input_shape[1])
            vx = float(crop_size_mm[2]) / float(input_shape[2])
            Z, Y, X = int(input_shape[0]), int(input_shape[1]), int(input_shape[2])
            for i in range(0, len(dev_cases), int(batch_size)):
                batch = dev_cases[i:i+int(batch_size)]
                Xb, Yb, gt_inb = make_batch(batch, return_gt_in=want_dist)
                with torch.no_grad():
                    logits = model(Xb)
                    logp = torch.log_softmax(logits.flatten(2), dim=-1).view_as(logits)
                    yt = Yb / (Yb.sum(dim=(2,3,4), keepdim=True) + 1e-12)
                    vloss = (yt * (torch.log(yt + 1e-12) - logp)).sum(dim=(2,3,4)).mean()
                vlosses.append(float(vloss.detach().cpu().item()))

                if want_dist and gt_inb is not None:
                    flat = logits.reshape(logits.shape[0], -1)
                    idxs = torch.argmax(flat, dim=1).detach().cpu().numpy().astype(np.int64)
                    zz = idxs // (Y * X)
                    rem = idxs % (Y * X)
                    yy = rem // X
                    xx = rem % X
                    pr = np.stack([zz, yy, xx], axis=1).astype(np.float32)
                    gt = gt_inb.astype(np.float32)
                    dz = (pr[:, 0] - gt[:, 0]) * vz
                    dy = (pr[:, 1] - gt[:, 1]) * vy
                    dx = (pr[:, 2] - gt[:, 2]) * vx
                    dist = np.sqrt(dz*dz + dy*dy + dx*dx)
                    vdist.extend(dist.tolist())

        tr = float(np.mean(losses)) if losses else float("nan")
        va = float(np.mean(vlosses)) if vlosses else float("nan")
        vd = float(np.mean(vdist)) if vdist else float("nan")

        if best_epoch_metric == "dev_dist" and vd == vd:
            metric = vd
            metric_name = "dev_dist_mm"
        else:
            metric = va
            metric_name = "val_KL"

        # Optional EMA smoothing for more stable best-epoch selection
        if best_epoch_ema and float(best_epoch_ema) > 0.0 and float(best_epoch_ema) < 1.0 and metric == metric:
            if metric_s is None:
                metric_s = float(metric)
            else:
                metric_s = float(best_epoch_ema) * float(metric_s) + (1.0 - float(best_epoch_ema)) * float(metric)
        else:
            metric_s = float(metric) if metric == metric else float('nan')

        extra = f" metric_s={metric_s:.4f}" if (best_epoch_ema and float(best_epoch_ema)>0.0 and float(best_epoch_ema)<1.0 and metric_s==metric_s) else ""
        print(f"[epoch {ep:03d}] train_KL={tr:.6f}  val_KL={va:.6f}  dev_dist_mm={(vd if vd==vd else float('nan')):.3f}  best_by={metric_name}{extra}")

        if metric_s == metric_s and (metric_s < best_val - float(best_epoch_min_delta)):
            best_val = float(metric_s)
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if best_epoch_patience and bad_epochs >= int(best_epoch_patience):
            print(f"[early_stop] patience={best_epoch_patience} reached at epoch={ep}, best_epoch={best_epoch} best_metric={best_val} (min_delta={best_epoch_min_delta}, ema={best_epoch_ema})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, int(best_epoch)

def infer_center_for_case(model: nn.Module,
                         case: CaseInfo,
                         iso_mm: float,
                         crop_size_mm: Tuple[float,float,float],
                         input_shape: Tuple[int,int,int],
                         click_keys: Optional[List[str]],
                         gt_keys: Optional[List[str]],
                         device: str,
                         vw_points_space: str,
                         clip_high: float = 3071.0,
                         crop_y_front_mm: Optional[float] = None,
                         crop_y_back_mm: Optional[float] = None,
                         # CT-only anchor
                         anchor_method: str = "bbox_frac",
                         anchor_x_frac: float = 0.70,
                         anchor_y_shift_mm: float = 0.0,
                         anchor_z_shift_mm: float = 0.0,
                         # Anterior mask in canonical cube (anterior = -Y)
                         anterior_mask_y_mm: float = 0.0,
                         anterior_mask_alpha: float = 0.2,
                         anterior_mask_ramp_mm: float = 10.0,
                         anterior_mask_mode: str = "attenuate",
                         # Metal-aware attenuation (CT-only)
                         metal_thr: float = 3000.0,
                         metal_alpha: float = 0.90,
                         metal_attenuate_on_guard: bool = False,
                         metal_attenuate_always: bool = False,
                         # Uncertainty guard (CT-only)
                         guard_top1_top2_lt: float = 1.05,
                         guard_entropy_gt: float = 0.62,
                         ) -> Dict[str, Any]:
    model.eval()
    inp = prepare_case_inputs(
        case=case, iso_mm=iso_mm, crop_size_mm=crop_size_mm, input_shape=input_shape,
        click_keys=click_keys,
                gt_keys=gt_keys, vw_points_space=vw_points_space,
        crop_y_front_mm=crop_y_front_mm, crop_y_back_mm=crop_y_back_mm,
        anchor_method=anchor_method, anchor_x_frac=anchor_x_frac, anchor_y_shift_mm=anchor_y_shift_mm, anchor_z_shift_mm=anchor_z_shift_mm,
        anterior_mask_y_mm=anterior_mask_y_mm, anterior_mask_alpha=anterior_mask_alpha, anterior_mask_ramp_mm=anterior_mask_ramp_mm, anterior_mask_mode=anterior_mask_mode,
    )

    x = inp["vol_in_zyx"][None, None, ...].astype(np.float32)
    x = np.clip(x, -1024.0, float(clip_high))
    x = (x - (-1024.0)) / (float(clip_high) - (-1024.0) + 1e-6)
    x = (x * 2.0) - 1.0

    X = torch.from_numpy(x).to(device)
    with torch.no_grad():
        logits = model(X)[0,0].detach().cpu().numpy().astype(np.float32)  # (Z,Y,X)
        # probability distribution over voxels
        a = logits.reshape(-1).astype(np.float64)
        a = a - a.max()
        expa = np.exp(a)
        p = (expa / (expa.sum() + 1e-12)).astype(np.float32).reshape(input_shape)
        pred_in = soft_argmax_zyx(p)  # float ZYX in input space

    # input -> crop -> full(canon)
    pred_crop = map_point_input_to_crop(pred_in, inp["scale_crop_to_in"], inp["shift_in"])
    pred_full_canon = pred_crop + inp["origin_full_zyx"].astype(np.float32)

    gt_canon = inp["gt_canon_zyx"]

    dz_mm = float((pred_full_canon[0] - gt_canon[0]) * iso_mm)
    dy_mm = float((pred_full_canon[1] - gt_canon[1]) * iso_mm)
    dx_mm = float((pred_full_canon[2] - gt_canon[2]) * iso_mm)
    dist_mm = float(math.sqrt(dz_mm*dz_mm + dy_mm*dy_mm + dx_mm*dx_mm))

    conf = confidence_from_logits(logits)

    # ---------------------------------
    # CT-only guard & metal-aware attenuation
    # ---------------------------------
    low_conf = (conf.get("top1_top2_ratio", 999.0) < float(guard_top1_top2_lt)) or (conf.get("softmax_entropy_norm", 0.0) > float(guard_entropy_gt))
    guard_triggered = bool(low_conf)

    used_metal_attenuation = False
    if (metal_attenuate_always or (metal_attenuate_on_guard and guard_triggered)):
        try:
            vol_in_hu = inp["vol_in_zyx"].astype(np.float32)  # (Z,Y,X) on input grid, before clip/normalize
            metal_mask = (vol_in_hu > float(metal_thr)).astype(np.float32)
            p2 = p * (1.0 - float(metal_alpha) * metal_mask)
            s = float(p2.sum())
            if s > 1e-12:
                p2 = (p2 / s).astype(np.float32)
                pred_in2 = soft_argmax_zyx(p2)
                pred_crop2 = map_point_input_to_crop(pred_in2, inp["scale_crop_to_in"], inp["shift_in"])
                pred_full_canon2 = pred_crop2 + inp["origin_full_zyx"].astype(np.float32)

                pred_in = pred_in2
                pred_crop = pred_crop2
                pred_full_canon = pred_full_canon2

                dz_mm = float((pred_full_canon[0] - gt_canon[0]) * iso_mm)
                dy_mm = float((pred_full_canon[1] - gt_canon[1]) * iso_mm)
                dx_mm = float((pred_full_canon[2] - gt_canon[2]) * iso_mm)
                dist_mm = float(math.sqrt(dz_mm*dz_mm + dy_mm*dy_mm + dx_mm*dx_mm))
                used_metal_attenuation = True
        except Exception:
            used_metal_attenuation = False

    return {
        "inp": inp,
        "pred_center_full_canon_f": pred_full_canon.astype(np.float32),
        "gt_center_full_canon_f": gt_canon.astype(np.float32),
        "eval": {"dist_mm": dist_mm, "dz_mm": dz_mm, "dy_mm": dy_mm, "dx_mm": dx_mm},
        "confidence": conf,
        "guards": {
            "low_conf": bool(low_conf),
            "triggered": bool(guard_triggered),
            "used_metal_attenuation": bool(used_metal_attenuation),
        },
    }

def eval_root(root: str, iso_mm: float, out_csv: str) -> None:
    pred_files = glob.glob(os.path.join(root, "**", "*_pred_center.json"), recursive=True)
    if len(pred_files) == 0:
        raise SystemExit(f"No *_pred_center.json found under: {root}")

    fieldnames = [
        "patient_id","file","status","dist_mm","dz_mm","dy_mm","dx_mm",
        "top1_value","top2_value","top1_top2_ratio","softmax_entropy_norm","qa_png"
    ]
    rows = []
    for fp in pred_files:
        try:
            d = load_json(fp)
        except Exception:
            continue
        pid = d.get("patient_id") or os.path.basename(fp).split("_pred_center.json")[0]
        gt = d.get("gt_center_zyx") or d.get("gt_center_zyx_canon_f")
        pr = d.get("pred_center_zyx") or d.get("pred_center_zyx_canon_f")
        if not (isinstance(gt,(list,tuple)) and len(gt)==3 and isinstance(pr,(list,tuple)) and len(pr)==3):
            rows.append({"patient_id":pid,"file":fp,"status":"missing_keys","dist_mm":"","dz_mm":"","dy_mm":"","dx_mm":"",
                         "top1_value":"","top2_value":"","top1_top2_ratio":"","softmax_entropy_norm":"",
                         "qa_png":d.get("qa_png","")})
            continue
        dz = (float(pr[0]) - float(gt[0])) * iso_mm
        dy = (float(pr[1]) - float(gt[1])) * iso_mm
        dx = (float(pr[2]) - float(gt[2])) * iso_mm
        dist = math.sqrt(dz*dz+dy*dy+dx*dx)
        conf = d.get("confidence",{})
        rows.append({
            "patient_id":pid,"file":fp,"status":"ok",
            "dist_mm":dist,"dz_mm":dz,"dy_mm":dy,"dx_mm":dx,
            "top1_value":conf.get("top1_value",""),
            "top2_value":conf.get("top2_value",""),
            "top1_top2_ratio":conf.get("top1_top2_ratio",""),
            "softmax_entropy_norm":conf.get("softmax_entropy_norm",""),
            "qa_png":d.get("qa_png",""),
        })

    ensure_dir(os.path.dirname(os.path.abspath(out_csv)))
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k,"") for k in fieldnames})

    ok = [r for r in rows if r["status"]=="ok" and isinstance(r["dist_mm"],(int,float))]
    dists = sorted([r["dist_mm"] for r in ok])
    def pct(p):
        if not dists:
            return None
        idx = int(round((p/100.0)*(len(dists)-1)))
        idx = max(0, min(len(dists)-1, idx))
        return dists[idx]
    def count_over(th): return sum(1 for x in dists if x>th)
    print("Saved:", out_csv)
    print("Summary:",
          f"n={len(dists)} median={pct(50)} p90={pct(90)} p95={pct(95)} max={max(dists) if dists else None} over10={count_over(10.0)}")


def reqa_missing(args: argparse.Namespace) -> None:
    """
    Regenerate QA images ONLY (no training, no inference).
    Typical use:
      - after a CV run, some cases may have *_pred_center.json but missing QA png.
      - this scans fold output dirs and recreates QA for cases with dist >= threshold,
        optionally only when QA file is missing.

    It reloads the volume and GT/click from vw_point.json (same pipeline as run),
    and reads pred_center from *_pred_center.json.
    """
    root = os.path.abspath(args.root)
    run_dir = args.run_dir
    if not os.path.isabs(run_dir):
        run_dir = os.path.join(root, run_dir)
    if not os.path.isdir(run_dir):
        raise SystemExit(f"run_dir not found: {run_dir}")

    gt_csv = os.path.join(root, args.gt_csv)
    point_json_dir = os.path.join(root, args.point_json_dir)

    cases = load_cases_from_csv(gt_csv, point_json_dir, root=root, dicom_root=args.dicom_root)
    case_map = {c.pid: c for c in cases}

    # which folds to scan
    fold_dirs = []
    if args.folds:
        foldset = set([int(x) for x in re.split(r"[,\s]+", args.folds.strip()) if x.strip()])
        for f in sorted(foldset):
            fold_dirs.append((f, os.path.join(run_dir, f"fold_{f}")))
    else:
        for d in sorted(glob.glob(os.path.join(run_dir, "fold_*"))):
            bn = os.path.basename(d)
            try:
                f = int(bn.split("_")[1])
            except Exception:
                continue
            fold_dirs.append((f, d))

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "cpu" else args.device)
    print("Device:", device)
    click_keys = args.click_keys.split(",") if getattr(args,'click_keys',"") else None
    click_keys = args.click_keys.split(",") if getattr(args,'click_keys',"") else None
    gt_keys = args.gt_keys.split(",") if args.gt_keys else None

    n_total = 0
    n_done = 0
    n_skip = 0
    n_fail = 0

    for fold, fold_out in fold_dirs:
        if not os.path.isdir(fold_out):
            print(f"[reqa] fold dir missing: {fold_out}")
            continue

        pred_files = sorted(glob.glob(os.path.join(fold_out, "**", "*_pred_center.json"), recursive=True))
        if args.ids:
            idset = set([normalize_pid(x) for x in re.split(r"[,\s]+", args.ids.strip()) if x.strip()])
            pred_files = [p for p in pred_files if normalize_pid(os.path.basename(p).split("_")[0]) in idset]

        print("\n" + "="*80)
        print(f"[reqa] fold={fold} pred_files={len(pred_files)}")
        for pf in pred_files:
            n_total += 1
            try:
                res = load_json(pf)
                pid = normalize_pid(str(res.get("patient_id") or os.path.basename(pf).split("_")[0]))
                c = case_map.get(pid)
                if c is None:
                    raise RuntimeError(f"Case not found in gt_csv for pid={pid}")

                # Determine QA output path (keep consistent with run mode)
                qa_png = os.path.join(fold_out, f"{pid}_qa.png")
                if args.missing_only:
                    if os.path.exists(qa_png):
                        n_skip += 1
                        continue

                # distance threshold check
                dist = None
                if isinstance(res.get("eval"), dict) and ("dist_mm" in res["eval"]):
                    dist = float(res["eval"]["dist_mm"])
                if dist is None:
                    # compute from stored points if eval missing
                    gt = np.asarray(res.get("gt_center_zyx_canon_f", [np.nan]*3), dtype=np.float32)
                    pr = np.asarray(res.get("pred_center_zyx_canon_f", [np.nan]*3), dtype=np.float32)
                    dist = float(np.linalg.norm((pr-gt) * float(args.iso_mm)))

                if dist < float(args.dist_ge):
                    n_skip += 1
                    continue

                # Rebuild inputs (canonical full volume + canonical gt/click)
                inp = prepare_case_inputs(
                    case=c,
                    iso_mm=args.iso_mm,
                    crop_size_mm=tuple(args.crop_mm),
                    input_shape=tuple(args.input_shape),
                    click_keys=click_keys,
                    gt_keys=gt_keys,
                    vw_points_space=args.vw_points_space,
                    crop_y_front_mm=args.crop_y_front_mm,
                    crop_y_back_mm=args.crop_y_back_mm,
                    anchor_method=args.anchor_method,
                    anchor_x_frac=args.anchor_x_frac,
                    anchor_y_shift_mm=args.anchor_y_shift_mm,
                    anchor_z_shift_mm=args.anchor_z_shift_mm,
                    anterior_mask_y_mm=args.anterior_mask_y_mm,
                    anterior_mask_alpha=args.anterior_mask_alpha,
                    anterior_mask_ramp_mm=args.anterior_mask_ramp_mm,
                    anterior_mask_mode=args.anterior_mask_mode,
                )
                gt_canon = np.asarray(res.get("gt_center_zyx_canon_f", inp["gt_canon_zyx"].tolist()), dtype=np.float32)
                pr_canon = np.asarray(res.get("pred_center_zyx_canon_f", None), dtype=np.float32)
                if pr_canon is None or (np.any(~np.isfinite(pr_canon))):
                    raise RuntimeError("pred_center_zyx_canon_f missing or invalid in pred_center.json")

                if not HAS_MPL:
                    raise RuntimeError("matplotlib is not available; cannot generate QA images.")
                title = (f"{pid} dist={dist:.2f}mm fold={fold} side={c.side_rl} "
                         f"flipped={inp['flipped_lr']} gt_key={inp['point_keys'].get('gt_key_used','')}")
                save_qa_montage(inp["vol_full_canon_zyx"], gt_canon, pr_canon, None, qa_png, title=title)
                n_done += 1
            except Exception as e:
                n_fail += 1
                print(f"[reqa][NG] {os.path.basename(pf)}: {e}")

    print("\n[reqa] Done.")
    print(f"[reqa] total={n_total}  regenerated={n_done}  skipped={n_skip}  failed={n_fail}")


def run_cv(args: argparse.Namespace) -> None:
    root = os.path.abspath(args.root)
    gt_csv = os.path.join(root, args.gt_csv)
    point_json_dir = os.path.join(root, args.point_json_dir)

    cases = load_cases_from_csv(gt_csv, point_json_dir, root=root, dicom_root=args.dicom_root)
    if args.ids:
        idset = set([normalize_pid(x) for x in re.split(r"[,\s]+", args.ids.strip()) if x.strip()])
        cases = [c for c in cases if c.pid in idset]
    if not cases:
        raise SystemExit("No cases found after filtering.")

    folds = sorted(set([c.fold for c in cases]))
    if args.folds:
        foldset = set([int(x) for x in re.split(r"[,\s]+", args.folds.strip()) if x.strip()])
        folds = [f for f in folds if f in foldset]

    out_dir = os.path.join(root, args.out_dir if args.out_dir else f"AutoROI_ClickCenter_CANON_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    ensure_dir(out_dir)
    print("Output:", out_dir)

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "cpu" else args.device)
    print("Device:", device)
    click_keys = args.click_keys.split(",") if getattr(args,'click_keys',"") else None
    gt_keys = args.gt_keys.split(",") if args.gt_keys else None

    for fold in folds:
        fold_out = os.path.join(out_dir, f"fold_{fold}")
        ensure_dir(fold_out)

        train_cases = [c for c in cases if c.fold != fold]
        val_cases   = [c for c in cases if c.fold == fold]

        print("\n" + "="*80)
        print(f"[Fold {fold}] train={len(train_cases)}  val={len(val_cases)}")

        # -------------------------
        # NO-LEAK guarantee:
        #  - val_cases (this fold) must never be used for training nor for early-stopping/validation.
        #  - we create dev_cases ONLY from train_cases (other folds).
        # -------------------------
        val_pids = set([c.pid for c in val_cases])
        train_pids = set([c.pid for c in train_cases])
        inter = sorted(list(val_pids.intersection(train_pids)))
        if inter:
            raise RuntimeError(f"[Fold {fold}] DATA LEAKAGE: train/val overlap: {inter[:10]} (and more)" if len(inter)>10 else f"[Fold {fold}] DATA LEAKAGE: train/val overlap: {inter}")

        # Dev split from training folds (for monitoring only). This keeps fold-{fold} completely untouched.
        rng = np.random.RandomState(int(args.seed) + int(fold) * 10007)
        train_cases_shuf = list(train_cases)
        rng.shuffle(train_cases_shuf)
        dev_cases = []
        train_cases_used = train_cases_shuf
        if args.dev_fraction > 0 and len(train_cases_shuf) >= 3:
            n_dev = int(round(len(train_cases_shuf) * float(args.dev_fraction)))
            n_dev = max(1, min(n_dev, len(train_cases_shuf)-2))  # keep >=2 for training
            dev_cases = train_cases_shuf[:n_dev]
            train_cases_used = train_cases_shuf[n_dev:]
        print(f"[Fold {fold}] dev(from-train)={len(dev_cases)}  train_used={len(train_cases_used)}")

        # If train set is empty (e.g., running with --ids only 1 case), skip training.
        # Still allow QA generation to verify GT/click alignment.
        if len(train_cases) == 0:
            print(f"[Fold {fold}] WARNING: train set is empty. Skipping training and running QA-only for val cases.")
            for c in val_cases:
                try:
                    inp = prepare_case_inputs(
                    case=c,
                    iso_mm=args.iso_mm,
                    crop_size_mm=tuple(args.crop_mm),
                    input_shape=tuple(args.input_shape),
                    click_keys=click_keys,
                    gt_keys=gt_keys,
                    vw_points_space=args.vw_points_space,
                    crop_y_front_mm=args.crop_y_front_mm,
                    crop_y_back_mm=args.crop_y_back_mm,
                    anchor_method=args.anchor_method,
                    anchor_x_frac=args.anchor_x_frac,
                    anchor_y_shift_mm=args.anchor_y_shift_mm,
                    anchor_z_shift_mm=args.anchor_z_shift_mm,
                    anterior_mask_y_mm=args.anterior_mask_y_mm,
                    anterior_mask_alpha=args.anterior_mask_alpha,
                    anterior_mask_ramp_mm=args.anterior_mask_ramp_mm,
                    anterior_mask_mode=args.anterior_mask_mode,
                )
                    # Always save a small json for debugging even if matplotlib is unavailable
                    qa_dir = os.path.join(fold_out, "qa")
                    ensure_dir(qa_dir)
                    qa_json = os.path.join(qa_dir, f"{c.pid}_qa_only.json")
                    save_json({
                        "patient_id": c.pid,
                        "fold": int(fold),
                        "side_rl": c.side_rl,
                        "dicom_dir": c.dicom_dir,
                        "point_json": c.point_json,
                        "vw_points_space": args.vw_points_space,
                        "iso_mm": float(args.iso_mm),
                        "flipped_lr": bool(inp.get("flipped_lr", False)),
                        "anchor_canon_zyx_f": [float(v) for v in inp["anchor_canon_zyx"].tolist()],
                        "gt_canon_zyx_f": [float(v) for v in inp["gt_canon_zyx"].tolist()],
                        "point_keys": inp.get("point_keys", {}),
                    }, qa_json)

                    if args.save_qa and HAS_MPL:
                        qa_png = os.path.join(qa_dir, f"{c.pid}_gt_click_qa.png")
                        save_qa_montage(
                            inp["vol_full_canon_zyx"],
                            inp["gt_canon_zyx"],
                            inp["gt_canon_zyx"],  # pred same as gt for QA-only
                            inp["anchor_canon_zyx"],
                            qa_png,
                            title=f"{c.pid} fold{fold} QA-only (train=0) flipped={inp.get('flipped_lr', False)}",
                        )
                        print(f"[Fold {fold}] QA-only OK: {c.pid} -> {os.path.basename(qa_png)}")
                    else:
                        if args.save_qa and (not HAS_MPL):
                            print(f"[Fold {fold}] QA-only note: matplotlib not available; wrote {os.path.basename(qa_json)}")
                        else:
                            print(f"[Fold {fold}] QA-only OK: {c.pid} -> {os.path.basename(qa_json)}")
                except Exception as e:
                    print(f"[Fold {fold}] QA-only failed for {c.pid}: {e}")
            continue


        model, best_epoch = train_one_fold(
            train_cases=train_cases_used, dev_cases=dev_cases,
            iso_mm=args.iso_mm, crop_size_mm=tuple(args.crop_mm),
            input_shape=tuple(args.input_shape), click_keys=click_keys, sigma_vox=args.sigma_vox,
            clip_high=float(args.clip_high),
            crop_y_front_mm=args.crop_y_front_mm, crop_y_back_mm=args.crop_y_back_mm,
            gt_keys=gt_keys,
            device=device, vw_points_space=args.vw_points_space, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed + fold,
            anchor_method=args.anchor_method, anchor_x_frac=args.anchor_x_frac, anchor_y_shift_mm=args.anchor_y_shift_mm, anchor_z_shift_mm=args.anchor_z_shift_mm,
            anterior_mask_y_mm=args.anterior_mask_y_mm, anterior_mask_alpha=args.anterior_mask_alpha, anterior_mask_ramp_mm=args.anterior_mask_ramp_mm, anterior_mask_mode=args.anterior_mask_mode,
            best_epoch_metric=args.best_epoch_metric, best_epoch_patience=args.best_epoch_patience, best_epoch_min_delta=getattr(args,'best_epoch_min_delta',0.0), best_epoch_ema=getattr(args,'best_epoch_ema',0.0)
        )

        ckpt_path = os.path.join(fold_out, "model.pt")
        torch.save(model.state_dict(), ckpt_path)
        with open(os.path.join(fold_out, "best_epoch.txt"), "w", encoding="utf-8") as f:
            f.write(str(best_epoch) + "\n")


        for c in val_cases:
            try:
                res = infer_center_for_case(
                    model=model, case=c, iso_mm=args.iso_mm, crop_size_mm=tuple(args.crop_mm),
                    input_shape=tuple(args.input_shape),
                    click_keys=click_keys,
                    gt_keys=gt_keys, device=device, vw_points_space=args.vw_points_space,
                    clip_high=float(args.clip_high),
                    crop_y_front_mm=args.crop_y_front_mm, crop_y_back_mm=args.crop_y_back_mm,
                    anchor_method=args.anchor_method, anchor_x_frac=args.anchor_x_frac, anchor_y_shift_mm=args.anchor_y_shift_mm, anchor_z_shift_mm=args.anchor_z_shift_mm,
                    anterior_mask_y_mm=args.anterior_mask_y_mm, anterior_mask_alpha=args.anterior_mask_alpha, anterior_mask_ramp_mm=args.anterior_mask_ramp_mm, anterior_mask_mode=args.anterior_mask_mode,
                    metal_thr=float(args.metal_thr), metal_alpha=float(args.metal_alpha),
                    metal_attenuate_on_guard=bool(args.metal_attenuate_on_guard),
                    metal_attenuate_always=bool(args.metal_attenuate_always),
                    guard_top1_top2_lt=float(args.guard_top1_top2_lt),
                    guard_entropy_gt=float(args.guard_entropy_gt),
                )
                inp = res["inp"]
                pred_canon = res["pred_center_full_canon_f"]
                gt_canon = res["gt_center_full_canon_f"]

                qa_png = ""
                if args.save_qa and HAS_MPL and (res["eval"]["dist_mm"] >= args.qa_only_if_dist_ge):
                    qa_png = os.path.join(fold_out, f"{c.pid}_qa.png")
                    title = (f"{c.pid} dist={res['eval']['dist_mm']:.2f}mm fold={fold} "
                             f"side={c.side_rl} flipped={inp['flipped_lr']} "
                             f"ctonly={inp['point_keys']['gt_key_used']} gt_key={inp['point_keys']['gt_key_used']}")
                    save_qa_montage(inp["vol_full_canon_zyx"], gt_canon, pred_canon, inp.get("anchor_canon_zyx", None), qa_png, title=title)

                roi_files = {}
                if args.save_rois:
                    roi_dir = os.path.join(fold_out, "rois")
                    ensure_dir(roi_dir)
                    for rmm in args.roi_sizes_mm:
                        roi = extract_spherical_roi(inp["vol_full_canon_zyx"], pred_canon, float(rmm), args.iso_mm, pad_hu=DEFAULT_HU_PAD)
                        npy_path = os.path.join(roi_dir, f"{c.pid}_pred_{int(rmm)}mm.npy")
                        np.save(npy_path, roi)
                        roi_files[f"pred_{int(rmm)}mm_npy"] = npy_path

                pred_meta = {
                    "patient_id": c.pid,
                    "fold": int(fold),
                    "side_rl": c.side_rl,
                    "flipped_lr": bool(inp["flipped_lr"]),
                    "dicom_dir": c.dicom_dir,
                    "point_json": c.point_json,
                    "point_keys": inp["point_keys"],
                    "qa_png": qa_png,
                    "confidence": res["confidence"],
                    "eval": {
                        "iso_spacing_mm": float(args.iso_mm),
                        "dist_mm": float(res["eval"]["dist_mm"]),
                        "dz_mm": float(res["eval"]["dz_mm"]),
                        "dy_mm": float(res["eval"]["dy_mm"]),
                        "dx_mm": float(res["eval"]["dx_mm"]),
                    },
                    "gt_center_zyx_canon_f": [float(v) for v in gt_canon.tolist()],
                    "pred_center_zyx_canon_f": [float(v) for v in pred_canon.tolist()],
                    "gt_center_zyx": [int(round(v)) for v in gt_canon.tolist()],
                    "pred_center_zyx": [int(round(v)) for v in pred_canon.tolist()],
                    "roi_files": roi_files,
                }

                out_json = os.path.join(fold_out, f"{c.pid}_pred_center.json")
                save_json(pred_meta, out_json)
                print(f"[OK] {c.pid} dist={res['eval']['dist_mm']:.2f}mm")

            except Exception as e:
                print(f"[NG] {c.pid}: {e}")

    print("\nAll Done. Output dir:", out_dir)
    if args.auto_eval:
        print("\nRunning internal eval...")
        eval_root(out_dir, iso_mm=args.iso_mm, out_csv=os.path.join(out_dir, "pred_center_errors.csv"))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_run = sub.add_parser("run")
    ap_run.add_argument("--root", required=True)
    ap_run.add_argument("--gt_csv", default="df_final_fixed.csv")
    ap_run.add_argument("--point_json_dir", default="vw_roi")
    ap_run.add_argument("--vw_points_space", default="auto", choices=["auto","raw","iso","iso_canon"],
                    help="Interpret vw_point.json coords: raw=DICOM grid, iso=after iso-resample, iso_canon=after iso-resample + LR-canonical. auto=heuristic")

    ap_run.add_argument("--dicom_root", default="DICOM")

    ap_run.add_argument("--folds", default="", help="comma/space separated folds. empty=all")
    ap_run.add_argument("--ids", default="", help="optional: ids to run")
    ap_run.add_argument("--out_dir", default="", help="output dir name under root")
    ap_run.add_argument("--run_dir", default="", dest="out_dir", help="alias of --out_dir")

    ap_run.add_argument("--iso_mm", type=float, default=DEFAULT_ISO_MM)
    ap_run.add_argument("--crop_mm", type=float, nargs=3, default=list(DEFAULT_CROP_MM))
    ap_run.add_argument("--crop_y_front_mm", type=float, default=None,
                    help="Optional asymmetric crop on Y: mm to keep to +Y (front/anterior) from anchor center. If set, reduces anterior FOV.")
    ap_run.add_argument("--crop_y_back_mm", type=float, default=None,
                    help="Optional asymmetric crop on Y: mm to keep to -Y (back/posterior) from anchor center. If set with crop_y_front_mm, enables asymmetric crop.")
    
    ap_run.add_argument("--clip_high", type=float, default=3071.0,
                    help="Upper HU clip value before normalization (e.g., 2000/2500/3000). Lower may suppress metal/teeth artifacts.")

    # CT-only crop anchor (deterministic; no click/GT)
    ap_run.add_argument("--anchor_method", type=str, default="bbox_frac", choices=["bbox_frac","volume_center"],
                    help="How to estimate crop center from CT only (after LR canonicalization).")
    ap_run.add_argument("--anchor_x_frac", type=float, default=0.70,
                    help="In bbox_frac anchor: x = xmin + frac*(xmax-xmin). >0.5 biases to RIGHT (target ear).")
    ap_run.add_argument("--anchor_y_shift_mm", type=float, default=0.0, help="Additional +Y shift (mm) applied to anchor.")
    ap_run.add_argument("--anchor_z_shift_mm", type=float, default=0.0, help="Additional +Z shift (mm) applied to anchor.")

    # Conditional anterior mask in canonical crop cube (anterior assumed -Y)
    ap_run.add_argument("--anterior_mask_y_mm", type=float, default=0.0,
                    help="Start suppressing anterior region at this distance (mm) from crop center toward -Y. 0 disables.")
    ap_run.add_argument("--anterior_mask_alpha", type=float, default=0.2,
                    help="Suppression strength: smaller=stronger. Used in attenuate mode.")
    ap_run.add_argument("--anterior_mask_ramp_mm", type=float, default=10.0,
                    help="Ramp width (mm) for smooth transition of anterior suppression.")
    ap_run.add_argument("--anterior_mask_mode", type=str, default="attenuate", choices=["attenuate","zero"],
                    help="attenuate: blend HU toward pad; zero: set to pad HU.")
    ap_run.add_argument("--metal_thr", type=float, default=3000.0,
                    help="HU threshold to define metal mask on the *input grid* (after crop+resize). Used for heatmap attenuation.")
    ap_run.add_argument("--metal_alpha", type=float, default=0.90,
                    help="Attenuation strength in metal regions: p *= (1 - metal_alpha*metal_mask).")
    ap_run.add_argument("--metal_attenuate_on_guard", action="store_true",
                    help="When guard triggers, attenuate probability inside metal mask and recompute center (no second forward pass).")
    ap_run.add_argument("--metal_attenuate_always", action="store_true",
                    help="Always attenuate inside metal mask (may hurt normal cases). Prefer --metal_attenuate_on_guard.")
    
    ap_run.add_argument("--guard_top1_top2_lt", type=float, default=1.05,
                    help="Low-confidence guard: triggers if top1/top2 ratio is below this.")
    ap_run.add_argument("--guard_entropy_gt", type=float, default=0.62,
                    help="Low-confidence guard: triggers if softmax entropy (normalized) is above this.")
    ap_run.add_argument("--input_shape", type=int, nargs=3, default=list(DEFAULT_INPUT_SHAPE))
    ap_run.add_argument("--sigma_vox", type=float, default=DEFAULT_SIGMA_VOX)

    ap_run.add_argument("--epochs", type=int, default=10)
    ap_run.add_argument("--best_epoch_metric", type=str, default="val_kl", choices=["val_kl","dev_dist","dev_p90","dev_max","dev_count_gt15"],
                        help="Select best epoch by this metric: val_kl (default) or dev_dist (mean distance in mm on dev set).")
    ap_run.add_argument("--best_epoch_patience", type=int, default=0,
                        help="Early stopping patience based on best_epoch_metric. 0 disables early stopping.")
    ap_run.add_argument("--best_epoch_min_delta", type=float, default=0.0,
                        help="Minimum improvement required to update best epoch (applied to selected metric).")
    ap_run.add_argument("--best_epoch_ema", type=float, default=0.0,
                        help="EMA smoothing factor for best-epoch metric in [0,1). 0 disables smoothing. Example: 0.9.")
    ap_run.add_argument("--batch_size", type=int, default=2)
    ap_run.add_argument("--lr", type=float, default=1e-3)
    ap_run.add_argument("--seed", type=int, default=1337)
    ap_run.add_argument("--deterministic", action="store_true",
                        help="Enable deterministic/reproducible mode (cudnn/cublas/tf32/seed/series sort).")
    ap_run.add_argument("--deterministic_strict", action="store_true", help="Deterministic mode but fail on non-deterministic ops.")

    ap_run.add_argument("--dev_fraction", type=float, default=0.1,
                        help="Fraction of training-fold cases used as dev set for monitoring. (NO-LEAK: dev is drawn only from non-val folds). Set 0 to disable.")

    ap_run.add_argument("--device", default="auto", choices=["auto","cpu","cuda"])
    ap_run.add_argument("--save_qa", action="store_true")
    ap_run.add_argument("--qa_only_if_dist_ge", type=float, default=0.0)
    ap_run.add_argument("--save_rois", action="store_true")
    ap_run.add_argument("--roi_sizes_mm", type=int, nargs="+", default=DEFAULT_ROI_SIZES_MM)
    ap_run.add_argument("--auto_eval", action="store_true", default=True,
                     help="Run evaluation after prediction (default: on).")
    ap_run.add_argument("--no_auto_eval", action="store_false", dest="auto_eval",
                     help="Disable automatic evaluation step.")

    ap_run.add_argument("--click_keys", default="", help="override click keys (comma-separated)")
    ap_run.add_argument("--gt_keys", default="", help="override gt keys (comma-separated)")

    ap_eval = sub.add_parser("eval")

    ap_eval.add_argument("--root", required=True)
    ap_eval.add_argument("--run_dir", required=True, help="Existing CV output dir (absolute or under --root).")
    ap_eval.add_argument("--iso_mm", type=float, default=DEFAULT_ISO_MM)
    ap_eval.add_argument("--seed", type=int, default=1337)
    ap_eval.add_argument("--deterministic", action="store_true",
                        help="Enable deterministic/reproducible mode (cudnn/cublas/tf32/seed/series sort).")
    ap_eval.add_argument("--deterministic_strict", action="store_true", help="Deterministic mode but fail on non-deterministic ops.")
    ap_eval.add_argument("--out_csv", default="pred_center_errors_eval.csv", help="Output CSV filename (under run_dir).")


    ap_reqa = sub.add_parser("reqa", aliases=["repa"])
    ap_reqa.add_argument("--root", required=True)
    ap_reqa.add_argument("--run_dir", required=True, help="Existing CV output dir (absolute or under --root).")
    ap_reqa.add_argument("--gt_csv", default="df_final_fixed.csv")
    ap_reqa.add_argument("--point_json_dir", default="vw_roi")
    ap_reqa.add_argument("--dicom_root", default="DICOM")
    ap_reqa.add_argument("--vw_points_space", default="auto", choices=["auto","raw","iso","iso_canon"])
    ap_reqa.add_argument("--iso_mm", type=float, default=DEFAULT_ISO_MM)
    ap_reqa.add_argument("--seed", type=int, default=1337)
    ap_reqa.add_argument("--deterministic", action="store_true",
                        help="Enable deterministic/reproducible mode (cudnn/cublas/tf32/seed/series sort).")
    ap_reqa.add_argument("--deterministic_strict", action="store_true", help="Deterministic mode but fail on non-deterministic ops.")
    ap_reqa.add_argument("--crop_mm", type=float, nargs=3, default=list(DEFAULT_CROP_MM))
    ap_reqa.add_argument("--input_shape", type=int, nargs=3, default=list(DEFAULT_INPUT_SHAPE))

    # (optional) match CT-only preprocessing used during run
    ap_reqa.add_argument("--anchor_method", type=str, default="bbox_frac", choices=["bbox_frac","volume_center"])
    ap_reqa.add_argument("--anchor_x_frac", type=float, default=0.70)
    ap_reqa.add_argument("--anchor_y_shift_mm", type=float, default=0.0)
    ap_reqa.add_argument("--anchor_z_shift_mm", type=float, default=0.0)
    ap_reqa.add_argument("--anterior_mask_y_mm", type=float, default=0.0)
    ap_reqa.add_argument("--anterior_mask_alpha", type=float, default=0.2)
    ap_reqa.add_argument("--anterior_mask_ramp_mm", type=float, default=10.0)
    ap_reqa.add_argument("--anterior_mask_mode", type=str, default="attenuate", choices=["attenuate","zero"])
    ap_reqa.add_argument("--folds", default="", help="comma/space separated folds to scan. empty=all")
    ap_reqa.add_argument("--ids", default="", help="optional: ids to scan")
    ap_reqa.add_argument("--device", default="auto", choices=["auto","cpu","cuda"])
    ap_reqa.add_argument("--click_keys", default="", help="override click keys (comma-separated)")
    ap_reqa.add_argument("--gt_keys", default="", help="override gt keys (comma-separated)")
    ap_reqa.add_argument("--dist_ge", type=float, default=5.0, help="Regenerate QA only when dist_mm >= this threshold.")
    ap_reqa.add_argument("--missing_only", action="store_true", help="Only regenerate if QA png does not exist.")


    args = ap.parse_args()
    if getattr(args, "deterministic", False):
        enable_determinism(getattr(args, "seed", 1337), verbose=True)
    if args.cmd == "run":
        run_cv(args)
    elif args.cmd == "eval":
        run_dir = args.run_dir
        if not os.path.isabs(run_dir):
            run_dir = os.path.join(args.root, run_dir)
        out_csv_path = os.path.join(run_dir, args.out_csv)
        eval_root(run_dir, iso_mm=args.iso_mm, out_csv=out_csv_path)
    elif args.cmd in ("reqa","repa"):
        reqa_missing(args)
    else:
        raise SystemExit("Unknown cmd")

if __name__ == "__main__":
    main()