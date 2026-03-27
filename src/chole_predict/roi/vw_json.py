from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from chole_predict.io.json_io import load_json

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
