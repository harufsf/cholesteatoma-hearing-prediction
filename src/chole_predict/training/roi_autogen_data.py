from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from chole_predict.data.case_schema import CaseInfo
from chole_predict.io.dicom_io import load_dicom_series_to_hu_zyx, resample_to_iso
from chole_predict.roi.geometry import scale_point_to_iso, map_point_crop_to_input
from chole_predict.roi.canonicalize import canonicalize_lr
from chole_predict.roi.vw_json import load_points_from_vw_json
from chole_predict.roi.anchor import estimate_anchor_center_ctonly
from chole_predict.roi.crop import crop_around_center, resize_crop_to_input, apply_conditional_anterior_mask

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
