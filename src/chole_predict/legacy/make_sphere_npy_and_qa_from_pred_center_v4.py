#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""make_sphere_npy_and_qa_from_pred_center.py

Create sphere ROI .npy files from *_pred_center.json and optionally export QA PNGs.
(Windows paths omitted in docstring to avoid escape issues.)
"""



import argparse
import glob
import json
import os
from typing import Tuple, List

import numpy as np

def _try_import_project_normalizer():
    try:
        from dicom_normalize import read_and_normalize_ct  # type: ignore
        return read_and_normalize_ct
    except Exception:
        return None

_READ_AND_NORMALIZE_CT = _try_import_project_normalizer()

def load_dicom_volume_normalized(dicom_dir: str, iso_spacing_mm: float = 0.5, target_orient: str = "LPS"):
    """
    Load and normalize a DICOM CT series.

    Preference:
      1) Use project implementation: dicom_normalize.read_and_normalize_ct (if available)
      2) Fallback: SimpleITK-based reader + reorient + resample (internal)

    Returns:
      vol_zyx: np.ndarray (float32) in HU (or close) after resampling
      spacing_zyx_mm: (sz, sy, sx)
    """
    if _READ_AND_NORMALIZE_CT is not None:
        norm = _READ_AND_NORMALIZE_CT(dicom_dir, target_orient=target_orient, iso_spacing_mm=iso_spacing_mm)
        return norm.vol_zyx, norm.spacing_zyx_mm

    # ---- Fallback path (no dicom_normalize.py available) ----
    try:
        import SimpleITK as sitk
    except Exception as e:
        raise RuntimeError(
            "dicom_normalize.read_and_normalize_ct is not importable, and SimpleITK is also unavailable. "
            "Install SimpleITK or make dicom_normalize.py importable."
        ) from e

    # Read series
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dicom_dir)
    if not series_ids:
        raise RuntimeError(f"No DICOM series found in: {dicom_dir}")
    # pick first series
    file_names = reader.GetGDCMSeriesFileNames(dicom_dir, series_ids[0])
    reader.SetFileNames(file_names)
    img = reader.Execute()

    # Reorient (best-effort). If it fails, keep as-is.
    try:
        img = sitk.DICOMOrient(img, target_orient)
    except Exception:
        pass

    # Resample to isotropic spacing
    orig_spacing = np.array(list(img.GetSpacing()), dtype=float)  # (sx, sy, sz) in SimpleITK (x,y,z)
    orig_size = np.array(list(img.GetSize()), dtype=int)          # (nx, ny, nz)
    new_spacing = np.array([iso_spacing_mm, iso_spacing_mm, iso_spacing_mm], dtype=float)

    new_size = np.round(orig_size * (orig_spacing / new_spacing)).astype(int)
    new_size = np.maximum(new_size, 1).tolist()

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(tuple(new_spacing.tolist()))
    resampler.SetSize([int(x) for x in new_size])
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(-1024.0)

    img_rs = resampler.Execute(img)

    arr = sitk.GetArrayFromImage(img_rs).astype(np.float32)  # Z,Y,X
    # spacing_zyx
    spacing_zyx = (float(new_spacing[2]), float(new_spacing[1]), float(new_spacing[0]))
    return arr, spacing_zyx


def extract_spherical_roi(
    vol_zyx: np.ndarray,
    center_zyx: Tuple[int, int, int],
    spacing_zyx_mm: Tuple[float, float, float],
    diameter_mm: float,
    out_shape: Tuple[int, int, int] = (64, 64, 64),
    fill_hu: float = -1000.0,
) -> np.ndarray:
    radius_mm = float(diameter_mm) / 2.0
    crop_radius_vox = [radius_mm / float(s) for s in spacing_zyx_mm]

    cz, cy, cx = center_zyx
    z_min = int(cz - crop_radius_vox[0]); z_max = int(cz + crop_radius_vox[0])
    y_min = int(cy - crop_radius_vox[1]); y_max = int(cy + crop_radius_vox[1])
    x_min = int(cx - crop_radius_vox[2]); x_max = int(cx + crop_radius_vox[2])

    pad_z = (max(0, -z_min), max(0, z_max - vol_zyx.shape[0]))
    pad_y = (max(0, -y_min), max(0, y_max - vol_zyx.shape[1]))
    pad_x = (max(0, -x_min), max(0, x_max - vol_zyx.shape[2]))

    vol_padded = np.pad(vol_zyx, (pad_z, pad_y, pad_x), mode="constant", constant_values=fill_hu)
    crop = vol_padded[
        z_min + pad_z[0] : z_max + pad_z[0],
        y_min + pad_y[0] : y_max + pad_y[0],
        x_min + pad_x[0] : x_max + pad_x[0],
    ]

    try:
        from scipy.ndimage import zoom
    except Exception as e:
        raise RuntimeError("scipy is required for scipy.ndimage.zoom(). Please install scipy.") from e

    scale = np.array(out_shape, dtype=np.float32) / np.array(crop.shape, dtype=np.float32)
    roi_resized = zoom(crop, scale, order=1)

    # spherical mask (inscribed sphere)
    cz2, cy2, cx2 = (np.array(out_shape, dtype=np.float32) / 2.0)
    zz, yy, xx = np.ogrid[:out_shape[0], :out_shape[1], :out_shape[2]]
    dist_sq = (zz - cz2) ** 2 + (yy - cy2) ** 2 + (xx - cx2) ** 2
    mask_radius_sq = (min(out_shape) / 2.0) ** 2

    roi = roi_resized.astype(np.float32, copy=True)
    roi[dist_sq > mask_radius_sq] = float(fill_hu)
    return roi

def save_center_qa_png(
    vol_zyx: np.ndarray,
    center_zyx: Tuple[int, int, int],
    out_png: str,
    wl: float = 400.0,
    ww: float = 4000.0,
    title: str | None = None,
):
    import matplotlib.pyplot as plt

    cz, cy, cx = center_zyx
    cz = int(np.clip(cz, 0, vol_zyx.shape[0]-1))
    cy = int(np.clip(cy, 0, vol_zyx.shape[1]-1))
    cx = int(np.clip(cx, 0, vol_zyx.shape[2]-1))

    # slices
    axial = vol_zyx[cz, :, :]          # YX
    coronal = vol_zyx[:, cy, :]        # ZX
    sagittal = vol_zyx[:, :, cx]       # ZY

    vmin = wl - ww/2.0
    vmax = wl + ww/2.0

    fig = plt.figure(figsize=(12, 4))
    if title:
        fig.suptitle(title)

    # Axial
    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(axial, cmap="gray", vmin=vmin, vmax=vmax)
    ax1.axhline(cy, color="lime", linewidth=0.8)
    ax1.axvline(cx, color="lime", linewidth=0.8)
    ax1.set_title(f"Axial (z={cz})")
    ax1.axis("off")

    # Coronal
    ax2 = plt.subplot(1, 3, 2)
    ax2.imshow(coronal, cmap="gray", vmin=vmin, vmax=vmax)
    ax2.axhline(cz, color="lime", linewidth=0.8)
    ax2.axvline(cx, color="lime", linewidth=0.8)
    ax2.set_title(f"Coronal (y={cy})")
    ax2.axis("off")

    # Sagittal
    ax3 = plt.subplot(1, 3, 3)
    ax3.imshow(sagittal, cmap="gray", vmin=vmin, vmax=vmax)
    ax3.axhline(cz, color="lime", linewidth=0.8)
    ax3.axvline(cy, color="lime", linewidth=0.8)
    ax3.set_title(f"Sagittal (x={cx})")
    ax3.axis("off")

    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

def _parse_sizes(s: str) -> List[int]:
    out = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(int(float(x)))
    return out

def _infer_id_from_json_path(json_path: str) -> str:
    base = os.path.basename(json_path)
    if base.endswith("_pred_center.json"):
        return base[: -len("_pred_center.json")]
    return os.path.splitext(base)[0]

def _resolve_dicom_dir(meta: dict, pid: str, dicom_root: str | None) -> str:
    if meta.get("dicom_dir"):
        return meta["dicom_dir"]
    for k in ["dicom_path", "dicom", "dicomFolder"]:
        if meta.get(k):
            return meta[k]
    if not dicom_root:
        raise ValueError("JSON missing dicom_dir and --dicom_root not provided.")
    cand1 = os.path.join(dicom_root, pid, "DICOM")
    cand2 = os.path.join(dicom_root, pid)
    if os.path.isdir(cand1):
        return cand1
    if os.path.isdir(cand2):
        return cand2
    raise ValueError(f"Cannot resolve DICOM dir for id={pid} under dicom_root={dicom_root}")

def _resolve_center_zyx(meta: dict) -> Tuple[int, int, int]:
    if meta.get("pred_center_zyx") is not None:
        c = meta["pred_center_zyx"]
        return (int(c[0]), int(c[1]), int(c[2]))
    if meta.get("pred_center_zyx_canon_f") is not None:
        c = meta["pred_center_zyx_canon_f"]
        return (int(round(c[0])), int(round(c[1])), int(round(c[2])))
    raise ValueError("Missing pred_center_zyx / pred_center_zyx_canon_f in JSON.")

def _resolve_iso_spacing(meta: dict, default_iso: float) -> float:
    ev = meta.get("eval", {})
    if isinstance(ev, dict) and ev.get("iso_spacing_mm") is not None:
        return float(ev["iso_spacing_mm"])
    if meta.get("iso_spacing_mm") is not None:
        return float(meta["iso_spacing_mm"])
    return float(default_iso)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roi_dir", required=True, help="Folder containing *_pred_center.json (and where outputs go).")
    ap.add_argument("--recursive", action="store_true", help="Search *_pred_center.json recursively under roi_dir (e.g., fold subfolders).")
    ap.add_argument("--dicom_root", default=None)
    ap.add_argument("--ids", default=None, help="Optional comma-separated IDs to process (default: all).")
    ap.add_argument("--sizes", default="25,40,60")
    ap.add_argument("--out_shape", default="64,64,64")
    ap.add_argument("--target_orient", default="LPS")
    ap.add_argument("--default_iso_spacing", type=float, default=0.5)
    ap.add_argument("--fill_hu", type=float, default=-1000.0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--qa", action="store_true", help="Also output QA PNG.")
    ap.add_argument("--qa_dir", default=None, help="Optional. If not set, QA is saved in roi_dir.")
    ap.add_argument("--wl", type=float, default=400.0)
    ap.add_argument("--ww", type=float, default=4000.0)
    args = ap.parse_args()

    roi_dir = args.roi_dir
    sizes = _parse_sizes(args.sizes)
    out_shape = tuple(int(x) for x in args.out_shape.split(","))

    want_ids = None
    if args.ids:
        want_ids = set([x.strip() for x in args.ids.split(",") if x.strip()])

    json_paths = sorted(glob.glob(os.path.join(roi_dir, "**", "*_pred_center.json"), recursive=True))
    if want_ids is not None:
        json_paths = [p for p in json_paths if _infer_id_from_json_path(p) in want_ids]
    if not json_paths:
        raise SystemExit(f"No *_pred_center.json found under: {roi_dir}")

    qa_dir = args.qa_dir  # if None, save next to each json
    if qa_dir is not None:
        os.makedirs(qa_dir, exist_ok=True)

    print(f"[INFO] Found {len(json_paths)} pred_center JSON files in {roi_dir}")
    ok = 0
    failed = 0

    for jp in json_paths:
        pid = _infer_id_from_json_path(jp)
        try:
            with open(jp, "r", encoding="utf-8") as f:
                meta = json.load(f)

            dicom_dir = _resolve_dicom_dir(meta, pid, args.dicom_root)
            center_zyx = _resolve_center_zyx(meta)
            iso_spacing = _resolve_iso_spacing(meta, args.default_iso_spacing)
            flipped_lr = bool(meta.get("flipped_lr", False))

            vol, spacing = load_dicom_volume_normalized(
                dicom_dir,
                iso_spacing_mm=iso_spacing,
                target_orient=args.target_orient
            )

            # IMPORTANT: match "iso_canon" space used by your autogen scripts
            if flipped_lr:
                vol = vol[:, :, ::-1].copy()

            # Save QA (center consistency) if requested
            if args.qa:
                qa_base = qa_dir if qa_dir is not None else os.path.dirname(jp)
                qa_path = os.path.join(qa_base, f"QA_{pid}_center_check.png")
                title = f"{pid}  center_zyx={center_zyx}  flipped_lr={flipped_lr}  iso={iso_spacing}"
                save_center_qa_png(vol, center_zyx, qa_path, wl=args.wl, ww=args.ww, title=title)

            # Save sphere ROIs
            for mm in sizes:
                out_name = f"{pid}_{mm}x{mm}x{mm}mm_sphere.npy"
                out_path = os.path.join(os.path.dirname(jp), out_name)
                if (not args.overwrite) and os.path.exists(out_path):
                    continue
                roi = extract_spherical_roi(vol, center_zyx, spacing, mm, out_shape=out_shape, fill_hu=args.fill_hu)
                np.save(out_path, roi)

            ok += 1
            if ok % 20 == 0:
                print(f"[INFO] processed {ok}/{len(json_paths)} ...")
        except Exception as e:
            failed += 1
            print(f"[FAIL] {pid}: {e}")

    print(f"[DONE] ok={ok}, failed={failed}, roi_dir={roi_dir}, qa_dir={qa_dir}")

if __name__ == "__main__":
    main()
