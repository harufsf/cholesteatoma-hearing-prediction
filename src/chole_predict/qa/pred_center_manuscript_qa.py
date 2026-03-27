#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""make_pred_center_qa_final_smallmarker.py

Manuscript-ready QA figure generator.
- Smaller markers than the previous version
- Keeps scale bar
- Outputs:
    CLICK_{id}_click_only.png
    BOTH_{id}_click_and_pred.png

Default roi_dir is set for the current project.
"""

import argparse
import glob
import json
import os
from typing import Tuple, Optional

import numpy as np


DEFAULT_ROI_DIR = r"path/to/roi_dir"


def _try_import_project_normalizer():
    try:
        from dicom_normalize import read_and_normalize_ct  # type: ignore
        return read_and_normalize_ct
    except Exception:
        return None


_READ_AND_NORMALIZE_CT = _try_import_project_normalizer()


def load_dicom_volume_normalized(dicom_dir: str, iso_spacing_mm: float = 0.5, target_orient: str = "LPS"):
    if _READ_AND_NORMALIZE_CT is not None:
        norm = _READ_AND_NORMALIZE_CT(dicom_dir, target_orient=target_orient, iso_spacing_mm=iso_spacing_mm)
        return norm.vol_zyx, norm.spacing_zyx_mm

    try:
        import SimpleITK as sitk
    except Exception as e:
        raise RuntimeError(
            "dicom_normalize.read_and_normalize_ct is not importable, and SimpleITK is also unavailable."
        ) from e

    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dicom_dir)
    if not series_ids:
        raise RuntimeError(f"No DICOM series found in: {dicom_dir}")

    file_names = reader.GetGDCMSeriesFileNames(dicom_dir, series_ids[0])
    reader.SetFileNames(file_names)
    img = reader.Execute()

    try:
        img = sitk.DICOMOrient(img, target_orient)
    except Exception:
        pass

    orig_spacing = np.array(list(img.GetSpacing()), dtype=float)
    orig_size = np.array(list(img.GetSize()), dtype=int)
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
    arr = sitk.GetArrayFromImage(img_rs).astype(np.float32)
    spacing_zyx = (float(new_spacing[2]), float(new_spacing[1]), float(new_spacing[0]))
    return arr, spacing_zyx


def _infer_id_from_json_path(json_path: str) -> str:
    base = os.path.basename(json_path)
    if base.endswith("_pred_center.json"):
        return base[: -len("_pred_center.json")]
    return os.path.splitext(base)[0]


def _resolve_dicom_dir(meta: dict, pid: str, dicom_root: Optional[str]) -> str:
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


def _resolve_iso_spacing(meta: dict, default_iso: float) -> float:
    ev = meta.get("eval", {})
    if isinstance(ev, dict) and ev.get("iso_spacing_mm") is not None:
        return float(ev["iso_spacing_mm"])
    if meta.get("iso_spacing_mm") is not None:
        return float(meta["iso_spacing_mm"])
    return float(default_iso)


def _to_int3(v):
    return (int(round(v[0])), int(round(v[1])), int(round(v[2])))


def _resolve_pred_center_zyx(meta: dict) -> Tuple[int, int, int]:
    if meta.get("pred_center_zyx") is not None:
        return _to_int3(meta["pred_center_zyx"])
    if meta.get("pred_center_zyx_canon_f") is not None:
        return _to_int3(meta["pred_center_zyx_canon_f"])
    raise ValueError("Missing pred_center_zyx / pred_center_zyx_canon_f in JSON.")


def _resolve_click_center_zyx(meta: dict) -> Tuple[int, int, int]:
    ev = meta.get("eval", {})
    gt_key = ev.get("gt_key") if isinstance(ev, dict) else None

    candidate_keys = []
    if gt_key:
        candidate_keys.extend([gt_key, f"{gt_key}_zyx", f"{gt_key}_canon_zyx", f"{gt_key}_full_zyx"])
    candidate_keys.extend([
        "vw_click_canon_full_zyx", "gt_center_zyx", "gt_center_zyx_canon",
        "click_center_zyx", "click_zyx", "vw_point_zyx", "center_zyx"
    ])

    for k in candidate_keys:
        if meta.get(k) is not None:
            return _to_int3(meta[k])

    raise ValueError("Could not resolve original click point from JSON.")


def _clip_center(center_zyx: Tuple[int, int, int], shape_zyx: Tuple[int, int, int]) -> Tuple[int, int, int]:
    cz, cy, cx = center_zyx
    return (
        int(np.clip(cz, 0, shape_zyx[0] - 1)),
        int(np.clip(cy, 0, shape_zyx[1] - 1)),
        int(np.clip(cx, 0, shape_zyx[2] - 1)),
    )


def _window_range(wl: float, ww: float) -> Tuple[float, float]:
    return wl - ww / 2.0, wl + ww / 2.0


def _extract_views(vol_zyx: np.ndarray, center_zyx: Tuple[int, int, int]):
    cz, cy, cx = center_zyx
    return vol_zyx[cz, :, :], vol_zyx[:, cy, :], vol_zyx[:, :, cx]


def _add_scale_bar(ax, image_shape_hw, spacing_col_mm, bar_mm=10.0,
                   color="white", text_color="white", origin="lower",
                   margin_px=12, line_width=3, fontsize=9):
    h, w = image_shape_hw
    bar_px = max(1.0, float(bar_mm) / float(spacing_col_mm))
    x0 = margin_px
    x1 = min(w - margin_px, x0 + bar_px)

    if origin == "upper":
        y = h - margin_px
        text_y = y - 6
        va = "bottom"
    else:
        y = margin_px
        text_y = y + 6
        va = "top"

    ax.plot([x0, x1], [y, y], color=color, linewidth=line_width, solid_capstyle="butt")
    ax.text((x0 + x1) / 2.0, text_y, f"{bar_mm:g} mm",
            color=text_color, fontsize=fontsize, ha="center", va=va)


def save_click_only_png(vol_zyx, click_zyx, spacing_zyx, out_png, wl=400.0, ww=4000.0,
                        title=None, dpi=300, marker_size=42.0, scalebar_mm=10.0):
    import matplotlib.pyplot as plt

    click_zyx = _clip_center(click_zyx, vol_zyx.shape)
    cz, cy, cx = click_zyx
    axial, coronal, sagittal = _extract_views(vol_zyx, click_zyx)
    vmin, vmax = _window_range(wl, ww)
    sz, sy, sx = spacing_zyx

    fig = plt.figure(figsize=(12, 4))
    if title:
        fig.suptitle(title)

    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(axial, cmap="gray", vmin=vmin, vmax=vmax, origin="upper")
    ax1.scatter([cx], [cy], s=marker_size, c="cyan", marker="x", linewidths=1.2)
    _add_scale_bar(ax1, axial.shape, spacing_col_mm=sx, bar_mm=scalebar_mm, origin="upper")
    ax1.set_title("Axial")
    ax1.axis("off")

    ax2 = plt.subplot(1, 3, 2)
    ax2.imshow(coronal, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
    ax2.scatter([cx], [cz], s=marker_size, c="cyan", marker="x", linewidths=1.2)
    _add_scale_bar(ax2, coronal.shape, spacing_col_mm=sx, bar_mm=scalebar_mm, origin="lower")
    ax2.set_title("Coronal")
    ax2.axis("off")

    ax3 = plt.subplot(1, 3, 3)
    ax3.imshow(sagittal, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
    ax3.scatter([cy], [cz], s=marker_size, c="cyan", marker="x", linewidths=1.2)
    _add_scale_bar(ax3, sagittal.shape, spacing_col_mm=sy, bar_mm=scalebar_mm, origin="lower")
    ax3.set_title("Sagittal")
    ax3.axis("off")

    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_click_and_pred_png(vol_zyx, click_zyx, pred_zyx, spacing_zyx, out_png, wl=400.0, ww=4000.0,
                            title=None, dpi=300, x_size=42.0, o_size=26.0, scalebar_mm=10.0):
    import matplotlib.pyplot as plt

    click_zyx = _clip_center(click_zyx, vol_zyx.shape)
    pred_zyx = _clip_center(pred_zyx, vol_zyx.shape)
    cz, cy, cx = click_zyx
    axial, coronal, sagittal = _extract_views(vol_zyx, click_zyx)
    vmin, vmax = _window_range(wl, ww)
    sz, sy, sx = spacing_zyx
    pz, py, px = pred_zyx

    fig = plt.figure(figsize=(12, 4))
    if title:
        fig.suptitle(title)

    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(axial, cmap="gray", vmin=vmin, vmax=vmax, origin="upper")
    ax1.scatter([cx], [cy], s=x_size, c="cyan", marker="x", linewidths=1.2)
    ax1.scatter([px], [py], s=o_size, c="orange", marker="o")
    _add_scale_bar(ax1, axial.shape, spacing_col_mm=sx, bar_mm=scalebar_mm, origin="upper")
    ax1.set_title("Axial")
    ax1.axis("off")

    ax2 = plt.subplot(1, 3, 2)
    ax2.imshow(coronal, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
    ax2.scatter([cx], [cz], s=x_size, c="cyan", marker="x", linewidths=1.2)
    ax2.scatter([px], [pz], s=o_size, c="orange", marker="o")
    _add_scale_bar(ax2, coronal.shape, spacing_col_mm=sx, bar_mm=scalebar_mm, origin="lower")
    ax2.set_title("Coronal")
    ax2.axis("off")

    ax3 = plt.subplot(1, 3, 3)
    ax3.imshow(sagittal, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
    ax3.scatter([cy], [cz], s=x_size, c="cyan", marker="x", linewidths=1.2)
    ax3.scatter([py], [pz], s=o_size, c="orange", marker="o")
    _add_scale_bar(ax3, sagittal.shape, spacing_col_mm=sy, bar_mm=scalebar_mm, origin="lower")
    ax3.set_title("Sagittal")
    ax3.axis("off")

    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roi_dir", default=DEFAULT_ROI_DIR)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--dicom_root", default=None)
    ap.add_argument("--ids", default=None)
    ap.add_argument("--target_orient", default="LPS")
    ap.add_argument("--default_iso_spacing", type=float, default=0.5)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--click_only", action="store_true")
    ap.add_argument("--both", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--wl", type=float, default=400.0)
    ap.add_argument("--ww", type=float, default=4000.0)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--add_title", action="store_true")
    ap.add_argument("--scalebar_mm", type=float, default=10.0)
    ap.add_argument("--x_size", type=float, default=42.0)
    ap.add_argument("--o_size", type=float, default=26.0)
    args = ap.parse_args()

    if not (args.click_only or args.both or args.all):
        args.all = True

    want_click_only = args.click_only or args.all
    want_both = args.both or args.all

    pattern = os.path.join(args.roi_dir, "**", "*_pred_center.json") if args.recursive else os.path.join(args.roi_dir, "*_pred_center.json")
    json_paths = sorted(glob.glob(pattern, recursive=args.recursive))
    if args.ids:
        want_ids = set(x.strip() for x in args.ids.split(",") if x.strip())
        json_paths = [p for p in json_paths if _infer_id_from_json_path(p) in want_ids]
    if not json_paths:
        raise SystemExit(f"No *_pred_center.json found under: {args.roi_dir}")

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    ok = 0
    failed = 0

    for jp in json_paths:
        pid = _infer_id_from_json_path(jp)
        try:
            with open(jp, "r", encoding="utf-8") as f:
                meta = json.load(f)

            dicom_dir = _resolve_dicom_dir(meta, pid, args.dicom_root)
            pred_zyx = _resolve_pred_center_zyx(meta)
            click_zyx = _resolve_click_center_zyx(meta)
            iso_spacing = _resolve_iso_spacing(meta, args.default_iso_spacing)
            flipped_lr = bool(meta.get("flipped_lr", False))

            vol, spacing_zyx = load_dicom_volume_normalized(
                dicom_dir,
                iso_spacing_mm=iso_spacing,
                target_orient=args.target_orient,
            )
            if flipped_lr:
                vol = vol[:, :, ::-1].copy()

            save_dir = args.out_dir if args.out_dir else os.path.dirname(jp)
            os.makedirs(save_dir, exist_ok=True)

            title = None
            if args.add_title:
                dist_mm = float(np.linalg.norm(np.array(pred_zyx, dtype=float) - np.array(click_zyx, dtype=float))) * iso_spacing
                title = f"{pid} dist={dist_mm:.2f}mm"

            if want_click_only:
                out_png = os.path.join(save_dir, f"CLICK_{pid}_click_only.png")
                if args.overwrite or (not os.path.exists(out_png)):
                    save_click_only_png(vol, click_zyx, spacing_zyx, out_png,
                                        wl=args.wl, ww=args.ww, title=title, dpi=args.dpi,
                                        marker_size=args.x_size, scalebar_mm=args.scalebar_mm)

            if want_both:
                out_png = os.path.join(save_dir, f"BOTH_{pid}_click_and_pred.png")
                if args.overwrite or (not os.path.exists(out_png)):
                    save_click_and_pred_png(vol, click_zyx, pred_zyx, spacing_zyx, out_png,
                                            wl=args.wl, ww=args.ww, title=title, dpi=args.dpi,
                                            x_size=args.x_size, o_size=args.o_size,
                                            scalebar_mm=args.scalebar_mm)

            ok += 1
        except Exception as e:
            failed += 1
            print(f"[FAIL] {pid}: {e}")

    print(f"[DONE] ok={ok}, failed={failed}")


if __name__ == "__main__":
    main()
