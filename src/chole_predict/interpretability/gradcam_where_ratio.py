#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
compute_cam_where_ratio_v4.py

Compute "CAM where-ratio" metrics and boxplots by group (improved/neutral/worsened).

Supports your layout:
  <cam_root>/fold1/<ID>/<cam_name>
Example (Windows):
  C:\\Users\\path\\to\\resid_cam_yhat_upsampled.npy

Key features
- Resolves cam_root relative to CWD / exp_root / project_root (same as ROI path resolution)
- Recursively finds CAM file under cam_root/**/<ID>/<cam_name>
- Optionally uses fold column in case_list_csv
- Loads HU ROI volume from base_csv[roi_col] (.npy)
- Works for 2D or 3D CAM; if 3D, selects axial z slice by "top_pct_mean" within HU mask
- Outputs:
    exp_root/out_dir/cam_where_metrics.csv
    exp_root/out_dir/box_*.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------
# IO helpers
# -------------------------
def load_npy_or_npz(path: Path) -> np.ndarray:
    suf = path.suffix.lower()
    if suf == ".npy":
        return np.load(path)
    if suf == ".npz":
        z = np.load(path)
        if "cam" in z.files:
            return z["cam"]
        return z[z.files[0]]
    raise ValueError(f"Unsupported file: {path}")


def safe_mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def resolve_path(p: str, exp_root: Path, project_root: Path | None) -> Path:
    """
    Resolve a path that may be absolute or relative.
    Tries:
      1) as-is relative to CWD
      2) relative to exp_root
      3) relative to project_root (if provided)
    """
    pp = Path(p)
    if pp.is_absolute():
        return pp

    cand = Path.cwd() / pp
    if cand.exists():
        return cand

    cand = exp_root / pp
    if cand.exists():
        return cand

    if project_root is not None:
        cand = project_root / pp
        if cand.exists():
            return cand

    return cand


def find_cam_file(cam_root: Path, case_id: str, cam_name: str, fold: str | None = None) -> Path | None:
    """
    Search target file:
      cam_root/fold1/<ID>/<cam_name>
    If fold is None, search any fold:
      cam_root/**/<ID>/<cam_name>
    """
    cid = str(case_id)

    if fold is not None and str(fold).strip() != "":
        candidates = [
            cam_root / f"fold{fold}" / cid / cam_name,
            cam_root / f"fold_{fold}" / cid / cam_name,
            cam_root / str(fold) / cid / cam_name,
        ]
        for p in candidates:
            if p.exists():
                return p

    matches = list(cam_root.glob(f"**/{cid}/{cam_name}"))
    if matches:
        return sorted(matches)[0]
    return None


# -------------------------
# Region masks
# -------------------------
def center_disk_mask(h: int, w: int, center_frac: float = 0.35) -> np.ndarray:
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r = center_frac * min(h, w) / 2.0
    yy, xx = np.ogrid[:h, :w]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= (r ** 2)


def border_mask(h: int, w: int, border_px: int = 0) -> np.ndarray:
    if border_px <= 0:
        return np.zeros((h, w), dtype=bool)
    m = np.zeros((h, w), dtype=bool)
    m[:border_px, :] = True
    m[-border_px:, :] = True
    m[:, :border_px] = True
    m[:, -border_px:] = True
    return m


def rect_mask(h: int, w: int, x0: float, x1: float, y0: float, y1: float) -> np.ndarray:
    x0i = int(np.floor(x0 * w))
    x1i = int(np.ceil(x1 * w))
    y0i = int(np.floor(y0 * h))
    y1i = int(np.ceil(y1 * h))
    x0i, x1i = np.clip([x0i, x1i], 0, w)
    y0i, y1i = np.clip([y0i, y1i], 0, h)
    m = np.zeros((h, w), dtype=bool)
    m[y0i:y1i, x0i:x1i] = True
    return m


def _nan_if_zero(num: float, denom: float) -> float:
    return float(num / denom) if denom > 0 else float("nan")


# -------------------------
# Core metrics
# -------------------------
def compute_metrics_2d(
    cam2d: np.ndarray,
    hu2d: np.ndarray,
    hu_low: float = -1000,
    hu_high: float = 300,
    top_pct: float = 0.05,
    center_frac: float = 0.35,
    border_px: int = 2,
    inner_rect=(0.55, 1.00, 0.00, 0.45),
) -> dict:
    cam2d = cam2d.astype(np.float32)
    hu2d = hu2d.astype(np.float32)
    h, w = cam2d.shape

    hu_mask = (hu2d >= hu_low) & (hu2d <= hu_high)

    bmask = border_mask(h, w, border_px=border_px)
    eval_mask = hu_mask & (~bmask)

    if eval_mask.sum() < 10:
        return {
            "n_eval_pix": int(eval_mask.sum()),
            "cam_sum": np.nan,
            "center_energy_frac": np.nan,
            "periph_energy_frac": np.nan,
            "inner_rect_energy_frac": np.nan,
            "border_energy_frac": np.nan,
            "topk_center_frac": np.nan,
            "topk_periph_frac": np.nan,
            "topk_inner_rect_frac": np.nan,
            "peak_dist_norm": np.nan,
            "topk_thr": np.nan,
            "topk_count": 0,
        }

    cam_pos = np.maximum(cam2d, 0.0)
    cam_masked = np.where(eval_mask, cam_pos, 0.0)
    cam_sum = float(cam_masked.sum())

    cdisk = center_disk_mask(h, w, center_frac=center_frac)
    center = eval_mask & cdisk
    periph = eval_mask & (~cdisk)
    inner = rect_mask(h, w, *inner_rect) & eval_mask

    border_on_hu = hu_mask & bmask
    cam_border = float(np.where(border_on_hu, cam_pos, 0.0).sum())
    cam_hu_sum = float(np.where(hu_mask, cam_pos, 0.0).sum())

    cam_vals = cam_pos[eval_mask]
    thr = float(np.quantile(cam_vals, 1.0 - top_pct))
    topk = eval_mask & (cam_pos >= thr)

    peak_idx = np.unravel_index(np.argmax(cam_masked), cam_masked.shape)
    py, px = peak_idx
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    dist = float(np.sqrt((py - cy) ** 2 + (px - cx) ** 2))
    half_diag = float(np.sqrt((cy) ** 2 + (cx) ** 2))
    peak_dist_norm = dist / half_diag if half_diag > 0 else float("nan")

    return {
        "n_eval_pix": int(eval_mask.sum()),
        "cam_sum": cam_sum,
        "center_energy_frac": _nan_if_zero(float(np.where(center, cam_pos, 0.0).sum()), cam_sum),
        "periph_energy_frac": _nan_if_zero(float(np.where(periph, cam_pos, 0.0).sum()), cam_sum),
        "inner_rect_energy_frac": _nan_if_zero(float(np.where(inner, cam_pos, 0.0).sum()), cam_sum),
        "border_energy_frac": _nan_if_zero(cam_border, cam_hu_sum),
        "topk_center_frac": _nan_if_zero(float((topk & center).sum()), float(topk.sum())),
        "topk_periph_frac": _nan_if_zero(float((topk & periph).sum()), float(topk.sum())),
        "topk_inner_rect_frac": _nan_if_zero(float((topk & inner).sum()), float(topk.sum())),
        "peak_dist_norm": float(peak_dist_norm),
        "topk_thr": thr,
        "topk_count": int(topk.sum()),
    }


def _to_zhw(vol: np.ndarray, ref: np.ndarray | None = None) -> np.ndarray:
    if vol.ndim != 3:
        raise ValueError("Expected 3D volume")
    if ref is not None and ref.ndim == 3:
        if vol.shape == ref.shape:
            return vol
        if vol.shape[-1] == ref.shape[0] and vol.shape[:-1] == ref.shape[1:]:
            return np.moveaxis(vol, -1, 0)
    return vol


def select_z_by_cam_top_pct_mean(
    cam_zhw: np.ndarray,
    hu_zhw: np.ndarray,
    hu_low: float,
    hu_high: float,
    top_pct: float = 0.05,
    min_count: int = 100
) -> tuple[int, float]:
    Z, _, _ = cam_zhw.shape
    scores = np.full((Z,), -np.inf, dtype=np.float32)
    for z in range(Z):
        cam2d = np.maximum(cam_zhw[z].astype(np.float32), 0.0)
        hu2d = hu_zhw[z].astype(np.float32)
        m = (hu2d >= hu_low) & (hu2d <= hu_high)
        vals = cam2d[m]
        if vals.size < min_count:
            continue
        thr = np.quantile(vals, 1.0 - top_pct)
        top = vals[vals >= thr]
        if top.size < max(10, int(min_count * top_pct * 0.5)):
            continue
        scores[z] = float(top.mean())
    best_z = int(np.argmax(scores))
    best_score = float(scores[best_z])
    if not np.isfinite(best_score):
        best_z = int(Z // 2)
        best_score = float("nan")
    return best_z, best_score


# -------------------------
# Plotting
# -------------------------
def boxplot_by_group(df: pd.DataFrame, group_col: str, metric: str, out_png: Path,
                     title: str | None = None, ylabel: str | None = None):
    groups = ["improved", "neutral", "worsened"]
    data = [df.loc[df[group_col] == g, metric].dropna().values for g in groups]
    plt.figure(figsize=(6, 4))
    plt.boxplot(data, labels=groups, showfliers=True)
    plt.ylabel(ylabel or metric)
    plt.title(title or metric)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case_list_csv", required=True)
    ap.add_argument("--base_csv", required=True)
    ap.add_argument("--roi_col", required=True)

    ap.add_argument("--exp_root", required=True,
                    help="Experiment root, e.g. experiments\\20260218_143700_CroppedROI_out8_AB")
    ap.add_argument("--project_root", default="",
                    help="Optional: chole_predict root to resolve relative paths (recommended).")

    ap.add_argument("--cam_root", required=True,
                    help="GradCAM root directory (absolute or relative).")
    ap.add_argument("--cam_name", default="resid_cam_yhat_upsampled.npy")
    ap.add_argument("--use_fold_in_case_list", action="store_true")

    ap.add_argument("--out_dir", default="_cam_where_metrics")

    ap.add_argument("--hu_low", type=float, default=-1000)
    ap.add_argument("--hu_high", type=float, default=300)
    ap.add_argument("--top_pct", type=float, default=0.05)
    ap.add_argument("--center_frac", type=float, default=0.35)
    ap.add_argument("--border_px", type=int, default=2)

    ap.add_argument("--inner_rect", type=float, nargs=4, default=[0.55, 1.00, 0.00, 0.45],
                    metavar=("X0", "X1", "Y0", "Y1"))
    ap.add_argument("--force_z", type=int, default=-1)
    ap.add_argument("--min_count", type=int, default=100)

    args = ap.parse_args()

    exp_root = Path(args.exp_root)
    project_root = Path(args.project_root) if args.project_root.strip() else None

    cam_root = resolve_path(str(args.cam_root), exp_root=exp_root, project_root=project_root)
    if not cam_root.exists():
        raise FileNotFoundError(
            f"cam_root not found: {cam_root}\n"
            f"  tried relative to CWD / exp_root / project_root.\n"
            f"  args.cam_root={args.cam_root}\n"
            f"  exp_root={exp_root}\n"
            f"  project_root={project_root}"
        )

    out_dir = safe_mkdir(exp_root / args.out_dir)

    case_df = pd.read_csv(args.case_list_csv)
    base_df = pd.read_csv(args.base_csv)

    # ID column
    id_col = None
    for c in case_df.columns:
        if c.lower() in ("id", "case_id", "seriesid", "series_id"):
            id_col = c
            break
    if id_col is None:
        raise ValueError(f"Cannot find ID column in case_list_csv. Columns={list(case_df.columns)}")

    # group column (improved/neutral/worsened)
    group_col = None
    for c in case_df.columns:
        if c.lower() in ("group", "label", "category"):
            group_col = c
            break
    if group_col is None:
        for c in case_df.columns:
            if "group" in c.lower():
                group_col = c
                break
    if group_col is None and "stratum" in case_df.columns:
        group_col = "stratum"
    if group_col is None:
        if "delta_db" not in case_df.columns:
            raise ValueError(f"Cannot find group column and no delta_db to derive it. Columns={list(case_df.columns)}")
        th = 5.0
        d = pd.to_numeric(case_df["delta_db"], errors="coerce")
        case_df["group"] = np.where(d >= th, "improved",
                          np.where(d <= -th, "worsened", "neutral"))
        group_col = "group"

    case_df[group_col] = case_df[group_col].astype(str).str.strip().str.lower()
    case_df[group_col] = case_df[group_col].str.replace(r"_\d+$", "", regex=True)

    if case_df[group_col].isin(["improved", "neutral", "worsened"]).sum() == 0 and "delta_db" in case_df.columns:
        th = 5.0
        d = pd.to_numeric(case_df["delta_db"], errors="coerce")
        case_df["group"] = np.where(d >= th, "improved",
                          np.where(d <= -th, "worsened", "neutral"))
        group_col = "group"

    # fold col (optional)
    fold_col = None
    if args.use_fold_in_case_list:
        for c in case_df.columns:
            if c.lower() == "fold":
                fold_col = c
                break

    # base ID column
    base_id_col = None
    for c in base_df.columns:
        if c.lower() in ("id", "case_id", "seriesid", "series_id"):
            base_id_col = c
            break
    if base_id_col is None:
        raise ValueError(f"Cannot find ID column in base_csv. Columns={list(base_df.columns)}")
    if args.roi_col not in base_df.columns:
        raise ValueError(f"roi_col '{args.roi_col}' not found in base_csv columns.")

    roi_map = dict(zip(base_df[base_id_col].astype(str), base_df[args.roi_col].astype(str)))

    rows = []
    missing_cam = 0
    missing_roi = 0
    shape_mismatch = 0

    for _, r in case_df.iterrows():
        cid = str(r[id_col])
        grp = str(r[group_col]).strip().lower()

        fold_val = None
        if fold_col is not None:
            try:
                fold_val = str(int(r[fold_col]))
            except Exception:
                fold_val = str(r[fold_col])

        cam_path = find_cam_file(cam_root, cid, args.cam_name, fold=fold_val)
        if cam_path is None:
            missing_cam += 1
            continue

        roi_path_str = roi_map.get(cid, None)
        if roi_path_str is None or str(roi_path_str).lower() == "nan":
            missing_roi += 1
            continue

        roi_path = resolve_path(str(roi_path_str), exp_root=exp_root, project_root=project_root)
        if not roi_path.exists():
            missing_roi += 1
            continue

        cam_arr = load_npy_or_npz(cam_path)
        hu_arr = np.load(roi_path)

        info = {
            "id": cid,
            "group": grp,
            "fold": fold_val if fold_val is not None else "",
            "cam_file": str(cam_path),
            "roi_file": str(roi_path),
        }

        if cam_arr.ndim == 2:
            if hu_arr.ndim == 3:
                z = args.force_z if args.force_z >= 0 else (hu_arr.shape[0] // 2)
                hu2d = hu_arr[int(z)]
                info["z_used"] = int(z)
                info["z_score"] = np.nan
            elif hu_arr.ndim == 2:
                hu2d = hu_arr
                info["z_used"] = -1
                info["z_score"] = np.nan
            else:
                shape_mismatch += 1
                continue

            m = compute_metrics_2d(
                cam2d=cam_arr,
                hu2d=hu2d,
                hu_low=args.hu_low,
                hu_high=args.hu_high,
                top_pct=args.top_pct,
                center_frac=args.center_frac,
                border_px=args.border_px,
                inner_rect=tuple(args.inner_rect),
            )
            rows.append({**info, **m})
            continue

        if cam_arr.ndim == 3:
            if hu_arr.ndim != 3:
                shape_mismatch += 1
                continue

            cam_zhw = _to_zhw(cam_arr, ref=hu_arr)
            hu_zhw = _to_zhw(hu_arr, ref=cam_zhw)

            if args.force_z >= 0:
                z = int(args.force_z)
                z_score = float("nan")
            else:
                z, z_score = select_z_by_cam_top_pct_mean(
                    cam_zhw, hu_zhw,
                    hu_low=args.hu_low,
                    hu_high=args.hu_high,
                    top_pct=args.top_pct,
                    min_count=args.min_count
                )

            info["z_used"] = int(z)
            info["z_score"] = float(z_score)

            cam2d = cam_zhw[int(z)]
            hu2d = hu_zhw[int(z)]

            m = compute_metrics_2d(
                cam2d=cam2d,
                hu2d=hu2d,
                hu_low=args.hu_low,
                hu_high=args.hu_high,
                top_pct=args.top_pct,
                center_frac=args.center_frac,
                border_px=args.border_px,
                inner_rect=tuple(args.inner_rect),
            )
            rows.append({**info, **m})
            continue

        shape_mismatch += 1

    df = pd.DataFrame(rows)
    out_csv = out_dir / "cam_where_metrics.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    metrics = [
        ("center_energy_frac", "Center energy fraction (HU-masked)"),
        ("periph_energy_frac", "Peripheral energy fraction (HU-masked)"),
        ("inner_rect_energy_frac", "Inner-rect energy fraction (proxy; HU-masked)"),
        ("border_energy_frac", "Border energy fraction (on HU mask)"),
        ("topk_center_frac", "Top-k center occupancy"),
        ("topk_inner_rect_frac", "Top-k inner-rect occupancy"),
        ("peak_dist_norm", "Peak distance (normalized)"),
        ("z_score", "Auto-z score (top-pct mean)"),
    ]
    for met, ylabel in metrics:
        png = out_dir / f"box_{met}.png"
        boxplot_by_group(df, "group", met, png, title=met, ylabel=ylabel)

    print(f"[OK] wrote: {out_csv}")
    print(f"[OK] plots in: {out_dir}")
    print(f"[INFO] n_rows={len(df)} missing_cam={missing_cam} missing_roi={missing_roi} shape_mismatch={shape_mismatch}")
    print(f"[INFO] cam_root={cam_root}")
    print(f"[INFO] cam_name={args.cam_name}")
    print(f"[INFO] groups: {df['group'].value_counts(dropna=False).to_dict() if len(df) else {}}")


if __name__ == "__main__":
    main()
