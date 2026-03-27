#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Assembles Supplementary Figure S1 from three localization QA images and
ROI center error statistics from pred_center_errors.csv.

Panels:
(A) Clicked point only
(B) Clicked point and predicted point
(C) The worst case in prediction
(D) Histogram of ROI center error
(E) CDF of ROI center error
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .common import compute_cdf, resolve_out_dir, save_png_pdf

DIST_CANDS = [
    "dist_mm", "distance_mm", "error_mm", "center_error_mm", "pred_center_error_mm",
    "dist", "distance", "error", "center_error"
]
DX_CANDS = ["dx_mm", "dx", "err_x_mm", "x_mm"]
DY_CANDS = ["dy_mm", "dy", "err_y_mm", "y_mm"]
DZ_CANDS = ["dz_mm", "dz", "err_z_mm", "z_mm"]

def _find_col(df, candidates):
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    for cand in candidates:
        for c in df.columns:
            if cand.lower() in c.lower():
                return c
    return None

def _get_dist_mm(df):
    cdist = _find_col(df, DIST_CANDS)
    if cdist is not None:
        return pd.to_numeric(df[cdist], errors="coerce").to_numpy(dtype=float)
    cx = _find_col(df, DX_CANDS)
    cy = _find_col(df, DY_CANDS)
    cz = _find_col(df, DZ_CANDS)
    if cx is None or cy is None or cz is None:
        raise ValueError(f"Could not infer ROI-center error columns from: {list(df.columns)}")
    dx = pd.to_numeric(df[cx], errors="coerce").to_numpy(dtype=float)
    dy = pd.to_numeric(df[cy], errors="coerce").to_numpy(dtype=float)
    dz = pd.to_numeric(df[cz], errors="coerce").to_numpy(dtype=float)
    return np.sqrt(dx*dx + dy*dy + dz*dz)

def _read_image(path):
    arr = plt.imread(path)
    if arr.ndim == 2:
        return arr
    if arr.shape[-1] == 4:
        return arr[..., :3]
    return arr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--click_only_png", required=True)
    ap.add_argument("--click_pred_png", required=True)
    ap.add_argument("--worst_case_png", required=True)
    ap.add_argument("--csv", required=True, help="pred_center_errors.csv")
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--bins", type=int, default=20)
    ap.add_argument("--title", default="Supplementary Fig. S1 | ROI localization examples and center error distribution")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    dist = _get_dist_mm(df)
    dist = dist[np.isfinite(dist)]
    if dist.size == 0:
        raise ValueError("No valid distance values found.")

    out_dir = resolve_out_dir(Path(args.csv), args.out_dir)

    fig = plt.figure(figsize=(11, 8.4))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 0.9])

    imgs = [
        (_read_image(args.click_only_png), "A. Clicked point only"),
        (_read_image(args.click_pred_png), "B. Clicked point and predicted point"),
        (_read_image(args.worst_case_png), "C. The worst case in prediction"),
    ]
    positions = [(0, slice(0,2)), (0, slice(2,4)), (0, slice(4,6))]
    for (img, ttl), (r, csl) in zip(imgs, positions):
        ax = fig.add_subplot(gs[r, csl])
        ax.imshow(img, cmap="gray")
        ax.set_title(ttl, fontsize=11)
        ax.axis("off")

    ax4 = fig.add_subplot(gs[1, 0:3])
    ax4.hist(dist, bins=args.bins)
    ax4.set_title("D. Histogram of ROI center error", fontsize=11)
    ax4.set_xlabel("ROI center error (mm)")
    ax4.set_ylabel("Count")

    ax5 = fig.add_subplot(gs[1, 3:6])
    xs, ys = compute_cdf(dist)
    ax5.plot(xs, ys)
    for thr in [2, 5]:
        ax5.axvline(thr, linestyle="--", linewidth=0.8, alpha=0.5)
    ax5.set_title("E. CDF of ROI center error", fontsize=11)
    ax5.set_xlabel("ROI center error (mm)")
    ax5.set_ylabel("Cumulative proportion")
    ax5.set_ylim(0, 1.01)

    fig.suptitle(args.title, y=0.98, fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    png, pdf = save_png_pdf(fig, out_dir, "FigS1_roi_localization_and_error", dpi=args.dpi)
    plt.close(fig)
    print("[OK] Saved:")
    print("  ", png)
    print("  ", pdf)

if __name__ == "__main__":
    main()
