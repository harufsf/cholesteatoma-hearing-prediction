#!/usr/bin/env python
# -*- coding: utf-8 -*-
NOTE: This script is kept as a reusable component. In the current final figure order, ROI center error histogram/CDF is integrated into Fig. S1 rather than used as a standalone figure.
r"""
make_supp_figS3_roi_center_error_hist_cdf.py

Supplementary Fig. S3 (2-panel): ROI center error distribution from pred_center_errors.csv
  (A) Histogram of ROI center error distance (mm)
  (B) CDF of ROI center error distance (mm)

Input:
  --csv  Path to pred_center_errors.csv

Output (to --out_dir or alongside input):
  - SuppFigS3_ROI_center_error_hist_cdf.png
  - SuppFigS3_ROI_center_error_hist_cdf.pdf

Design requirements:
- Title uses "Supplementary Fig. S3"
- No gridlines
- No in-figure numeric annotations (n, median, etc.)
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

THR_LIST = [2, 5, 10]  # shown as vertical reference lines on CDF


def _find_col(df: pd.DataFrame, candidates):
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    for cand in candidates:
        for c in df.columns:
            if cand.lower() in c.lower():
                return c
    return None


def _get_dist_mm(df: pd.DataFrame) -> np.ndarray:
    cdist = _find_col(df, DIST_CANDS)
    if cdist is not None:
        return pd.to_numeric(df[cdist], errors="coerce").to_numpy(dtype=float)

    cx = _find_col(df, DX_CANDS)
    cy = _find_col(df, DY_CANDS)
    cz = _find_col(df, DZ_CANDS)
    if cx is None or cy is None or cz is None:
        raise ValueError(
            "Could not find a distance column (e.g., dist_mm) and could not find dx/dy/dz columns to compute it.\n"
            f"Columns: {list(df.columns)}"
        )
    dx = pd.to_numeric(df[cx], errors="coerce").to_numpy(dtype=float)
    dy = pd.to_numeric(df[cy], errors="coerce").to_numpy(dtype=float)
    dz = pd.to_numeric(df[cz], errors="coerce").to_numpy(dtype=float)
    return np.sqrt(dx*dx + dy*dy + dz*dz)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to pred_center_errors.csv")
    ap.add_argument("--out_dir", default="", help="Output directory (default: alongside input)")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--bins", type=int, default=30)
    ap.add_argument("--xmax", type=float, default=-1.0, help="Optional x-axis max (mm). If <0, auto.")
    args = ap.parse_args()

    in_path = Path(args.csv)
    df = pd.read_csv(in_path)

    dist = _get_dist_mm(df)
    dist = dist[np.isfinite(dist)]
    if dist.size == 0:
        raise ValueError("No valid distance values after filtering NaNs.")

    out_dir = resolve_out_dir(in_path, args.out_dir)

    # x-range
    if args.xmax > 0:
        xmax = float(args.xmax)
    else:
        xmax = float(np.quantile(dist, 0.995))
        xmax = max(xmax, float(dist.max()))
    xmax = max(xmax, 1.0)

    plt.figure(figsize=(10.5, 4.2))

    # (A) histogram
    ax1 = plt.subplot(1, 2, 1)
    ax1.hist(dist, bins=args.bins)
    ax1.set_title("(A) Histogram")
    ax1.set_xlabel("ROI center error (mm)")
    ax1.set_ylabel("Count")
    ax1.set_xlim(0, xmax)

    # (B) CDF
    ax2 = plt.subplot(1, 2, 2)
    xs, ys = compute_cdf(dist)
    ax2.plot(xs, ys)
    ax2.set_title("(B) CDF")
    ax2.set_xlabel("ROI center error (mm)")
    ax2.set_ylabel("Cumulative proportion")
    ax2.set_xlim(0, xmax)
    ax2.set_ylim(0, 1.01)

    # reference thresholds as dashed vertical lines (no labels, to keep the figure clean)
    for t in THR_LIST:
        ax2.axvline(t, linestyle="--", linewidth=1)

    plt.suptitle("Supplementary Fig. S3 | ROI center error distribution", y=1.03, fontsize=12)
    plt.tight_layout()
    png, pdf = save_png_pdf(plt.gcf(), out_dir, "SuppFigS3_ROI_center_error_hist_cdf", dpi=args.dpi)
    plt.close()

    print("[OK] Saved:")
    print(" ", png)
    print(" ", pdf)


if __name__ == "__main__":
    main()
