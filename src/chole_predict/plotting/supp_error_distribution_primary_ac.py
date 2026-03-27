#!/usr/bin/env python
# -*- coding: utf-8 -*-
# NOTE: This script is kept as an optional / reviewer-response figure 
# and is not mapped to the current final figure numbering.
"""
make_figSx_error_distribution_primary_ac_ptamean.py

Creates Supplementary Figure Sx: distribution of per-patient absolute error |yhat - y| (dB)
for Primary endpoint (post-op AC PTA mean), comparing TAB / RESID / GATED.

Input:
  --per_patient_csv per_patient_results.csv

Output (written next to input unless --out_dir is provided):
  - FigSx_error_distribution_primary_ac_ptamean.png
  - FigSx_error_distribution_primary_ac_ptamean.pdf

Notes:
- PTA mean (AC) is computed as the mean of 0.5/1/2/3 kHz thresholds (A=air conduction).
- Uses matplotlib defaults (no explicit colors/styles).
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .common import build_primary_ac_cols, build_true_primary_ac_cols, compute_cdf, resolve_out_dir, save_png_pdf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_patient_csv", required=True, help="per_patient_results.csv")
    ap.add_argument("--out_dir", default="", help="optional output directory")
    ap.add_argument("--suffix", default="", help="optional suffix for filenames, e.g. _v2")
    args = ap.parse_args()

    in_path = Path(args.per_patient_csv)
    df = pd.read_csv(in_path)

    true_cols = build_true_primary_ac_cols()
    pred_tab_cols = build_primary_ac_cols("pred_tab")
    pred_resid_cols = build_primary_ac_cols("pred_resid")
    pred_gated_cols = build_primary_ac_cols("pred_gated")

    missing = [c for c in (true_cols + pred_tab_cols + pred_resid_cols + pred_gated_cols) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns (showing up to 15): {missing[:15]}")

    true = df[true_cols].to_numpy(dtype=float)
    tab = df[pred_tab_cols].to_numpy(dtype=float)
    resid = df[pred_resid_cols].to_numpy(dtype=float)
    gated = df[pred_gated_cols].to_numpy(dtype=float)

    abs_err_tab = np.nanmean(np.abs(tab - true), axis=1)
    abs_err_resid = np.nanmean(np.abs(resid - true), axis=1)
    abs_err_gated = np.nanmean(np.abs(gated - true), axis=1)

    order = ["Tabular", "Residual", "Gated"]
    data = [abs_err_tab[~np.isnan(abs_err_tab)],
            abs_err_resid[~np.isnan(abs_err_resid)],
            abs_err_gated[~np.isnan(abs_err_gated)]]

    # Output paths
    out_dir = resolve_out_dir(in_path, args.out_dir)
    suf = args.suffix
    stem = f"FigSx_error_distribution_primary_ac_ptamean{suf}"

    plt.figure(figsize=(10, 4.5))

    # (A) Violin + box
    ax1 = plt.subplot(1, 2, 1)
    ax1.violinplot(data, showmeans=False, showmedians=False, showextrema=False)
    ax1.boxplot(data, labels=order, showfliers=True)
    ax1.set_ylabel("Absolute error |ŷ − y| (dB)")
    ax1.set_title("(A) Distribution of per-patient |error| (PTA mean, AC)")
    ax1.grid(axis="y", alpha=0.3)

    # (B) CDF
    ax2 = plt.subplot(1, 2, 2)
    for m, arr in zip(order, data):
        xs, ys = compute_cdf(arr)
        ax2.plot(xs, ys, label=m)
    ax2.set_xlabel("Absolute error |ŷ − y| (dB)")
    ax2.set_ylabel("Cumulative proportion")
    ax2.set_title("(B) CDF of |error|")
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.suptitle("Supplementary Figure Sx. Primary endpoint: distribution of prediction error (OOF, PTA mean AC)",
                 y=1.02, fontsize=12)
    plt.tight_layout()
    png_path, pdf_path = save_png_pdf(plt.gcf(), out_dir, stem, dpi=200)
    plt.close()

    # Print a small summary
    mae = [float(np.mean(arr)) for arr in data]
    print("[OK] Saved:")
    print("  ", png_path)
    print("  ", pdf_path)
    print("[Summary] Mean |error| (dB):")
    for m, v in zip(order, mae):
        print(f"  {m}: {v:.3f}")

if __name__ == "__main__":
    main()
