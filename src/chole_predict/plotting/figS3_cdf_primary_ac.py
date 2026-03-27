#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Creates Supplementary Figure S3: CDF of absolute error for primary endpoint
(post-op AC PTA mean, OOF), comparing Tabular / Residual / Gated.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .common import build_primary_ac_cols, build_true_primary_ac_cols, compute_cdf, resolve_out_dir, save_png_pdf

MODEL_COLOR = {"Tabular": "tab:blue", "Residual": "tab:orange", "Gated": "tab:green"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_patient_csv", required=True, help="per_patient_results.csv")
    ap.add_argument("--out_dir", default="", help="optional output directory")
    ap.add_argument("--suffix", default="", help="optional suffix for filenames")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    in_path = Path(args.per_patient_csv)
    df = pd.read_csv(in_path)

    true_cols = build_true_primary_ac_cols()
    pred_cols = {
        "Tabular": build_primary_ac_cols("pred_tab"),
        "Residual": build_primary_ac_cols("pred_resid"),
        "Gated": build_primary_ac_cols("pred_gated"),
    }
    need = true_cols + pred_cols["Tabular"] + pred_cols["Residual"] + pred_cols["Gated"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns (showing up to 15): {missing[:15]}")

    true = df[true_cols].to_numpy(dtype=float)
    err = {}
    for name, cols in pred_cols.items():
        pred = df[cols].to_numpy(dtype=float)
        arr = np.nanmean(np.abs(pred - true), axis=1)
        err[name] = arr[np.isfinite(arr)]

    out_dir = resolve_out_dir(in_path, args.out_dir)
    stem = f"FigS3_cdf_primary_ac_ptamean_harm{args.suffix}"

    plt.figure(figsize=(8, 6))
    for name in ["Tabular", "Residual", "Gated"]:
        xs, ys = compute_cdf(err[name])
        plt.plot(xs, ys, label=name, color=MODEL_COLOR[name], linewidth=2)
    plt.xlabel("Absolute error |ŷ − y| (dB)")
    plt.ylabel("Cumulative proportion")
    plt.title("Supplementary Fig. S3 | CDF of |error| (PTA mean, AC)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    png_path, pdf_path = save_png_pdf(plt.gcf(), out_dir, stem, dpi=args.dpi)
    plt.close()
    print("[OK] Saved:")
    print("  ", png_path)
    print("  ", pdf_path)

if __name__ == "__main__":
    main()
