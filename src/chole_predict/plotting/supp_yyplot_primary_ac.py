#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_figS2_yyplot_primary_ac_ptamean.py

Creates a 3-panel y-y plot (observed vs predicted) for Primary endpoint (post-op AC PTA mean):
  (A) Tabular   (B) Residual   (C) Gated

Changes vs original:
- Consistent colors with the manuscript: Tabular=blue, Residual=orange, Gated=green
- Gridlines removed

Input:
  --per_patient_csv per_patient_results.csv

Output:
  - FigS2_yyplot_primary_ac_ptamean.png
  - FigS2_yyplot_primary_ac_ptamean.pdf
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .common import axis_limits_with_padding, build_primary_ac_cols, build_true_primary_ac_cols, pta_mean, resolve_out_dir, save_png_pdf

MODEL_COLOR = {"Tabular": "tab:blue", "Residual": "tab:orange", "Gated": "tab:green"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_patient_csv", required=True, help="per_patient_results.csv")
    ap.add_argument("--out_dir", default="", help="optional output directory")
    ap.add_argument("--suffix", default="", help="optional suffix for filenames, e.g. _v2")
    ap.add_argument("--lims", type=float, nargs=2, default=None,
                    help="Optional axis limits: min max (e.g., 0 80). If omitted, uses data-driven limits.")
    ap.add_argument("--dpi", type=int, default=200)
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

    y_true = pta_mean(df[true_cols].to_numpy(dtype=float))
    y_tab  = pta_mean(df[pred_tab_cols].to_numpy(dtype=float))
    y_res  = pta_mean(df[pred_resid_cols].to_numpy(dtype=float))
    y_gat  = pta_mean(df[pred_gated_cols].to_numpy(dtype=float))

    models = [("Tabular", y_tab), ("Residual", y_res), ("Gated", y_gat)]

    # Axis limits
    if args.lims is None:
        lims = axis_limits_with_padding(y_true, y_tab, y_res, y_gat)
    else:
        lims = (float(args.lims[0]), float(args.lims[1]))

    out_dir = resolve_out_dir(in_path, args.out_dir)
    suf = args.suffix
    stem = f"FigS2_yyplot_primary_ac_ptamean{suf}"

    plt.figure(figsize=(12, 4))

    for i, (name, y_pred) in enumerate(models, start=1):
        ax = plt.subplot(1, 3, i)
        mask = (~np.isnan(y_true)) & (~np.isnan(y_pred))
        ax.scatter(y_true[mask], y_pred[mask], s=14, color=MODEL_COLOR[name])
        ax.plot([lims[0], lims[1]], [lims[0], lims[1]], linestyle="--", color="0.3")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"({chr(64+i)}) {name}")
        ax.set_xlabel("Observed PTA mean (AC, dB)")
        if i == 1:
            ax.set_ylabel("Predicted PTA mean (AC, dB)")
        ax.grid(False)

        mae = float(np.mean(np.abs(y_pred[mask] - y_true[mask])))
        ax.text(0.02, 0.98, f"MAE={mae:.2f} dB\nn={mask.sum()}",
                transform=ax.transAxes, va="top", ha="left")

    plt.suptitle("Supplementary Figure S2. Observed vs predicted post-op AC PTA mean (OOF)", y=1.02, fontsize=12)
    plt.tight_layout()
    png_path, pdf_path = save_png_pdf(plt.gcf(), out_dir, stem, dpi=args.dpi)
    plt.close()

    print("[OK] Saved:")
    print("  ", png_path)
    print("  ", pdf_path)

if __name__ == "__main__":
    main()
