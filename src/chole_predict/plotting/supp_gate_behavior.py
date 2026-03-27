#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Make Supplementary Figure S6: Gate behavior (same layout/style as original Fig4).

Panels (left->right):
(A) Gate distribution (mean over 4 AC frequencies)
(B) Benefit stratified by gate quartiles: Δ|error| = |e_TAB| - |e_GATED| for AC PTA mean
(C) Worsened rate by gate quartiles: P(|e_GATED| > |e_TAB|)

Inputs
- per_patient_results.csv : must contain id and absolute error columns for AC PTA mean:
    ae_tab_post_PTA_mean_A, ae_gated_post_PTA_mean_A
  (If those don't exist, script tries to compute from pred/true columns.)
- GATED_oof_predictions.csv : must contain gate columns for AC freqs:
    gate_post_PTA_0.5k_A, gate_post_PTA_1k_A, gate_post_PTA_2k_A, gate_post_PTA_3k_A
  (If your gate columns are named slightly differently, see --gate_cols.)

Outputs
- FigS6_gate_behavior_like_Fig4_color.pdf / _gray.pdf in --out_dir
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .common import require_cols, safe_float_series, set_frame_no_grid

# ------------------------
# helpers
# ------------------------
def _ensure_abs_error(df, model_prefix, true_col, pred_col, out_col):
    if out_col in df.columns:
        return
    if true_col in df.columns and pred_col in df.columns:
        df[out_col] = (safe_float_series(df[pred_col]) - safe_float_series(df[true_col])).abs()
        return
    raise KeyError(f"Cannot compute {out_col}: need {true_col} and {pred_col} (or provide {out_col}).")

def _quantile_groups(x, q=4):
    # returns integer group 1..q with NaN preserved
    x = pd.Series(x)
    out = pd.Series(np.nan, index=x.index, dtype="float")
    ok = x.notna()
    if ok.sum() < q:
        return out
    try:
        bins = pd.qcut(x[ok], q=q, labels=False, duplicates="drop")
        out.loc[ok] = bins.astype(float) + 1.0
    except Exception:
        # fallback: rank-based cut
        r = x[ok].rank(method="average")
        bins = pd.qcut(r, q=q, labels=False, duplicates="drop")
        out.loc[ok] = bins.astype(float) + 1.0
    return out

def _save_pdf(fig, path, dpi=300):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)

# ------------------------
def make_figure(df, out_pdf, color=True):
    # palette (npjDM-ish, muted). Use same mapping across panels.
    if color:
        palette = {
            "Q1": "#4C78A8",  # blue
            "Q2": "#F58518",  # orange
            "Q3": "#54A24B",  # green
            "Q4": "#E45756",  # red
        }
    else:
        palette = {"Q1": "0.2", "Q2": "0.4", "Q3": "0.6", "Q4": "0.8"}

    gate = df["gate_mean_A"]
    qgrp = _quantile_groups(gate, q=4)  # 1..4
    df = df.copy()
    df["gate_q"] = qgrp

    # Δ|error| and worsened
    df["delta_abs_err"] = df["ae_tab_post_PTA_mean_A"] - df["ae_gated_post_PTA_mean_A"]
    df["worsened"] = (df["ae_gated_post_PTA_mean_A"] > df["ae_tab_post_PTA_mean_A"]).astype(float)

    # make 1x3 layout
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axA, axB, axC = axes

    # (A) Gate distribution
    gm = gate.dropna().values
    axA.boxplot([gm], tick_labels=["Gate\n(mean)"], vert=True, widths=0.5, patch_artist=True,
                boxprops=dict(facecolor=("0.85" if not color else "#A7C6ED"), edgecolor="black"),
                medianprops=dict(color="black"),
                whiskerprops=dict(color="black"),
                capprops=dict(color="black"))
    axA.set_ylabel("Gate value")
    axA.set_title("(A) Gate distribution")
    set_frame_no_grid(axA)

    # (B) Benefit by quartiles
    data = []
    labels = []
    colors = []
    for k in [1,2,3,4]:
        vals = df.loc[df["gate_q"]==k, "delta_abs_err"].dropna().values
        data.append(vals)
        labels.append(f"Q{k}")
        colors.append(palette[f"Q{k}"])
    bp = axB.boxplot(data, tick_labels=labels, widths=0.55, patch_artist=True,
                     showfliers=False,
                     medianprops=dict(color="black"),
                     whiskerprops=dict(color="black"),
                     capprops=dict(color="black"))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_edgecolor("black")
        patch.set_alpha(0.85 if color else 1.0)
    axB.axhline(0, color="black", lw=1, ls="--", alpha=0.7)
    axB.set_ylabel(r"$\Delta |error|$ (dB) = $|e_{TAB}| - |e_{GATED}|$")
    axB.set_title("(B) Benefit stratified by gate quartiles")
    set_frame_no_grid(axB)

    # (C) Worsened rate by quartiles
    rates = []
    cis = []
    ns = []
    rng = np.random.default_rng(0)
    B = 2000
    for k in [1,2,3,4]:
        v = df.loc[df["gate_q"]==k, "worsened"].dropna().values
        ns.append(len(v))
        if len(v)==0:
            rates.append(np.nan); cis.append((np.nan,np.nan)); continue
        p = float(np.mean(v))
        rates.append(p)
        # bootstrap CI
        boots = []
        for _ in range(B):
            samp = rng.choice(v, size=len(v), replace=True)
            boots.append(float(np.mean(samp)))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        cis.append((lo, hi))

    x = np.arange(4)
    y = np.array(rates, dtype=float)
    yerr = np.array([[y[i]-cis[i][0] if np.isfinite(y[i]) else 0 for i in range(4)],
                     [cis[i][1]-y[i] if np.isfinite(y[i]) else 0 for i in range(4)]], dtype=float)

    bars = axC.bar(x, y, yerr=yerr, capsize=4, edgecolor="black", linewidth=1)
    for b, k in zip(bars, [1,2,3,4]):
        b.set_facecolor(palette[f"Q{k}"])
        b.set_alpha(0.85 if color else 1.0)
    axC.set_xticks(x, [f"Q{k}" for k in [1,2,3,4]])
    axC.set_ylim(0, 1)
    axC.set_ylabel("Worsened rate\n(P(|e_GATED| > |e_TAB|))")
    axC.set_title("(C) Worsened by gate quartiles (95% CI)")
    set_frame_no_grid(axC)

    # mild spacing without tight_layout pitfalls
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.15, top=0.90, wspace=0.35)
    _save_pdf(fig, out_pdf)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_patient_results", required=True, help="per_patient_results.csv")
    ap.add_argument("--gated_oof", required=True, help="GATED_oof_predictions.csv (contains gate_*)")
    ap.add_argument("--out_dir", required=True, help="output directory")
    ap.add_argument("--id_col", default="id")
    ap.add_argument("--gate_cols", default=None,
                    help="Comma-separated gate columns for AC freqs. Default: gate_post_PTA_0.5k_A,gate_post_PTA_1k_A,gate_post_PTA_2k_A,gate_post_PTA_3k_A")
    args = ap.parse_args()

    df = pd.read_csv(args.per_patient_results)
    g = pd.read_csv(args.gated_oof)

    idc = args.id_col
    if idc not in df.columns or idc not in g.columns:
        raise KeyError(f"id_col '{idc}' must exist in both files")

    # gate cols
    if args.gate_cols:
        gate_cols = [c.strip() for c in args.gate_cols.split(",") if c.strip()]
    else:
        gate_cols = ["gate_post_PTA_0.5k_A","gate_post_PTA_1k_A","gate_post_PTA_2k_A","gate_post_PTA_3k_A"]
    require_cols(g, gate_cols, "gated_oof")

    gate_mean = g[[idc] + gate_cols].copy()
    for c in gate_cols:
        gate_mean[c] = safe_float_series(gate_mean[c])
    gate_mean["gate_mean_A"] = gate_mean[gate_cols].mean(axis=1, skipna=True)
    gate_mean = gate_mean[[idc, "gate_mean_A"]].drop_duplicates(subset=[idc], keep="last")

    # merge
    df = df.merge(gate_mean, on=idc, how="left")

    # ensure abs error columns exist
    _ensure_abs_error(df, "tab", "true_post_PTA_mean_A", "pred_tab_post_PTA_mean_A", "ae_tab_post_PTA_mean_A")
    _ensure_abs_error(df, "gated", "true_post_PTA_mean_A", "pred_gated_post_PTA_mean_A", "ae_gated_post_PTA_mean_A")

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    make_figure(df, os.path.join(out_dir, "Supp_Fig_gate_behavior_like_Fig4_color.pdf"), color=True)
    make_figure(df, os.path.join(out_dir, "Supp_Fig_gate_behavior_like_Fig4_gray.pdf"), color=False)

    print("[OK] Wrote:",
          os.path.join(out_dir, "Supp_Fig_gate_behavior_like_Fig4_color.pdf"),
          "and",
          os.path.join(out_dir, "Supp_Fig_gate_behavior_like_Fig4_gray.pdf"))

if __name__ == "__main__":
    main()
