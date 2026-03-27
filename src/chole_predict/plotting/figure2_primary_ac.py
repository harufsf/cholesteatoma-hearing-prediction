#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure 2 (Primary endpoint) 

  - Panel A: Violin + box of per-patient absolute error for PRIMARY outcome
             |error| = |PTAmean_pred - PTAmean_true|  (AC PTA mean)
             (harm-consistent; uses ae_*_post_PTA_mean_A if available)
  - Panel B: Frequency-specific MAE with 95% bootstrap CI (AC, 0.5/1/2/3 kHz)

Inputs:
  --per_patient_results  per_patient_results.csv

Outputs (to --out_dir):
  - Fig2_primary_ac_patternA_color.png/pdf
  - Fig2_primary_ac_patternA_gray.png/pdf

Notes:
- Colors: Tabular=blue, Residual=orange, Gated=green.
- No grid; keep spines.
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODELS = ["tab", "resid", "gated"]
MODEL_LABEL = {"tab": "Tabular", "resid": "Residual", "gated": "Gated"}
MODEL_COLOR = {"tab": "tab:blue", "resid": "tab:orange", "gated": "tab:green"}
FREQS = ["0.5k", "1k", "2k", "3k"]


def _set_frame_no_grid(ax):
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(1.0)


def _col(df: pd.DataFrame, *cands: str):
    for c in cands:
        if c in df.columns:
            return c
    return None


def _bootstrap_mean_ci(x: np.ndarray, B: int, seed: int = 0):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return np.nan, (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(x.mean()), (float(lo), float(hi))


def _pta_mean_from_freq(df: pd.DataFrame, prefix: str):
    cols = [f"{prefix}_post_PTA_{f}_A" for f in FREQS]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for PTA mean computation: {missing}")
    return np.nanmean(df[cols].to_numpy(dtype=float), axis=1)


def make_figure2_patternA(
    per_patient_results: str,
    out_dir: str,
    dpi: int = 300,
    B: int = 10000,
    seed: int = 0,
):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(per_patient_results)

    # Panel A: harm-consistent primary |error|
    col_primary = {
        "tab": _col(df, "ae_tab_post_PTA_mean_A"),
        "resid": _col(df, "ae_resid_post_PTA_mean_A"),
        "gated": _col(df, "ae_gated_post_PTA_mean_A"),
    }
    has_all = all(col_primary[m] is not None for m in MODELS)

    primary_data = []
    for m in MODELS:
        if has_all:
            x = df[col_primary[m]].to_numpy(dtype=float)
        else:
            y_true = _pta_mean_from_freq(df, "true")
            y_pred = _pta_mean_from_freq(df, f"pred_{m}")
            x = np.abs(y_pred - y_true)
        x = x[np.isfinite(x)]
        primary_data.append(x)

    # Panel B: freq MAE ± CI
    mae_freq = {f: {} for f in FREQS}
    ci_freq = {f: {} for f in FREQS}

    for m in MODELS:
        for f in FREQS:
            col_f = _col(df, f"ae_{m}_post_PTA_{f}_A")
            if col_f is None:
                tcol = f"true_post_PTA_{f}_A"
                pcol = f"pred_{m}_post_PTA_{f}_A"
                if tcol not in df.columns or pcol not in df.columns:
                    raise ValueError(f"Missing freq columns for {m} {f}: {tcol}, {pcol}")
                x = np.abs(df[pcol].to_numpy(dtype=float) - df[tcol].to_numpy(dtype=float))
            else:
                x = df[col_f].to_numpy(dtype=float)
            mu, (lo, hi) = _bootstrap_mean_ci(x, B=B, seed=seed)
            mae_freq[f][m] = mu
            ci_freq[f][m] = (lo, hi)

    def _plot(color: bool, prefix: str):
        fig = plt.figure(figsize=(9.2, 3.6))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.9], wspace=0.35)

        # Panel A: violin + box
        ax0 = fig.add_subplot(gs[0, 0])
        _set_frame_no_grid(ax0)

        pos = np.arange(1, len(MODELS) + 1)
        vio = ax0.violinplot(
            primary_data,
            positions=pos,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for i, body in enumerate(vio["bodies"]):
            body.set_facecolor(MODEL_COLOR[MODELS[i]] if color else "0.75")
            body.set_edgecolor("0.0")
            body.set_alpha(0.7)

        bp = ax0.boxplot(primary_data, positions=pos, widths=0.25, showfliers=True, patch_artist=True)
        for i, box in enumerate(bp["boxes"]):
            box.set_facecolor(MODEL_COLOR[MODELS[i]] if color else "0.85")
            box.set_alpha(0.9)
            box.set_edgecolor("0.0")
        for k in ("whiskers", "caps", "medians"):
            for item in bp[k]:
                item.set_color("0.0")

        ax0.set_xticks(pos, [MODEL_LABEL[m] for m in MODELS])
        ax0.set_ylabel("Absolute error |ŷ − y| (dB)")
        ax0.set_title("(A) PTA mean (AC): |error| distribution")

        # Panel B: frequency-specific MAE + 95% CI
        ax1 = fig.add_subplot(gs[0, 1])
        _set_frame_no_grid(ax1)

        x = np.arange(len(FREQS))
        offsets = {"tab": -0.22, "resid": 0.0, "gated": 0.22}
        for m in MODELS:
            y = [mae_freq[f][m] for f in FREQS]
            yerr = np.array(
                [
                    [y[i] - ci_freq[FREQS[i]][m][0] for i in range(len(FREQS))],
                    [ci_freq[FREQS[i]][m][1] - y[i] for i in range(len(FREQS))],
                ]
            )
            ax1.errorbar(
                x + offsets[m],
                y,
                yerr=yerr,
                fmt="o",
                capsize=3,
                label=MODEL_LABEL[m],
                color=(MODEL_COLOR[m] if color else "0.25"),
            )

        ax1.set_xticks(x, ["0.5", "1", "2", "3"])
        ax1.set_xlabel("Frequency (kHz)")
        ax1.set_ylabel("MAE (dB)")
        ax1.set_title("(B) Frequency-specific MAE (AC) with 95% CI")
        ax1.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

        fig.suptitle("Figure 2. Primary endpoint: post-op AC thresholds (OOF)", y=1.03, fontsize=11)
        fig.tight_layout()

        pdf = os.path.join(out_dir, f"{prefix}.pdf")
        png = os.path.join(out_dir, f"{prefix}.png")
        fig.savefig(pdf, bbox_inches="tight")
        fig.savefig(png, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print("[OK]", pdf)
        print("[OK]", png)

    _plot(True, "Fig2_primary_ac_color")
    _plot(False, "Fig2_primary_ac_gray")

    print("[Panel A] mean |error| (harm-consistent):")
    for m, arr in zip(MODELS, primary_data):
        mu, (lo, hi) = _bootstrap_mean_ci(arr, B=B, seed=seed)
        print(f"  {MODEL_LABEL[m]}: {mu:.2f} [{lo:.2f}, {hi:.2f}] (n={len(arr)})")


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_patient_results", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    return ap


def main():
    args = build_argparser().parse_args()
    make_figure2_patternA(
        per_patient_results=args.per_patient_results,
        out_dir=args.out_dir,
        dpi=args.dpi,
        B=args.B,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()