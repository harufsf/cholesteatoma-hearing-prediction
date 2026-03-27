# -*- coding: utf-8 -*-
"""
Fig3: Key secondary endpoint ABG<=thr (derived from predictions)
(A) ROC curve
(B) Calibration curve (no background heatmap)
(C) Threshold sweep (Accuracy + optional extra curves)

Updates in v7:
- Legend moved to RIGHT side (outside), keeping each panel square.
- Generates both color and grayscale versions.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .common import require_cols, resolve_out_dir, save_png_pdf, safe_float_series, set_frame_no_grid

# -------------------------
# Utilities
# -------------------------
def _as_numeric(s):
    return safe_float_series(s)

def _get_freq_cols(prefix: str, suffix: str):
    # expected: {prefix}_0.5k{suffix}, {prefix}_1k{suffix}, {prefix}_2k{suffix}, {prefix}_3k{suffix}
    return [f"{prefix}_0.5k{suffix}", f"{prefix}_1k{suffix}", f"{prefix}_2k{suffix}", f"{prefix}_3k{suffix}"]

def _mean_from_freqs(df: pd.DataFrame, prefix: str, suffix: str):
    """Mean over 0.5/1/2/3kHz columns.

    Accepts either {prefix}_{freq}{suffix} (e.g. post_PTA_0.5k_A)
    or true_{prefix}_{freq}{suffix} (e.g. true_post_PTA_0.5k_A).
    """
    cols = _get_freq_cols(prefix, suffix)
    if not all(c in df.columns for c in cols):
        cols_true = _get_freq_cols(f"true_{prefix}", suffix)
        if all(c in df.columns for c in cols_true):
            cols = cols_true
        else:
            miss = [c for c in cols if c not in df.columns]
            miss2 = [c for c in cols_true if c not in df.columns]
            raise KeyError(f"Missing columns in per_patient_results: {miss} (also tried true_ prefix; missing={miss2})")
    arr = np.vstack([_as_numeric(df[c]).to_numpy() for c in cols]).T
    return np.nanmean(arr, axis=1)

def _infer_pred_prefixes(df: pd.DataFrame):
    # Try common naming patterns in per_patient_results
    # We only need mean AC/BC predictions per model (TAB/RESID/GATED) to derive ABG.
    # Patterns handled:
    #  - pred_tab_post_PTA_0.5k_A  (etc)
    #  - tab_pred_post_PTA_0.5k_A
    #  - pred_post_PTA_0.5k_A_tab
    #  - pred_post_PTA_0.5k_A (single-model; then won't work for 3 models)
    targets = [
        "post_PTA_0.5k_A","post_PTA_1k_A","post_PTA_2k_A","post_PTA_3k_A",
        "post_PTA_0.5k_B","post_PTA_1k_B","post_PTA_2k_B","post_PTA_3k_B"
    ]
    models = ["TAB","RESID","GATED"]

    # candidate formatters return list of 8 cols
    cand = []
    # pred_{model_lower}_{target}
    cand.append(lambda m: [f"pred_{m.lower()}_{t}" for t in targets])
    # {model_lower}_pred_{target}
    cand.append(lambda m: [f"{m.lower()}_pred_{t}" for t in targets])
    # pred_{target}_{model_lower}
    cand.append(lambda m: [f"pred_{t}_{m.lower()}" for t in targets])
    # {model}_{target} (rare)
    cand.append(lambda m: [f"{m}_{t}" for t in targets])
    # {model_lower}_{target}
    cand.append(lambda m: [f"{m.lower()}_{t}" for t in targets])

    found = {}
    for m in models:
        for fn in cand:
            cols = fn(m)
            if all(c in df.columns for c in cols):
                found[m] = cols
                break

    if len(found) != 3:
        # Provide a helpful error
        msg = ["Could not infer prediction columns for all models (TAB/RESID/GATED).",
               "Please ensure per_patient_results has per-frequency predictions with clear prefixes.",
               "Tried patterns like: pred_tab_post_PTA_0.5k_A, tab_pred_post_PTA_0.5k_A, pred_post_PTA_0.5k_A_tab, etc.",
               "Found:"]
        msg.append(str({k: v[:2] + ["..."] for k, v in found.items()}))
        raise KeyError("\n".join(msg))
    return found

def _roc_curve(y_true, y_score):
    # y_true in {0,1}
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    # remove NaN
    ok = np.isfinite(y_true) & np.isfinite(y_score)
    y_true = y_true[ok]
    y_score = y_score[ok]
    if y_true.size == 0:
        return np.array([0,1]), np.array([0,1]), np.array([np.inf, -np.inf])
    order = np.argsort(-y_score)
    y_true = y_true[order]
    y_score = y_score[order]
    P = (y_true == 1).sum()
    N = (y_true == 0).sum()
    if P == 0 or N == 0:
        # degenerate
        return np.array([0,1]), np.array([0,1]), np.array([np.inf, -np.inf])
    tps = np.cumsum(y_true == 1)
    fps = np.cumsum(y_true == 0)
    tpr = np.r_[0, tps / P, 1]
    fpr = np.r_[0, fps / N, 1]
    thr = np.r_[np.inf, y_score, -np.inf]
    return fpr, tpr, thr

def _auc_trapz(fpr, tpr):
    # assumes fpr increasing
    return float(np.trapz(tpr, fpr))

def _calibration_bins(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    ok = np.isfinite(y_true) & np.isfinite(y_prob)
    y_true = y_true[ok]
    y_prob = y_prob[ok]
    if y_true.size == 0:
        return None
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(y_prob, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            out.append((np.nan, np.nan, 0))
        else:
            out.append((np.nanmean(y_prob[m]), np.mean(y_true[m]), int(m.sum())))
    return np.array(out, dtype=float)  # col0=mean_pred, col1=obs_freq, col2=count

def _threshold_sweep(y_true, y_prob, thresholds):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    ok = np.isfinite(y_true) & np.isfinite(y_prob)
    y_true = y_true[ok]
    y_prob = y_prob[ok]
    res = []
    for t in thresholds:
        y_hat = (y_prob >= t).astype(int)
        tp = int(((y_hat == 1) & (y_true == 1)).sum())
        tn = int(((y_hat == 0) & (y_true == 0)).sum())
        fp = int(((y_hat == 1) & (y_true == 0)).sum())
        fn = int(((y_hat == 0) & (y_true == 1)).sum())
        acc = (tp + tn) / max(1, (tp + tn + fp + fn))
        sens = tp / max(1, (tp + fn))
        spec = tn / max(1, (tn + fp))
        ppv = tp / max(1, (tp + fp))
        npv = tn / max(1, (tn + fn))
        res.append((t, acc, sens, spec, ppv, npv))
    return np.array(res, dtype=float)

def _set_frame_no_grid(ax):
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)

# -------------------------
# Core
# -------------------------
def make_fig3(df: pd.DataFrame, out_base: str, abg_thr: float, chosen_t: float, color: bool):
    # Truth AC/BC means from post_PTA freqs
    true_ac = _mean_from_freqs(df, "post_PTA", "_A")
    true_bc = _mean_from_freqs(df, "post_PTA", "_B")
    true_abg = true_ac - true_bc
    y_true = (true_abg <= abg_thr).astype(int)

    # Predictions: derive predicted ABG = pred_AC - pred_BC; then map to probability via logistic on score
    pred_cols = _infer_pred_prefixes(df)

    # We need a mapping from predicted ABG to probability for ABG<=thr.
    # Use a monotonic transform: p = sigmoid((thr - pred_abg)/s), with s estimated from distribution.
    # This yields higher probability when predicted ABG is smaller (better).
    def abg_to_prob(pred_abg):
        pred_abg = np.asarray(pred_abg, dtype=float)
        # robust scale: IQR / 1.349 ~ sigma
        q25, q75 = np.nanpercentile(pred_abg, [25, 75])
        iqr = max(1e-6, (q75 - q25))
        s = max(1.0, iqr / 1.349)  # avoid too sharp
        z = (abg_thr - pred_abg) / s
        return 1.0 / (1.0 + np.exp(-z))

    model_order = ["TAB", "RESID", "GATED"]

    if color:
        colors = {"TAB": "#4C72B0", "RESID": "#DD8452", "GATED": "#55A868"}  # muted, npjDM-ish
    else:
        colors = {"TAB": "0.25", "RESID": "0.55", "GATED": "0.75"}

    # compute probabilities for each model
    probs = {}
    aucs = {}
    for m in model_order:
        cols = pred_cols[m]
        # build AC/BC mean predictions
        ac = np.nanmean(np.vstack([_as_numeric(df[c]).to_numpy() for c in cols[0:4]]).T, axis=1)
        bc = np.nanmean(np.vstack([_as_numeric(df[c]).to_numpy() for c in cols[4:8]]).T, axis=1)
        pred_abg = ac - bc
        p = abg_to_prob(pred_abg)
        probs[m] = p
        fpr, tpr, _ = _roc_curve(y_true, p)
        aucs[m] = _auc_trapz(fpr, tpr)

    # ---- Figure layout: 1x3, each square; legend on RIGHT outside.
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.6), dpi=200)
    axA, axB, axC = axes

    # Make each axes square
    for ax in axes:
        try:
            ax.set_box_aspect(1)
        except Exception:
            pass

    # ---- (A) ROC
    for m in model_order:
        fpr, tpr, _ = _roc_curve(y_true, probs[m])
        axA.plot(fpr, tpr, lw=2.5, color=colors[m], label=f"{m} (AUC={aucs[m]:.3f})")
    axA.plot([0,1],[0,1], ls="--", lw=1.5, color="0.6")
    axA.set_title("(A) ROC")
    axA.set_xlabel("False positive rate")
    axA.set_ylabel("True positive rate")
    axA.set_xlim(0,1); axA.set_ylim(0,1)
    _set_frame_no_grid(axA)

    # ---- (B) Calibration (no background heatmap)
    n_bins = 10
    for m in model_order:
        cal = _calibration_bins(y_true, probs[m], n_bins=n_bins)
        if cal is None:
            continue
        mp = cal[:,0]; of = cal[:,1]; cnt = cal[:,2]
        axB.plot(mp, of, marker="o", lw=2.0, color=colors[m])
    axB.plot([0,1],[0,1], ls="--", lw=1.5, color="0.6")
    axB.set_title("(B) Calibration")
    axB.set_xlabel("Predicted probability")
    axB.set_ylabel("Observed frequency")
    axB.set_xlim(0,1); axB.set_ylim(0,1)
    _set_frame_no_grid(axB)

    # ---- (C) Threshold sweep (Accuracy)
    thresholds = np.linspace(0.05, 0.95, 19)
    for m in model_order:
        sweep = _threshold_sweep(y_true, probs[m], thresholds)
        axC.plot(sweep[:,0], sweep[:,1], lw=2.5, color=colors[m])
    axC.axvline(chosen_t, ls="--", lw=1.8, color="0.4")
    axC.set_title("(C) Threshold sweep")
    axC.set_xlabel("Probability threshold")
    axC.set_ylabel("Accuracy")
    axC.set_xlim(0,1); axC.set_ylim(0,1)
    _set_frame_no_grid(axC)

    # ---- Legend: RIGHT outside
    handles = [Line2D([0],[0], color=colors[m], lw=3, label=m) for m in model_order]
    # chosen threshold marker legend inside panel (C)
    chosen_handle = Line2D([0],[0], color="0.4", lw=2, ls="--", label=f"chosen t={chosen_t:.2f}")
    axC.legend(handles=[chosen_handle], loc="lower right", frameon=False, fontsize=9)
    # Reserve right margin for legend
    fig.subplots_adjust(left=0.07, right=0.80, wspace=0.35, top=0.88, bottom=0.16)
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.82, 0.5),
               frameon=False, borderaxespad=0.0)

    # overall title
    fig.suptitle(f"Key secondary: ABG≤{abg_thr:.0f} dB (derived: pred_ABG = pred_AC − pred_BC)", y=0.98, fontsize=12)

    # Save
    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    suffix = "color" if color else "gray"
    pdf_path = f"{out_base}_{suffix}.pdf"
    png_path = f"{out_base}_{suffix}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_patient_results", required=True, help="per_patient_results.csv (must include per-frequency post_PTA_*_A/B true and TAB/RESID/GATED predictions)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--abg_thr", type=float, default=20.0)
    ap.add_argument("--chosen_t", type=float, default=0.36)
    args = ap.parse_args()

    df = pd.read_csv(args.per_patient_results)
    out_base = os.path.join(args.out_dir, "Fig3_keysecondary_abg_leq20")

    make_fig3(df, out_base, args.abg_thr, args.chosen_t, color=True)
    make_fig3(df, out_base, args.abg_thr, args.chosen_t, color=False)

    print("[OK] Wrote:")
    print(" -", out_base + "_color.pdf")
    print(" -", out_base + "_gray.pdf")

if __name__ == "__main__":
    main()
