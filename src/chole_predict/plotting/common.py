from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

FREQS_AC = ["0.5k", "1k", "2k", "3k"]
MODEL_ORDER = ["tab", "resid", "gated"]
MODEL_LABEL = {"tab": "Tabular", "resid": "Residual", "gated": "Gated"}
MODEL_LABEL_LONG = ["Tabular", "Residual", "Gated"]


def set_frame_no_grid(ax, linewidth: float = 1.0):
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(linewidth)


def require_cols(df: pd.DataFrame, cols, name: str = "df"):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise KeyError(f"Missing columns in {name}: {miss}")


def safe_float_series(s):
    return pd.to_numeric(s, errors="coerce")


def first_existing_col(df: pd.DataFrame, *cands: str) -> str:
    for c in cands:
        if c in df.columns:
            return c
    raise KeyError(f"None of columns exist: {cands}")


def pta_mean(arr_4):
    arr = np.asarray(arr_4, dtype=float)
    return np.nanmean(arr, axis=1)


def build_primary_ac_cols(prefix: str):
    return [f"{prefix}_post_PTA_{f}_A" for f in FREQS_AC]


def build_true_primary_ac_cols():
    return [f"true_post_PTA_{f}_A" for f in FREQS_AC]


def compute_cdf(arr):
    x = np.asarray(arr, dtype=float)
    x = x[np.isfinite(x)]
    x = np.sort(x)
    y = np.arange(1, len(x) + 1) / len(x) if len(x) else np.array([])
    return x, y


def axis_limits_with_padding(*arrays, pad_ratio: float = 0.05, min_span: float = 1.0):
    vals = []
    for a in arrays:
        aa = np.asarray(a, dtype=float)
        aa = aa[np.isfinite(aa)]
        if aa.size:
            vals.append(aa)
    if not vals:
        return (0.0, 1.0)
    allv = np.concatenate(vals)
    vmin = float(np.nanmin(allv))
    vmax = float(np.nanmax(allv))
    span = vmax - vmin
    if span <= 0:
        span = min_span
    pad = max(pad_ratio * span, min_span * 0.05)
    return (vmin - pad, vmax + pad)


def resolve_out_dir(input_path: str | Path, out_dir: str | Path | None = None) -> Path:
    src = Path(input_path)
    dst = Path(out_dir) if out_dir else src.parent
    dst.mkdir(parents=True, exist_ok=True)
    return dst


def save_png_pdf(fig, out_dir: str | Path, stem: str, dpi: int = 300):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    return png, pdf
