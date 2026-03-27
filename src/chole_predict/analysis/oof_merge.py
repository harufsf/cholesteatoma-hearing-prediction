from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


FREQS = ["0.5k", "1k", "2k", "3k"]
EARS = ["A", "B"]


def _target_cols_from_freqs() -> list[str]:
    return [f"post_PTA_{f}_{ear}" for ear in EARS for f in FREQS]


def _pta_mean_cols(prefix: str, ear: str) -> list[str]:
    return [f"{prefix}_post_PTA_{f}_{ear}" for f in FREQS]


def _safe_abs(x):
    return np.abs(pd.to_numeric(x, errors="coerce"))


def _safe_mean(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    arr = df[cols].apply(pd.to_numeric, errors="coerce")
    return arr.mean(axis=1, skipna=True)


def _rename_tab_oof(tab_df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    keep = [id_col]
    rename = {}

    for t in _target_cols_from_freqs():
        t_true = f"true_{t}"
        t_pred = f"pred_{t}"
        t_err = f"err_{t}"

        if t_true in tab_df.columns:
            keep.append(t_true)
            rename[t_true] = f"true_{t}"
        if t_pred in tab_df.columns:
            keep.append(t_pred)
            rename[t_pred] = f"pred_tab_{t}"
        if t_err in tab_df.columns:
            keep.append(t_err)
            rename[t_err] = f"err_tab_{t}"

    if "fold" in tab_df.columns:
        keep.append("fold")

    out = tab_df[keep].copy()
    return out.rename(columns=rename)


def _rename_resid_oof(resid_df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    keep = [id_col]
    rename = {}

    for t in _target_cols_from_freqs():
        t_pred = f"pred_{t}"
        t_err = f"err_{t}"

        if t_pred in resid_df.columns:
            keep.append(t_pred)
            rename[t_pred] = f"pred_resid_{t}"
        if t_err in resid_df.columns:
            keep.append(t_err)
            rename[t_err] = f"err_resid_{t}"

    out = resid_df[keep].copy()
    return out.rename(columns=rename)


def _rename_gated_oof(gated_df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    keep = [id_col]
    rename = {}

    for t in _target_cols_from_freqs():
        t_pred = f"pred_{t}"
        t_ytab = f"y_tab_{t}"
        t_delta = f"delta_{t}"
        t_gate = f"gate_{t}"
        t_err = f"err_{t}"
        t_abs = f"abs_err_{t}"

        if t_pred in gated_df.columns:
            keep.append(t_pred)
            rename[t_pred] = f"pred_gated_{t}"
        if t_ytab in gated_df.columns:
            keep.append(t_ytab)
            rename[t_ytab] = f"y_tab_gated_{t}"
        if t_delta in gated_df.columns:
            keep.append(t_delta)
            rename[t_delta] = f"delta_gated_{t}"
        if t_gate in gated_df.columns:
            keep.append(t_gate)
            rename[t_gate] = f"gate_{t}"
        if t_err in gated_df.columns:
            keep.append(t_err)
            rename[t_err] = f"err_gated_{t}"
        if t_abs in gated_df.columns:
            keep.append(t_abs)
            rename[t_abs] = f"ae_gated_{t}"

    out = gated_df[keep].copy()
    return out.rename(columns=rename)


def _add_absolute_error_cols(df: pd.DataFrame) -> pd.DataFrame:
    for model in ["tab", "resid", "gated"]:
        for t in _target_cols_from_freqs():
            pred_col = f"pred_{model}_{t}"
            true_col = f"true_{t}"
            ae_col = f"ae_{model}_{t}"
            err_col = f"err_{model}_{t}"

            if pred_col in df.columns and true_col in df.columns and ae_col not in df.columns:
                df[ae_col] = _safe_abs(pd.to_numeric(df[pred_col], errors="coerce") - pd.to_numeric(df[true_col], errors="coerce"))

            if pred_col in df.columns and true_col in df.columns and err_col not in df.columns:
                df[err_col] = pd.to_numeric(df[pred_col], errors="coerce") - pd.to_numeric(df[true_col], errors="coerce")

    return df


def _add_pta_mean_cols(df: pd.DataFrame) -> pd.DataFrame:
    for ear in EARS:
        true_cols = _pta_mean_cols("true", ear)
        if all(c in df.columns for c in true_cols):
            df[f"true_post_PTA_mean_{ear}"] = _safe_mean(df, true_cols)

        for model in ["tab", "resid", "gated"]:
            pred_cols = _pta_mean_cols(f"pred_{model}", ear)
            if all(c in df.columns for c in pred_cols):
                df[f"pred_{model}_post_PTA_mean_{ear}"] = _safe_mean(df, pred_cols)

            ae_cols = _pta_mean_cols(f"ae_{model}", ear)
            if all(c in df.columns for c in ae_cols):
                df[f"ae_{model}_post_PTA_mean_{ear}"] = _safe_mean(df, ae_cols)

            err_cols = _pta_mean_cols(f"err_{model}", ear)
            if all(c in df.columns for c in err_cols):
                df[f"err_{model}_post_PTA_mean_{ear}"] = _safe_mean(df, err_cols)

    return df


def build_per_patient_results(
    source_csv: str,
    tab_oof_csv: str,
    resid_oof_csv: str,
    gated_oof_csv: str,
    out_csv: str,
    id_col: str = "id",
    keep_source_cols: Iterable[str] | None = None,
) -> pd.DataFrame:
    src = pd.read_csv(source_csv).copy()
    tab = pd.read_csv(tab_oof_csv).copy()
    resid = pd.read_csv(resid_oof_csv).copy()
    gated = pd.read_csv(gated_oof_csv).copy()

    # source columns to preserve
    default_keep = [id_col, "fold", "side", "sex", "disease", "primary_or_recur", "age"]
    if keep_source_cols is None:
        keep_source_cols = default_keep

    keep_source_cols = [c for c in keep_source_cols if c in src.columns]
    base = src[keep_source_cols].drop_duplicates(subset=[id_col]).copy()

    tab2 = _rename_tab_oof(tab, id_col=id_col)
    resid2 = _rename_resid_oof(resid, id_col=id_col)
    gated2 = _rename_gated_oof(gated, id_col=id_col)

    # fold from TAB is authoritative if present
    if "fold" in tab2.columns and "fold" not in base.columns:
        base = base.merge(tab2[[id_col, "fold"]], on=id_col, how="left")

    out = base.merge(tab2.drop(columns=["fold"], errors="ignore"), on=id_col, how="left")
    out = out.merge(resid2, on=id_col, how="left")
    out = out.merge(gated2, on=id_col, how="left")

    out = _add_absolute_error_cols(out)
    out = _add_pta_mean_cols(out)

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out