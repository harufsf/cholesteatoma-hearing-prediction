#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


FREQS = ["0.5k", "1k", "2k", "3k"]
EARS = ["A", "B"]


def target_cols() -> list[str]:
    return [f"post_PTA_{f}_{ear}" for ear in EARS for f in FREQS]


def pta_cols(prefix: str, ear: str) -> list[str]:
    return [f"{prefix}_post_PTA_{f}_{ear}" for f in FREQS]


def safe_mean(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    arr = df[cols].apply(pd.to_numeric, errors="coerce")
    return arr.mean(axis=1, skipna=True)


def rename_tab_oof(tab_df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    keep = [id_col]
    rename = {}

    if "fold" in tab_df.columns:
        keep.append("fold")

    for t in target_cols():
        true_col = f"true_{t}"
        pred_col = f"pred_{t}"
        err_col = f"err_{t}"
        abs_col = f"abs_err_{t}"

        if true_col in tab_df.columns:
            keep.append(true_col)
            rename[true_col] = true_col

        if pred_col in tab_df.columns:
            keep.append(pred_col)
            rename[pred_col] = f"pred_tab_{t}"

        if err_col in tab_df.columns:
            keep.append(err_col)
            rename[err_col] = f"err_tab_{t}"

        if abs_col in tab_df.columns:
            keep.append(abs_col)
            rename[abs_col] = f"ae_tab_{t}"

    out = tab_df[keep].copy()
    return out.rename(columns=rename)


def rename_resid_oof(resid_df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    keep = [id_col]
    rename = {}

    for t in target_cols():
        pred_col = f"pred_{t}"
        err_col = f"err_{t}"
        abs_col = f"abs_err_{t}"
        delta_col = f"delta_{t}"
        ytab_col = f"y_tab_{t}"

        if pred_col in resid_df.columns:
            keep.append(pred_col)
            rename[pred_col] = f"pred_resid_{t}"

        if err_col in resid_df.columns:
            keep.append(err_col)
            rename[err_col] = f"err_resid_{t}"

        if abs_col in resid_df.columns:
            keep.append(abs_col)
            rename[abs_col] = f"ae_resid_{t}"

        if delta_col in resid_df.columns:
            keep.append(delta_col)
            rename[delta_col] = f"delta_resid_{t}"

        if ytab_col in resid_df.columns:
            keep.append(ytab_col)
            rename[ytab_col] = f"y_tab_resid_{t}"

    out = resid_df[keep].copy()
    return out.rename(columns=rename)


def rename_gated_oof(gated_df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    keep = [id_col]
    rename = {}

    for t in target_cols():
        pred_col = f"pred_{t}"
        err_col = f"err_{t}"
        abs_col = f"abs_err_{t}"
        delta_col = f"delta_{t}"
        ytab_col = f"y_tab_{t}"
        gate_col = f"gate_{t}"

        if pred_col in gated_df.columns:
            keep.append(pred_col)
            rename[pred_col] = f"pred_gated_{t}"

        if err_col in gated_df.columns:
            keep.append(err_col)
            rename[err_col] = f"err_gated_{t}"

        if abs_col in gated_df.columns:
            keep.append(abs_col)
            rename[abs_col] = f"ae_gated_{t}"

        if delta_col in gated_df.columns:
            keep.append(delta_col)
            rename[delta_col] = f"delta_gated_{t}"

        if ytab_col in gated_df.columns:
            keep.append(ytab_col)
            rename[ytab_col] = f"y_tab_gated_{t}"

        if gate_col in gated_df.columns:
            keep.append(gate_col)
            rename[gate_col] = gate_col

    out = gated_df[keep].copy()
    return out.rename(columns=rename)


def add_error_cols(df: pd.DataFrame) -> pd.DataFrame:
    for model in ["tab", "resid", "gated"]:
        for t in target_cols():
            pred_col = f"pred_{model}_{t}"
            true_col = f"true_{t}"
            err_col = f"err_{model}_{t}"
            ae_col = f"ae_{model}_{t}"

            if pred_col in df.columns and true_col in df.columns:
                pred = pd.to_numeric(df[pred_col], errors="coerce")
                true = pd.to_numeric(df[true_col], errors="coerce")

                if err_col not in df.columns:
                    df[err_col] = pred - true
                if ae_col not in df.columns:
                    df[ae_col] = (pred - true).abs()
    return df


def add_pta_mean_cols(df: pd.DataFrame) -> pd.DataFrame:
    for ear in EARS:
        true_cols = pta_cols("true", ear)
        if all(c in df.columns for c in true_cols):
            df[f"true_post_PTA_mean_{ear}"] = safe_mean(df, true_cols)

        for model in ["tab", "resid", "gated"]:
            pred_cols = pta_cols(f"pred_{model}", ear)
            err_cols = pta_cols(f"err_{model}", ear)
            ae_cols = pta_cols(f"ae_{model}", ear)

            if all(c in df.columns for c in pred_cols):
                df[f"pred_{model}_post_PTA_mean_{ear}"] = safe_mean(df, pred_cols)
            if all(c in df.columns for c in err_cols):
                df[f"err_{model}_post_PTA_mean_{ear}"] = safe_mean(df, err_cols)
            if all(c in df.columns for c in ae_cols):
                df[f"ae_{model}_post_PTA_mean_{ear}"] = safe_mean(df, ae_cols)
    return df


def merge_per_patient_results(
    source_csv: str,
    tab_oof_csv: str,
    resid_oof_csv: str,
    gated_oof_csv: str,
    out_csv: str,
    id_col: str = "id",
    keep_source_cols: Iterable[str] | None = None,
) -> pd.DataFrame:
    src = pd.read_csv(source_csv)
    tab = pd.read_csv(tab_oof_csv)
    resid = pd.read_csv(resid_oof_csv)
    gated = pd.read_csv(gated_oof_csv)

    if keep_source_cols is None:
        keep_source_cols = [
            id_col,
            "fold",
            "side",
            "sex",
            "disease",
            "primary_or_recur",
            "age",
        ]

    keep_source_cols = [c for c in keep_source_cols if c in src.columns]
    base = src[keep_source_cols].drop_duplicates(subset=[id_col]).copy()

    tab2 = rename_tab_oof(tab, id_col=id_col)
    resid2 = rename_resid_oof(resid, id_col=id_col)
    gated2 = rename_gated_oof(gated, id_col=id_col)

    if "fold" in tab2.columns and "fold" not in base.columns:
        base = base.merge(tab2[[id_col, "fold"]], on=id_col, how="left")

    out = base.merge(tab2.drop(columns=["fold"], errors="ignore"), on=id_col, how="left")
    out = out.merge(resid2, on=id_col, how="left")
    out = out.merge(gated2, on=id_col, how="left")

    out = add_error_cols(out)
    out = add_pta_mean_cols(out)

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_csv", required=True, help="Base CSV such as df_fixed_with_cropped_roi.csv")
    ap.add_argument("--tab_oof", required=True, help="TAB_oof_predictions.csv")
    ap.add_argument("--resid_oof", required=True, help="RESID_oof_predictions.csv")
    ap.add_argument("--gated_oof", required=True, help="GATED_oof_predictions.csv")
    ap.add_argument("--out_csv", required=True, help="Output per_patient_results.csv")
    ap.add_argument("--id_col", default="id")
    return ap


def main():
    args = build_argparser().parse_args()

    df = merge_per_patient_results(
        source_csv=args.source_csv,
        tab_oof_csv=args.tab_oof,
        resid_oof_csv=args.resid_oof,
        gated_oof_csv=args.gated_oof,
        out_csv=args.out_csv,
        id_col=args.id_col,
    )

    print(f"[OK] wrote: {args.out_csv}")
    print(f"[INFO] rows={len(df)} cols={len(df.columns)}")

    preview_cols = [c for c in [
        args.id_col,
        "true_post_PTA_mean_A",
        "pred_tab_post_PTA_mean_A",
        "pred_resid_post_PTA_mean_A",
        "pred_gated_post_PTA_mean_A",
        "ae_tab_post_PTA_mean_A",
        "ae_resid_post_PTA_mean_A",
        "ae_gated_post_PTA_mean_A",
    ] if c in df.columns]

    if preview_cols:
        print(df[preview_cols].head())


if __name__ == "__main__":
    main()