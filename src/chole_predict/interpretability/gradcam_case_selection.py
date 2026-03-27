# make_case_list_for_gradcam.py
# ------------------------------------------------------------
# Auto-generate case_list.csv for Grad-CAM case selection
# Sets:
#   Set A: "RESID benefit" in (low pre_mean_air) x (attic) x (primary optional)
#   Set B: "GATE benefit" in (high pre_mean_air)
#   Set C: "Q4 gate outlier" (worst within Q4 by rule-based extraction)
#
# Inputs (required):
#   --per_patient_csv : per_patient_results.csv (must include ae_tab/resid/gated_post_PTA_mean_A)
#   --base_csv        : df_fixed_with_cropped_roi.csv (must include pre_mean_air, disease, primary_or_recur)
#
# Inputs (optional but recommended):
#   --outliers_csv    : outliers_table.csv produced by your Q4-extraction script
#                       (should include id, gate_mean, gate_quartile, delta_abs_err_ptamean, etc.)
#   --gated_oof_csv   : gated OOF predictions file with gate values (if you have it);
#                       script will try to infer a "gate_mean" column and merge by id.
#
# Output:
#   case_list.csv in the same folder as --per_patient_csv (or --out if specified)
#
# Example:
#   python make_case_list_for_gradcam.py ^
#     --per_patient_csv experiments\...\per_patient_results.csv ^
#     --base_csv df_fixed_with_cropped_roi.csv ^
#     --outliers_csv outliers_table.csv ^
#     --setA_k 2 --setB_k 1
#
# Notes:
# - "attic" is defined as disease == "弛緩部型" (edit if needed)
# - "primary" is defined as primary_or_recur == "初発"
# - pre_mean_air tertiles are computed on non-null values in base_csv
# ------------------------------------------------------------

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def infer_gate_column(df: pd.DataFrame) -> str | None:
    """
    Try to infer a gate column name from a gated_oof/outliers table.
    Priority: 'gate_mean' -> any column containing 'gate' and 'mean' -> any 'gate' column.
    """
    cols = list(df.columns)
    if "gate_mean" in cols:
        return "gate_mean"
    cand = [c for c in cols if "gate" in c.lower() and "mean" in c.lower()]
    if cand:
        return cand[0]
    cand = [c for c in cols if "gate" in c.lower()]
    return cand[0] if cand else None


def add_pre_mean_air_tertile(base: pd.DataFrame, col="pre_mean_air") -> pd.DataFrame:
    s = pd.to_numeric(base[col], errors="coerce")
    qs = np.nanquantile(s, [1/3, 2/3])
    base["pre_mean_air_tertile"] = pd.cut(
        s, bins=[-np.inf, qs[0], qs[1], np.inf], labels=["low", "mid", "high"]
    )
    return base


def safe_validate_unique_id(df: pd.DataFrame, name: str):
    if "id" not in df.columns:
        raise ValueError(f"{name} is missing required column: id")
    if df["id"].duplicated().any():
        d = int(df["id"].duplicated().sum())
        raise ValueError(f"{name} has duplicated id rows: duplicated().sum()={d}. Please deduplicate first.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_patient_csv", required=True, help="New per_patient_results.csv")
    ap.add_argument("--base_csv", required=True, help="df_fixed_with_cropped_roi.csv")
    ap.add_argument("--outliers_csv", default=None, help="outliers_table.csv (Q4 extraction output)")
    ap.add_argument("--gated_oof_csv", default=None, help="GATED oof file with gate values (optional)")
    ap.add_argument("--setA_k", type=int, default=2, help="How many cases in Set A")
    ap.add_argument("--setB_k", type=int, default=1, help="How many cases in Set B")
    ap.add_argument("--require_primary_in_setA", action="store_true",
                    help="If set, Set A requires primary_or_recur=='初発' (otherwise preferred but not required)")
    ap.add_argument("--attic_label", default="弛緩部型", help="Disease label treated as 'attic'")
    ap.add_argument("--primary_label", default="初発", help="Primary label")
    ap.add_argument("--out", default=None, help="Output path for case_list.csv (optional)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    per_path = Path(args.per_patient_csv)
    base_path = Path(args.base_csv)

    per = pd.read_csv(per_path)
    base = pd.read_csv(base_path)

    safe_validate_unique_id(per, "per_patient_csv")
    safe_validate_unique_id(base, "base_csv")

    # Required AE columns for Primary (AC PTA mean)
    ae_tab = "ae_tab_post_PTA_mean_A"
    ae_res = "ae_resid_post_PTA_mean_A"
    ae_gat = "ae_gated_post_PTA_mean_A"
    for c in [ae_tab, ae_res, ae_gat]:
        if c not in per.columns:
            raise ValueError(f"per_patient_csv missing required column: {c}")

    # Ensure base has subgroup columns
    for c in ["pre_mean_air", "disease", "primary_or_recur"]:
        if c not in base.columns:
            raise ValueError(f"base_csv missing required column: {c}")

    # Add tertiles
    base = add_pre_mean_air_tertile(base, col="pre_mean_air")

    # Merge
    df = per.merge(base, on="id", how="left", validate="one_to_one")

    # Basic computed fields
    df["delta_tab_resid"] = df[ae_tab] - df[ae_res]         # positive means RESID improved vs TAB
    df["delta_resid_gated"] = df[ae_res] - df[ae_gat]       # positive means GATED improved vs RESID
    df["is_attic"] = df["disease"].astype(str).eq(args.attic_label)
    df["is_primary"] = df["primary_or_recur"].astype(str).eq(args.primary_label)

    # ------------------------------------------------------------
    # Gate info merge (optional)  <-- REPLACE THIS WHOLE BLOCK
    # ------------------------------------------------------------
    if args.gated_oof_csv:
        gated_oof = pd.read_csv(args.gated_oof_csv)
        safe_validate_unique_id(gated_oof, "gated_oof_csv")

        # If gate_mean exists, use it.
        if "gate_mean" in gated_oof.columns:
            gated_oof_small = gated_oof[["id", "gate_mean"]].copy()

        else:
            # Try to compute mean gate for AC (A) across 0.5/1/2/3k
            gate_A_cols = [
                c for c in gated_oof.columns
                if c.startswith("gate_post_PTA_") and c.endswith("_A")
                and any(k in c for k in ["0.5k", "1k", "2k", "3k"])
            ]

            # Fallback: any gate columns
            if len(gate_A_cols) >= 2:
                gated_oof_small = gated_oof[["id"] + gate_A_cols].copy()
                gated_oof_small["gate_mean"] = gated_oof_small[gate_A_cols].mean(axis=1)
                gated_oof_small = gated_oof_small[["id", "gate_mean"]]
            else:
                # last resort: take the first gate* column
                gate_any = [c for c in gated_oof.columns if "gate" in c.lower()]
                if not gate_any:
                    raise KeyError(
                        "No gate columns found in gated_oof_csv. "
                        "Please provide a gate-included OOF file."
                    )
                gated_oof_small = gated_oof[["id", gate_any[0]]].copy()
                gated_oof_small = gated_oof_small.rename(columns={gate_any[0]: "gate_mean"})

        df = df.merge(gated_oof_small, on="id", how="left", validate="one_to_one")

    if args.outliers_csv:
        outliers = pd.read_csv(args.outliers_csv)
        if "id" not in outliers.columns:
            raise ValueError("outliers_csv missing required column: id")

        # Rename outliers gate column to avoid collision with gate_mean from gated_oof
        if "gate_mean" in outliers.columns:
            outliers = outliers.rename(columns={"gate_mean": "gate_mean_outliers"})

        out_gate_cols = [c for c in [
            "gate_mean_outliers", "gate_quartile", "delta_abs_err_ptamean",
            "ae_tab_ac_ptamean", "ae_gated_ac_ptamean"
        ] if c in outliers.columns]

        if out_gate_cols:
            out_small = outliers[["id"] + out_gate_cols].copy()

            # If duplicates exist, keep the worst (most negative delta_abs_err_ptamean) per id when available
            if out_small["id"].duplicated().any():
                if "delta_abs_err_ptamean" in out_small.columns:
                    out_small = out_small.sort_values("delta_abs_err_ptamean").drop_duplicates("id", keep="first")
                else:
                    out_small = out_small.drop_duplicates("id", keep="first")

            df = df.merge(out_small, on="id", how="left", validate="one_to_one")

            # Coalesce: keep gate_mean from gated_oof (all cases), fill missing with outliers (often Q4-only)
            if "gate_mean" in df.columns and "gate_mean_outliers" in df.columns:
                df["gate_mean"] = df["gate_mean"].fillna(df["gate_mean_outliers"])
    # ------------------------------------------------------------
    # Compute gate quartiles over ALL cases (from gate_mean)
    # Adds:
    #   gate_quartile_all: Q1/Q2/Q3/Q4 computed on the entire cohort
    # ------------------------------------------------------------
    if "gate_mean" in df.columns:
        gm = pd.to_numeric(df["gate_mean"], errors="coerce")
        if gm.notna().sum() >= 10:
            try:
                df["gate_quartile_all"] = pd.qcut(gm, q=4, labels=["Q1", "Q2", "Q3", "Q4"])
            except ValueError:
                # Fallback: qcut on ranks (stable even with ties)
                r = gm.rank(method="average")
                df["gate_quartile_all"] = pd.qcut(r, q=4, labels=["Q1", "Q2", "Q3", "Q4"])
        else:
            df["gate_quartile_all"] = np.nan



    # ------------------------------------------------------------
    # Set A selection: low x attic (x primary preferred or required), top by delta_tab_resid
    # ------------------------------------------------------------
    setA_pool = df[df["pre_mean_air_tertile"].astype(str).eq("low") & df["is_attic"]].copy()
    if args.require_primary_in_setA:
        setA_pool = setA_pool[setA_pool["is_primary"]].copy()
    else:
        # prefer primary: keep all, but we'll sort primary-first
        setA_pool["primary_rank"] = (~setA_pool["is_primary"]).astype(int)  # primary -> 0, others -> 1
    if len(setA_pool) == 0:
        raise ValueError("Set A pool is empty. Check attic_label/primary_label or data.")

    if args.require_primary_in_setA:
        setA_pool = setA_pool.sort_values(["delta_tab_resid"], ascending=False)
    else:
        setA_pool = setA_pool.sort_values(["primary_rank", "delta_tab_resid"], ascending=[True, False])

    setA = setA_pool.head(args.setA_k).copy()
    setA["set"] = "A"
    setA["selection_reason"] = (
        "Rule-based: pre_mean_air tertile=low AND disease=attic"
        + (" AND primary" if args.require_primary_in_setA else " (primary preferred)")
        + "; pick top by (ae_tab - ae_resid) on AC PTA mean"
    )

    # ------------------------------------------------------------
    # Set B selection: high, top by delta_resid_gated
    # ------------------------------------------------------------
    setB_pool = df[df["pre_mean_air_tertile"].astype(str).eq("high")].copy()
    if len(setB_pool) == 0:
        raise ValueError("Set B pool is empty. Check pre_mean_air tertile computation.")

    setB_pool = setB_pool.sort_values(["delta_resid_gated"], ascending=False)
    setB = setB_pool.head(args.setB_k).copy()
    setB["set"] = "B"
    setB["selection_reason"] = (
        "Rule-based: pre_mean_air tertile=high; pick top by (ae_resid - ae_gated) on AC PTA mean"
    )

    # ------------------------------------------------------------
    # Set C selection: Q4 gate outlier worst (must come from outliers_csv or gated_oof_csv)
    # Prefer outliers_csv since it encodes Q4 and delta_abs_err_ptamean directly.
    # ------------------------------------------------------------
    setC = None
    if args.outliers_csv:
        outliers = pd.read_csv(args.outliers_csv)
        # Q4 worst: minimum delta_abs_err_ptamean (TAB->GATED harm), or if not present use largest ae_gated
        if "gate_quartile" in outliers.columns:
            q4 = outliers[outliers["gate_quartile"].astype(str).str.upper().eq("Q4")].copy()
        else:
            q4 = outliers.copy()

        if len(q4) == 0:
            raise ValueError("outliers_csv has no Q4 rows (gate_quartile=='Q4').")

        if "delta_abs_err_ptamean" in q4.columns:
            q4 = q4.sort_values("delta_abs_err_ptamean", ascending=True)  # more negative = worse
            c_id = int(q4.iloc[0]["id"])
        else:
            # fallback: choose the largest gated abs error if available
            if "ae_gated_ac_ptamean" in q4.columns:
                q4 = q4.sort_values("ae_gated_ac_ptamean", ascending=False)
                c_id = int(q4.iloc[0]["id"])
            else:
                c_id = int(q4.iloc[0]["id"])

        setC = df[df["id"] == c_id].copy()
        if len(setC) != 1:
            raise ValueError(f"Could not locate Set C id={c_id} in merged dataframe.")
        setC["set"] = "C"
        setC["selection_reason"] = (
            "Rule-based: Q4 (high gate) outlier worst from outliers_csv "
            "(min delta_abs_err_ptamean if available); limitation example"
        )
    else:
        # If no outliers_csv, try to compute Q4 from gate_mean (must exist from gated_oof_csv)
        if "gate_mean" not in df.columns or df["gate_mean"].isna().all():
            raise ValueError(
                "Set C requires either --outliers_csv (recommended) or --gated_oof_csv providing gate_mean."
            )
        gate = pd.to_numeric(df["gate_mean"], errors="coerce")
        q3 = np.nanquantile(gate, 0.75)
        q4_pool = df[gate >= q3].copy()
        # worst within Q4 by delta_abs_err (TAB->GATED) = |e_gated|-|e_tab|
        q4_pool["delta_abs_err_ptamean"] = q4_pool[ae_gat] - q4_pool[ae_tab]
        q4_pool = q4_pool.sort_values("delta_abs_err_ptamean", ascending=False)  # more positive = worse
        setC = q4_pool.head(1).copy()
        setC["set"] = "C"
        setC["selection_reason"] = (
            "Rule-based: Q4 by gate_mean>=75th percentile; pick worst by (ae_gated - ae_tab); limitation example"
        )

    # ------------------------------------------------------------
    # Compute gate quartiles over ALL cases (from gate_mean)
    # Adds:
    #   gate_quartile_all: Q1/Q2/Q3/Q4 computed on the entire cohort
    # ------------------------------------------------------------
    if "gate_mean" in df.columns:
        gm = pd.to_numeric(df["gate_mean"], errors="coerce")
        if gm.notna().sum() >= 10:
            try:
                df["gate_quartile_all"] = pd.qcut(gm, q=4, labels=["Q1", "Q2", "Q3", "Q4"])
            except ValueError:
                # Fallback: qcut on ranks (stable even with ties)
                r = gm.rank(method="average")
                df["gate_quartile_all"] = pd.qcut(r, q=4, labels=["Q1", "Q2", "Q3", "Q4"])
        else:
            df["gate_quartile_all"] = np.nan



    # ------------------------------------------------------------
    # Combine and export
    # ------------------------------------------------------------
    keep_cols = [
        "set", "id",
        "selection_reason",
        "fold", "side",
        "pre_mean_air", "pre_mean_air_tertile",
        "disease", "primary_or_recur",
        "is_attic", "is_primary",
        ae_tab, ae_res, ae_gat,
        "delta_tab_resid", "delta_resid_gated",
        # optional gate/outlier context if present
        "gate_mean","gate_quartile_all", "gate_quartile", "delta_abs_err_ptamean",
        "ae_tab_ac_ptamean", "ae_gated_ac_ptamean",
    ]
    for c in keep_cols:
        if c not in df.columns:
            # allow missing optional columns
            pass

    def select_cols(frame: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in keep_cols if c in frame.columns]
        out = frame[cols].copy()
        # tidy types
        if "id" in out.columns:
            out["id"] = out["id"].astype(int)
        return out

    out_df = pd.concat([select_cols(setA), select_cols(setB), select_cols(setC)], ignore_index=True)

    # Sort for readability
    out_df["set_order"] = out_df["set"].map({"A": 0, "B": 1, "C": 2}).fillna(9).astype(int)
    out_df = out_df.sort_values(["set_order", "id"]).drop(columns=["set_order"])

    # Output path
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = per_path.parent / "case_list.csv"

    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"[OK] Wrote: {out_path}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
