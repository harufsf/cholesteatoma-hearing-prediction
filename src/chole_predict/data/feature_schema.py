from __future__ import annotations

import pandas as pd


def infer_tabular_num_cols(df: pd.DataFrame) -> tuple[list[str], str]:
    legacy = ["pre_PTA_0.5k", "pre_PTA_1k", "pre_PTA_2k", "pre_PTA_3k"]
    a_cols = ["pre_PTA_0.5k_A", "pre_PTA_1k_A", "pre_PTA_2k_A", "pre_PTA_3k_A"]
    b_cols = ["pre_PTA_0.5k_B", "pre_PTA_1k_B", "pre_PTA_2k_B", "pre_PTA_3k_B"]
    if all(c in df.columns for c in legacy):
        return ["age"] + legacy, "legacy_air_only"
    if all(c in df.columns for c in a_cols) and all(c in df.columns for c in b_cols):
        return ["age"] + a_cols + b_cols, "air_and_bone"
    if all(c in df.columns for c in a_cols):
        return ["age"] + a_cols, "air_only_A"
    raise ValueError("Cannot infer pre-op PTA columns.")
