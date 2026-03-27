from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def pearsonr_safe(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2:
        return float("nan")
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def evaluate_metrics(y_true, y_pred, target_cols: list[str]) -> pd.DataFrame:
    rows = []
    for k, name in enumerate(target_cols):
        yt = y_true[:, k]
        yp = y_pred[:, k]
        rows.append({
            "target": name,
            "MAE": float(mean_absolute_error(yt, yp)),
            "RMSE": rmse(yt, yp),
            "R2": float(r2_score(yt, yp)) if yt.size >= 2 else float("nan"),
            "Pearson_r": pearsonr_safe(yt, yp),
        })
    return pd.DataFrame(rows)
