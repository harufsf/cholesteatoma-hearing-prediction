from __future__ import annotations

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from chole_predict.analysis.metrics import evaluate_metrics
from chole_predict.data.categorical import build_category_map
from chole_predict.data.feature_schema import infer_tabular_num_cols
from chole_predict.data.tabular_dataset import TabularBatchSpec, TabularDataset
from chole_predict.models.tabular_mlp import TabularMLP
from chole_predict.training.tabular_engine import predict_tabular, run_tab_epoch
from chole_predict.utils.reproducibility import set_seed
from chole_predict.utils.torch_utils import get_grad_scaler


def run_train_tabular(csv_path: str, target_cols: list[str], out_prefix: str, id_col: str = "id", fold_col: str = "fold", seed: int = 42, batch_size: int = 32, epochs: int = 300, lr: float = 1e-3, val_size: float = 0.2, num_workers: int = 0, use_amp: bool = True, device_name: str | None = None):
    set_seed(seed)
    df = pd.read_csv(csv_path)
    num_cols, _ = infer_tabular_num_cols(df)
    spec = TabularBatchSpec(id_col=id_col, num_cols=num_cols, target_cols=target_cols)
    folds = sorted(pd.unique(df[fold_col].astype(int)))
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    oof_rows = []
    fold_metrics = []
    for fold in folds:
        test_df = df[df[fold_col].astype(int) == fold].copy()
        train_df = df[df[fold_col].astype(int) != fold].copy()
        tr_df, va_df = train_test_split(train_df, test_size=val_size, random_state=seed)
        scaler = StandardScaler().fit(tr_df[num_cols].to_numpy(dtype="float32"))
        cat_maps = {
            "sex": build_category_map(tr_df["sex"].values),
            "disease": build_category_map(tr_df["disease"].values),
            "primary_or_recur": build_category_map(tr_df["primary_or_recur"].values),
        }
        tr_ds = TabularDataset(tr_df, spec, scaler, cat_maps)
        va_ds = TabularDataset(va_df, spec, scaler, cat_maps)
        te_ds = TabularDataset(test_df, spec, scaler, cat_maps)
        tr_ld = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        va_ld = DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        te_ld = DataLoader(te_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        model = TabularMLP(len(num_cols), len(cat_maps["sex"]), len(cat_maps["disease"]), len(cat_maps["primary_or_recur"]), len(target_cols)).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        scaler_amp = get_grad_scaler(device, enabled=(use_amp and device.type == "cuda"))
        best_val = float("inf")
        best_state = None
        for _ in range(epochs):
            run_tab_epoch(model, tr_ld, opt, device, train=True, use_amp=use_amp, scaler=scaler_amp)
            va_loss = run_tab_epoch(model, va_ld, opt, device, train=False, use_amp=use_amp, scaler=scaler_amp)
            if va_loss < best_val:
                best_val = va_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        assert best_state is not None
        model.load_state_dict(best_state)
        ids, y_pred, y_true = predict_tabular(model, te_ld, device, use_amp=use_amp)
        metrics = evaluate_metrics(y_true, y_pred, target_cols)
        metrics.insert(0, "fold", fold)
        fold_metrics.append(metrics)
        oof = pd.DataFrame({id_col: ids, "fold": fold})
        for k, t in enumerate(target_cols):
            oof[f"true_{t}"] = y_true[:, k]
            oof[f"pred_{t}"] = y_pred[:, k]
            oof[f"err_{t}"] = y_pred[:, k] - y_true[:, k]
        oof_rows.append(oof)
    if fold_metrics:
        pd.concat(fold_metrics, ignore_index=True).to_csv(f"{out_prefix}_fold_metrics.csv", index=False)
    if oof_rows:
        pd.concat(oof_rows, ignore_index=True).to_csv(f"{out_prefix}_oof_predictions.csv", index=False)
