from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from chole_predict.analysis.metrics import evaluate_metrics
from chole_predict.data.categorical import build_category_map
from chole_predict.data.feature_schema import infer_tabular_num_cols
from chole_predict.data.roi_dataset import ResidualROIDataset
from chole_predict.data.scalers import ScalerFromStats
from chole_predict.data.tabular_dataset import TabularBatchSpec, TabularDataset
from chole_predict.models.residual_fusion import ResidualROIModel
from chole_predict.models.tabular_mlp import TabularMLP
from chole_predict.training.residual_engine import predict_residual, run_resid_epoch
from chole_predict.training.tabular_engine import predict_tabular, run_tab_epoch
from chole_predict.utils.parsing import parse_csv_list, parse_shape_csv
from chole_predict.utils.reproducibility import set_seed
from chole_predict.utils.torch_utils import get_grad_scaler


def _existing_roi_mask(df: pd.DataFrame, roi_cols: list[str]) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for col in roi_cols:
        mask &= df[col].notna()
        mask &= df[col].astype(str).map(lambda p: Path(p).exists())
    return mask


def _fit_tabular_model(
    tr_df: pd.DataFrame,
    va_df: pd.DataFrame,
    spec: TabularBatchSpec,
    batch_size: int,
    epochs: int,
    lr: float,
    num_workers: int,
    use_amp: bool,
    device: torch.device,
):
    scaler = StandardScaler().fit(tr_df[spec.num_cols].to_numpy(dtype=np.float32))
    cat_maps = {
        "sex": build_category_map(tr_df["sex"].values),
        "disease": build_category_map(tr_df["disease"].values),
        "primary_or_recur": build_category_map(tr_df["primary_or_recur"].values),
    }

    tr_ds = TabularDataset(tr_df, spec, scaler, cat_maps)
    va_ds = TabularDataset(va_df, spec, scaler, cat_maps)
    tr_ld = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    va_ld = DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = TabularMLP(
        len(spec.num_cols),
        len(cat_maps["sex"]),
        len(cat_maps["disease"]),
        len(cat_maps["primary_or_recur"]),
        len(spec.target_cols),
    ).to(device)
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

    if best_state is None:
        raise RuntimeError("Tabular training failed to produce a checkpoint.")
    model.load_state_dict(best_state)
    return model, scaler, cat_maps, best_val


@torch.no_grad()
def _build_y_tab_map(
    model: TabularMLP,
    df: pd.DataFrame,
    spec: TabularBatchSpec,
    scaler,
    cat_maps: dict[str, dict[str, int]],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    use_amp: bool,
) -> dict[str, np.ndarray]:
    ds = TabularDataset(df, spec, scaler, cat_maps)
    ld = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    ids, preds, _ = predict_tabular(model, ld, device, use_amp=use_amp)
    return {str(pid): preds[i].astype(np.float32) for i, pid in enumerate(ids)}


def _fit_residual_model(
    tr_df: pd.DataFrame,
    va_df: pd.DataFrame,
    spec: TabularBatchSpec,
    scaler,
    cat_maps: dict[str, dict[str, int]],
    roi_cols: list[str],
    out_dhw: tuple[int, int, int],
    y_tab_train: dict[str, np.ndarray],
    y_tab_val: dict[str, np.ndarray],
    roi_pool: str,
    batch_size: int,
    epochs: int,
    lr: float,
    num_workers: int,
    use_amp: bool,
    device: torch.device,
    delta_l2: float,
):
    tr_ds = ResidualROIDataset(tr_df, spec, scaler, cat_maps, roi_cols, y_tab_train, out_dhw=out_dhw)
    va_ds = ResidualROIDataset(va_df, spec, scaler, cat_maps, roi_cols, y_tab_val, out_dhw=out_dhw)
    tr_ld = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    va_ld = DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = ResidualROIModel(
        num_dim=len(spec.num_cols),
        n_sex=len(cat_maps["sex"]),
        n_dis=len(cat_maps["disease"]),
        n_pr=len(cat_maps["primary_or_recur"]),
        out_dim=len(spec.target_cols),
        n_rois=len(roi_cols),
        roi_pool=roi_pool,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scaler_amp = get_grad_scaler(device, enabled=(use_amp and device.type == "cuda"))

    best_val = float("inf")
    best_state = None
    for _ in range(epochs):
        run_resid_epoch(model, tr_ld, opt, device, train=True, use_amp=use_amp, scaler=scaler_amp, delta_l2=delta_l2)
        va_loss = run_resid_epoch(model, va_ld, opt, device, train=False, use_amp=use_amp, scaler=scaler_amp, delta_l2=delta_l2)
        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Residual training failed to produce a checkpoint.")
    model.load_state_dict(best_state)
    return model, best_val


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Train residual ROI-on-tabular model with fixed folds.")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--target", required=True, help="Comma-separated target columns.")
    ap.add_argument("--roi_cols", required=True, help="Comma-separated ROI npy columns.")
    ap.add_argument("--id_col", default="id")
    ap.add_argument("--fold_col", default="fold")
    ap.add_argument("--side_col", default="side")
    ap.add_argument("--roi_pool", default="concat", choices=["concat", "mean"])
    ap.add_argument("--out_dhw", default="160,192,192")
    ap.add_argument("--tab_epochs", type=int, default=300)
    ap.add_argument("--tab_batch", type=int, default=32)
    ap.add_argument("--tab_lr", type=float, default=1e-3)
    ap.add_argument("--roi_epochs", type=int, default=150)
    ap.add_argument("--roi_batch", type=int, default=2)
    ap.add_argument("--roi_lr", type=float, default=1e-3)
    ap.add_argument("--delta_l2", type=float, default=0.0)
    ap.add_argument("--val_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--start_fold", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--out_prefix", default="RESID_ROI_on_TAB_fixedfold")
    return ap


def run_train_residual(
    csv: str | None = None,
    target: str | Iterable[str] | None = None,
    roi_cols: str | Iterable[str] | None = None,
    id_col: str = "id",
    fold_col: str = "fold",
    side_col: str = "side",
    roi_pool: str = "concat",
    out_dhw: str | tuple[int, int, int] = "160,192,192",
    tab_epochs: int = 300,
    tab_batch: int = 32,
    tab_lr: float = 1e-3,
    roi_epochs: int = 150,
    roi_batch: int = 2,
    roi_lr: float = 1e-3,
    delta_l2: float = 0.0,
    val_size: float = 0.2,
    seed: int = 42,
    start_fold: int = 1,
    num_workers: int = 0,
    no_amp: bool = False,
    out_prefix: str = "RESID_ROI_on_TAB_fixedfold",
) -> None:
    if csv is None or target is None or roi_cols is None:
        args = build_parser().parse_args()
        return run_train_residual(
            csv=args.csv,
            target=args.target,
            roi_cols=args.roi_cols,
            id_col=args.id_col,
            fold_col=args.fold_col,
            side_col=args.side_col,
            roi_pool=args.roi_pool,
            out_dhw=args.out_dhw,
            tab_epochs=args.tab_epochs,
            tab_batch=args.tab_batch,
            tab_lr=args.tab_lr,
            roi_epochs=args.roi_epochs,
            roi_batch=args.roi_batch,
            roi_lr=args.roi_lr,
            delta_l2=args.delta_l2,
            val_size=args.val_size,
            seed=args.seed,
            start_fold=args.start_fold,
            num_workers=args.num_workers,
            no_amp=args.no_amp,
            out_prefix=args.out_prefix,
        )

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not no_amp
    target_cols = parse_csv_list(target)
    roi_cols_list = parse_csv_list(roi_cols)
    out_dhw_tuple = parse_shape_csv(out_dhw) if isinstance(out_dhw, str) else out_dhw

    df = pd.read_csv(csv).copy()
    num_cols, pre_mode = infer_tabular_num_cols(df)
    spec = TabularBatchSpec(id_col=id_col, num_cols=num_cols, target_cols=target_cols)
    Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)

    # Keep only rows with all requested targets available.
    target_mask = df[target_cols].notna().all(axis=1)
    df = df.loc[target_mask].copy()

    folds = sorted(int(v) for v in pd.unique(df[fold_col].astype(int)))
    folds = [f for f in folds if f >= int(start_fold)]

    fold_metric_rows: list[pd.DataFrame] = []
    oof_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    for fold in folds:
        test_df = df[df[fold_col].astype(int) == fold].copy()
        train_df = df[df[fold_col].astype(int) != fold].copy()
        tr_df, va_df = train_test_split(train_df, test_size=val_size, random_state=seed)

        tab_model, scaler, cat_maps, best_tab_val = _fit_tabular_model(
            tr_df=tr_df,
            va_df=va_df,
            spec=spec,
            batch_size=tab_batch,
            epochs=tab_epochs,
            lr=tab_lr,
            num_workers=num_workers,
            use_amp=use_amp,
            device=device,
        )

        # Restrict ROI model to cases with existing ROI npy files.
        tr_roi_df = tr_df.loc[_existing_roi_mask(tr_df, roi_cols_list)].copy()
        va_roi_df = va_df.loc[_existing_roi_mask(va_df, roi_cols_list)].copy()
        te_roi_df = test_df.loc[_existing_roi_mask(test_df, roi_cols_list)].copy()
        if tr_roi_df.empty or va_roi_df.empty or te_roi_df.empty:
            raise RuntimeError(
                f"Fold {fold}: ROI-eligible rows are empty after filtering. "
                f"train={len(tr_roi_df)} val={len(va_roi_df)} test={len(te_roi_df)}"
            )

        y_tab_train = _build_y_tab_map(tab_model, tr_roi_df, spec, scaler, cat_maps, tab_batch, num_workers, device, use_amp)
        y_tab_val = _build_y_tab_map(tab_model, va_roi_df, spec, scaler, cat_maps, tab_batch, num_workers, device, use_amp)
        y_tab_test = _build_y_tab_map(tab_model, te_roi_df, spec, scaler, cat_maps, tab_batch, num_workers, device, use_amp)

        resid_model, best_roi_val = _fit_residual_model(
            tr_df=tr_roi_df,
            va_df=va_roi_df,
            spec=spec,
            scaler=scaler,
            cat_maps=cat_maps,
            roi_cols=roi_cols_list,
            out_dhw=out_dhw_tuple,
            y_tab_train=y_tab_train,
            y_tab_val=y_tab_val,
            roi_pool=roi_pool,
            batch_size=roi_batch,
            epochs=roi_epochs,
            lr=roi_lr,
            num_workers=num_workers,
            use_amp=use_amp,
            device=device,
            delta_l2=delta_l2,
        )

        te_ds = ResidualROIDataset(te_roi_df, spec, scaler, cat_maps, roi_cols_list, y_tab_test, out_dhw=out_dhw_tuple, side_col=side_col)
        te_ld = DataLoader(te_ds, batch_size=roi_batch, shuffle=False, num_workers=num_workers)
        ids, y_pred, y_true = predict_residual(resid_model, te_ld, device, use_amp=use_amp)
        metrics = evaluate_metrics(y_true, y_pred, target_cols)
        metrics.insert(0, "fold", fold)
        fold_metric_rows.append(metrics)

        oof = pd.DataFrame({id_col: ids, "fold": fold})
        for k, t in enumerate(target_cols):
            oof[f"true_{t}"] = y_true[:, k]
            oof[f"pred_{t}"] = y_pred[:, k]
            oof[f"err_{t}"] = y_pred[:, k] - y_true[:, k]
        oof_rows.append(oof)

        summary_rows.append({
            "fold": fold,
            "pre_mode": pre_mode,
            "n_train_tab": len(tr_df),
            "n_val_tab": len(va_df),
            "n_train_roi": len(tr_roi_df),
            "n_val_roi": len(va_roi_df),
            "n_test_roi": len(te_roi_df),
            "best_tab_val_loss": best_tab_val,
            "best_roi_val_loss": best_roi_val,
        })

        tab_ckpt = {
            "model_state_dict": tab_model.state_dict(),
            "num_cols": spec.num_cols,
            "target_cols": spec.target_cols,
            "cat_maps": cat_maps,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "fold": fold,
        }
        torch.save(tab_ckpt, f"{out_prefix}_fold{fold}_tabular.pt")

        resid_ckpt = {
            "model_state_dict": resid_model.state_dict(),
            "num_cols": spec.num_cols,
            "target_cols": spec.target_cols,
            "roi_cols": roi_cols_list,
            "roi_pool": roi_pool,
            "out_dhw": list(out_dhw_tuple),
            "cat_maps": cat_maps,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "fold": fold,
        }
        torch.save(resid_ckpt, f"{out_prefix}_fold{fold}_residual.pt")

    if fold_metric_rows:
        pd.concat(fold_metric_rows, ignore_index=True).to_csv(f"{out_prefix}_fold_metrics.csv", index=False)
    if oof_rows:
        pd.concat(oof_rows, ignore_index=True).to_csv(f"{out_prefix}_oof_predictions.csv", index=False)
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(f"{out_prefix}_training_summary.csv", index=False)


def load_tabular_checkpoint(ckpt_path: str | Path, device: str | torch.device = "cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)
    scaler = ScalerFromStats(ckpt["scaler_mean"], ckpt["scaler_scale"])
    cat_maps = ckpt["cat_maps"]
    model = TabularMLP(
        num_dim=len(ckpt["num_cols"]),
        n_sex=len(cat_maps["sex"]),
        n_dis=len(cat_maps["disease"]),
        n_pr=len(cat_maps["primary_or_recur"]),
        out_dim=len(ckpt["target_cols"]),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, scaler, cat_maps, ckpt


__all__ = ["build_parser", "run_train_residual", "load_tabular_checkpoint"]
