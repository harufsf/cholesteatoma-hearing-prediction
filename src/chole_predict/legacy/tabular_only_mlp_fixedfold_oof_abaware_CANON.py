# -*- coding: utf-8 -*-
"""
tabular_only_mlp_fixedfold_oof.py

Clinical-only (no ROI) regression with fixed folds from df_final_fixed.csv.
- Same feature set as RF: age/sex/pre_PTA(0.5,1,2,3)/disease/primary_or_recur
- Same fold split as ROI model (fold column)
- Multi-output regression (4 frequencies)
- Saves:
   - <out_prefix>_fold_metrics.csv
   - <out_prefix>_summary_mean_sd.csv
   - <out_prefix>_oof_predictions.csv

Run:
  conda activate dl_fusion_bw
  python tabular_only_mlp_fixedfold_oof.py ^
    --csv df_final_fixed.csv ^
    --id_col id ^
    --fold_col fold ^
    --target post_PTA_0.5k,post_PTA_1k,post_PTA_2k,post_PTA_3k ^
    --epochs 300 --batch 32 --out_prefix TAB_MLP_out4_fixedfold
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# clean output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def parse_list_csv(s):
    return [x.strip() for x in str(s).split(",") if x.strip()]


def build_cat_map(values):
    uniq = sorted(set([str(v) for v in values]))
    m = {"__UNK__": 0}
    for i, v in enumerate(uniq, start=1):
        m[v] = i
    return m


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def pearsonr_safe(a, b):
    a = np.asarray(a).astype(np.float64)
    b = np.asarray(b).astype(np.float64)
    if a.size < 2:
        return np.nan
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def get_autocast(device, enabled: bool):
    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", enabled=enabled)
    return torch.amp.autocast(device_type="cpu", enabled=False)


def get_grad_scaler(device, enabled: bool):
    if device.type == "cuda":
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.cuda.amp.GradScaler(enabled=enabled)
    else:
        try:
            return torch.amp.GradScaler("cpu", enabled=False)
        except TypeError:
            return torch.cuda.amp.GradScaler(enabled=False)


class TabularDataset(Dataset):
    def __init__(self, df, id_col, num_cols, target_cols, scaler, sex_map, dis_map, pr_map):
        self.df = df.reset_index(drop=True)
        self.id_col = id_col
        self.num_cols = num_cols
        self.target_cols = target_cols
        self.scaler = scaler
        self.sex_map = sex_map
        self.dis_map = dis_map
        self.pr_map = pr_map

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = str(row[self.id_col])

        num = row[self.num_cols].to_numpy(dtype=np.float32).reshape(1, -1)
        num = self.scaler.transform(num).astype(np.float32).reshape(-1)
        num_t = torch.from_numpy(num)

        sex_id = self.sex_map.get(str(row["sex"]), self.sex_map.get("__UNK__", 0))
        dis_id = self.dis_map.get(str(row["disease"]), self.dis_map.get("__UNK__", 0))
        pr_id  = self.pr_map.get(str(row["primary_or_recur"]), self.pr_map.get("__UNK__", 0))

        sex_t = torch.tensor(sex_id, dtype=torch.long)
        dis_t = torch.tensor(dis_id, dtype=torch.long)
        pr_t  = torch.tensor(pr_id,  dtype=torch.long)

        y = row[self.target_cols].to_numpy(dtype=np.float32)
        y_t = torch.from_numpy(y)

        return pid, num_t, sex_t, dis_t, pr_t, y_t


class TabularMLP(nn.Module):
    def __init__(self, num_dim, n_sex, n_dis, n_pr, out_dim,
                 sex_emb=8, dis_emb=8, pr_emb=8, hidden=128):
        super().__init__()
        self.sex_emb = nn.Embedding(n_sex, sex_emb)
        self.dis_emb = nn.Embedding(n_dis, dis_emb)
        self.pr_emb  = nn.Embedding(n_pr,  pr_emb)

        in_dim = num_dim + sex_emb + dis_emb + pr_emb
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, num, sex_id, dis_id, pr_id):
        sex_e = self.sex_emb(sex_id)
        dis_e = self.dis_emb(dis_id)
        pr_e  = self.pr_emb(pr_id)
        x = torch.cat([num, sex_e, dis_e, pr_e], dim=1)
        return self.net(x)


def run_epoch(model, loader, optim, device, train=True, use_amp=True, scaler=None):
    model.train(train)
    losses = []
    for _, num, sex_id, dis_id, pr_id, y in loader:
        num = num.to(device, non_blocking=True)
        sex_id = sex_id.to(device, non_blocking=True)
        dis_id = dis_id.to(device, non_blocking=True)
        pr_id  = pr_id.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            with get_autocast(device, enabled=(use_amp and device.type == "cuda")):
                pred = model(num, sex_id, dis_id, pr_id)
                loss = F.smooth_l1_loss(pred, y)

            if train:
                optim.zero_grad(set_to_none=True)
                if use_amp and device.type == "cuda":
                    scaler.scale(loss).backward()
                    scaler.step(optim)
                    scaler.update()
                else:
                    loss.backward()
                    optim.step()

        losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses))


@torch.no_grad()
def predict(model, loader, device, use_amp=True):
    model.eval()
    ids, preds, ys = [], [], []
    for pid, num, sex_id, dis_id, pr_id, y in loader:
        num = num.to(device, non_blocking=True)
        sex_id = sex_id.to(device, non_blocking=True)
        dis_id = dis_id.to(device, non_blocking=True)
        pr_id  = pr_id.to(device, non_blocking=True)

        with get_autocast(device, enabled=(use_amp and device.type == "cuda")):
            pred = model(num, sex_id, dis_id, pr_id)

        ids.extend(list(pid))
        preds.append(pred.detach().cpu().numpy())
        ys.append(y.numpy())

    preds = np.concatenate(preds, 0)
    ys = np.concatenate(ys, 0)
    return ids, preds, ys


def evaluate_metrics(y_true, y_pred, target_cols):
    rows = []
    for k, name in enumerate(target_cols):
        yt = y_true[:, k]
        yp = y_pred[:, k]
        rows.append({
            "target": name,
            "MAE": float(mean_absolute_error(yt, yp)),
            "RMSE": rmse(yt, yp),
            "R2": float(r2_score(yt, yp)) if yt.size >= 2 else np.nan,
            "Pearson_r": pearsonr_safe(yt, yp),
        })
    return pd.DataFrame(rows)


class _ScalerFromStats:
    def __init__(self, mean, scale):
        self.mean_ = np.asarray(mean, dtype=np.float32)
        self.scale_ = np.asarray(scale, dtype=np.float32)
    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        return (X - self.mean_) / (self.scale_ + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--target", required=True)  # comma separated
    ap.add_argument("--id_col", default="id")
    ap.add_argument("--fold_col", default="fold")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_prefix", default="tabular_mlp_fixedfold")
    ap.add_argument("--start_fold", type=int, default=1)
    ap.add_argument("--skip_trained_folds", action="store_true")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--no_amp", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = pd.read_csv(args.csv)

    # ---- AB-aware pre-op PTA features ----

    legacy = ["pre_PTA_0.5k","pre_PTA_1k","pre_PTA_2k","pre_PTA_3k"]

    a_cols = ["pre_PTA_0.5k_A","pre_PTA_1k_A","pre_PTA_2k_A","pre_PTA_3k_A"]

    b_cols = ["pre_PTA_0.5k_B","pre_PTA_1k_B","pre_PTA_2k_B","pre_PTA_3k_B"]

    if all(c in df.columns for c in legacy):

        pre_cols = legacy

        pre_mode = "legacy_air_only"

    elif all(c in df.columns for c in a_cols) and all(c in df.columns for c in b_cols):

        pre_cols = a_cols + b_cols

        pre_mode = "air_and_bone"

    elif all(c in df.columns for c in a_cols):

        pre_cols = a_cols

        pre_mode = "air_only_A"

    else:

        raise ValueError("Cannot infer pre-op PTA columns. Expected legacy pre_PTA_* or new pre_PTA_*_A (and optionally *_B).")

    num_cols = ["age"] + pre_cols

    print(f"[INFO] Using pre-op feature mode: {pre_mode}; num_cols={num_cols}")

    # ---- end AB-aware block ----

    target_cols = parse_list_csv(args.target)
    num_cols = ["age"]  # will be inferred after reading CSV (AB-aware)

    required = [args.id_col, args.fold_col, "sex", "disease", "primary_or_recur"] + num_cols + target_cols
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # Drop NaNs just in case (df_final_fixed should already be clean)
    before = len(df)
    df = df.dropna(subset=required).reset_index(drop=True)
    if len(df) < before:
        print(f"[INFO] Dropped rows with NaN: {before-len(df)} (kept {len(df)})")

    folds = sorted(pd.unique(df[args.fold_col].astype(int)))
    print(f"[INFO] Using fixed folds from '{args.fold_col}': {folds}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp)

    fold_metrics = []
    oof_rows = []

    for fold in folds:
        if fold < args.start_fold:
            continue

        ckpt_path = f"{args.out_prefix}_fold{fold}_best.pt"

        test_df = df[df[args.fold_col].astype(int) == fold].copy()
        train_df = df[df[args.fold_col].astype(int) != fold].copy()
        if len(test_df) == 0 or len(train_df) == 0:
            print(f"[WARN] Fold {fold}: empty split. Skipping.")
            continue

        if args.skip_trained_folds and os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            scaler = _ScalerFromStats(ckpt["scaler_mean"], ckpt["scaler_scale"])
            sex_map = ckpt["sex_map"]
            dis_map = ckpt["dis_map"]
            pr_map  = ckpt["pr_map"]
            num_cols_ckpt = ckpt.get("num_cols", num_cols)

            te_ds = TabularDataset(test_df, args.id_col, num_cols_ckpt, target_cols, scaler, sex_map, dis_map, pr_map)
            te_ld = DataLoader(te_ds, batch_size=args.batch, shuffle=False,
                               num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

            model = TabularMLP(
                num_dim=len(num_cols_ckpt),
                n_sex=len(sex_map),
                n_dis=len(dis_map),
                n_pr=len(pr_map),
                out_dim=len(target_cols),
            ).to(device)
            model.load_state_dict(ckpt["model"])

            ids, y_pred, y_true = predict(model, te_ld, device, use_amp=use_amp)

            metrics = evaluate_metrics(y_true, y_pred, target_cols)
            metrics.insert(0, "fold", fold)
            fold_metrics.append(metrics)

            oof_df = pd.DataFrame({args.id_col: ids, "fold": fold})
            for k, name in enumerate(target_cols):
                oof_df["true_" + name] = y_true[:, k]
                oof_df["pred_" + name] = y_pred[:, k]
                oof_df["err_" + name]  = y_pred[:, k] - y_true[:, k]
            oof_rows.append(oof_df)

            print(f"Fold {fold} metrics (resumed):\n{metrics}")
            continue

        tr_df, va_df = train_test_split(train_df, test_size=args.val_size, random_state=args.seed)

        scaler = StandardScaler()
        scaler.fit(tr_df[num_cols].to_numpy(dtype=np.float32))

        sex_map = build_cat_map(tr_df["sex"].values)
        dis_map = build_cat_map(tr_df["disease"].values)
        pr_map  = build_cat_map(tr_df["primary_or_recur"].values)

        tr_ds = TabularDataset(tr_df, args.id_col, num_cols, target_cols, scaler, sex_map, dis_map, pr_map)
        va_ds = TabularDataset(va_df, args.id_col, num_cols, target_cols, scaler, sex_map, dis_map, pr_map)
        te_ds = TabularDataset(test_df, args.id_col, num_cols, target_cols, scaler, sex_map, dis_map, pr_map)

        pin = torch.cuda.is_available()
        tr_ld = DataLoader(tr_ds, batch_size=args.batch, shuffle=True,  num_workers=args.num_workers, pin_memory=pin)
        va_ld = DataLoader(va_ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
        te_ld = DataLoader(te_ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, pin_memory=pin)

        model = TabularMLP(
            num_dim=len(num_cols),
            n_sex=len(sex_map),
            n_dis=len(dis_map),
            n_pr=len(pr_map),
            out_dim=len(target_cols),
        ).to(device)

        optim = torch.optim.Adam(model.parameters(), lr=args.lr)
        scaler_amp = get_grad_scaler(device, enabled=(use_amp and device.type == "cuda"))

        best_val = 1e18
        for ep in range(1, args.epochs + 1):
            tr_loss = run_epoch(model, tr_ld, optim, device, train=True,  use_amp=use_amp, scaler=scaler_amp)
            va_loss = run_epoch(model, va_ld, optim, device, train=False, use_amp=use_amp, scaler=scaler_amp)

            if ep == 1 or ep % 20 == 0 or ep == args.epochs:
                print(f"Fold {fold} Ep {ep:03d} | train {tr_loss:.4f} | val {va_loss:.4f}")

            if va_loss < best_val:
                best_val = va_loss
                torch.save({
                    "model": model.state_dict(),
                    "scaler_mean": scaler.mean_,
                    "scaler_scale": scaler.scale_,
                    "num_cols": num_cols,
                    "sex_map": sex_map,
                    "dis_map": dis_map,
                    "pr_map": pr_map,
                    "target_cols": target_cols,
                    "fold": fold,
                    "torch_version": torch.__version__,
                }, ckpt_path)

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])

        ids, y_pred, y_true = predict(model, te_ld, device, use_amp=use_amp)

        metrics = evaluate_metrics(y_true, y_pred, target_cols)
        metrics.insert(0, "fold", fold)
        fold_metrics.append(metrics)

        oof_df = pd.DataFrame({args.id_col: ids, "fold": fold})
        for k, name in enumerate(target_cols):
            oof_df["true_" + name] = y_true[:, k]
            oof_df["pred_" + name] = y_pred[:, k]
            oof_df["err_" + name]  = y_pred[:, k] - y_true[:, k]
        oof_rows.append(oof_df)

        print(f"Fold {fold} metrics:\n{metrics}")

    # save
    if fold_metrics:
        fold_metrics_df = pd.concat(fold_metrics, ignore_index=True)
        fold_metrics_path = f"{args.out_prefix}_fold_metrics.csv"
        fold_metrics_df.to_csv(fold_metrics_path, index=False, encoding="utf-8-sig")

        summary = (
            fold_metrics_df.groupby("target")[["MAE","RMSE","R2","Pearson_r"]]
            .agg(["mean","std"])
        )
        summary_path = f"{args.out_prefix}_summary_mean_sd.csv"
        summary.to_csv(summary_path, encoding="utf-8-sig")

        print("\n=== K-FOLD SUMMARY (mean ± sd) ===")
        out_rows = []
        for t in fold_metrics_df["target"].unique():
            sub = fold_metrics_df[fold_metrics_df["target"] == t]
            out_rows.append({
                "target": t,
                "MAE_mean": sub["MAE"].mean(), "MAE_sd": sub["MAE"].std(ddof=1),
                "RMSE_mean": sub["RMSE"].mean(), "RMSE_sd": sub["RMSE"].std(ddof=1),
                "R2_mean": sub["R2"].mean(), "R2_sd": sub["R2"].std(ddof=1),
                "Pearson_r_mean": sub["Pearson_r"].mean(), "Pearson_r_sd": sub["Pearson_r"].std(ddof=1),
            })
        print(pd.DataFrame(out_rows))

        print(f"[SAVED] fold metrics -> {fold_metrics_path}")
        print(f"[SAVED] summary     -> {summary_path}")

    if oof_rows:
        oof_all = pd.concat(oof_rows, ignore_index=True)
        oof_path = f"{args.out_prefix}_oof_predictions.csv"
        oof_all.to_csv(oof_path, index=False, encoding="utf-8-sig")
        print(f"[SAVED] OOF predictions -> {oof_path}")


if __name__ == "__main__":
    main()
