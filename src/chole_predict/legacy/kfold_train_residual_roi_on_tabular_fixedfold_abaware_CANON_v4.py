# -*- coding: utf-8 -*-
"""
kfold_train_residual_roi_on_tabular_fixedfold.py

Residual model:
  y_hat = y_tab_hat + delta_roi

- Fixed folds from CSV
- Uses same tabular features as RF:
    age, sex, disease, primary_or_recur, pre_PTA_0.5k,1k,2k,3k
- ROI inputs: roi_path_25, roi_path_40, roi_path_60 (or any list)
- Multi-output regression (4 freq)
- Saves OOF predictions + metrics

Run example:
  conda activate dl_fusion_bw
  cd C:\\Users\\path\\to\\project\\directory

  python kfold_train_residual_roi_on_tabular_fixedfold.py ^
    --csv df_final_fixed.csv ^
    --id_col id ^
    --fold_col fold ^
    --roi_cols roi_path_25,roi_path_40,roi_path_60 ^
    --target post_PTA_0.5k,post_PTA_1k,post_PTA_2k,post_PTA_3k ^
    --tab_epochs 300 --tab_batch 32 ^
    --roi_epochs 150 --roi_batch 2 ^
    --out_prefix RESID_ROI_on_TAB_attn_out4_fixedfold
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd

def infer_tabular_num_cols(df):
    """
    Infer pre-op PTA numeric columns.
    - legacy: pre_PTA_0.5k/1k/2k/3k (air only)
    - new   : pre_PTA_0.5k_A..3k_A and pre_PTA_0.5k_B..3k_B (use BOTH air & bone as inputs)
    """
    legacy = ["pre_PTA_0.5k","pre_PTA_1k","pre_PTA_2k","pre_PTA_3k"]
    a_cols = ["pre_PTA_0.5k_A","pre_PTA_1k_A","pre_PTA_2k_A","pre_PTA_3k_A"]
    b_cols = ["pre_PTA_0.5k_B","pre_PTA_1k_B","pre_PTA_2k_B","pre_PTA_3k_B"]
    if all(c in df.columns for c in legacy):
        return ["age"] + legacy, "legacy_air_only"
    if all(c in df.columns for c in a_cols) and all(c in df.columns for c in b_cols):
        return ["age"] + a_cols + b_cols, "air_and_bone"
    if all(c in df.columns for c in a_cols):  # allow A-only if B missing
        return ["age"] + a_cols, "air_only_A"
    raise ValueError(
        "Cannot infer pre-op PTA columns. Expected either "
        "pre_PTA_0.5k/1k/2k/3k (legacy) or "
        "pre_PTA_0.5k_A..3k_A (and ideally *_B)."
    )

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


# -----------------------
# AMP helpers (no FutureWarning)
# -----------------------
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


# -----------------------
# Tabular-only dataset/model
# -----------------------
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
            nn.Linear(in_dim, hidden), nn.ReLU(True), nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.ReLU(True), nn.Dropout(0.1),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, num, sex_id, dis_id, pr_id):
        sex_e = self.sex_emb(sex_id)
        dis_e = self.dis_emb(dis_id)
        pr_e  = self.pr_emb(pr_id)
        x = torch.cat([num, sex_e, dis_e, pr_e], dim=1)
        return self.net(x)


def run_epoch_tab(model, loader, optim, device, train=True, use_amp=True, scaler=None):
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
def predict_tab(model, loader, device, use_amp=True):
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
    return ids, np.concatenate(preds, 0), np.concatenate(ys, 0)


# -----------------------
# ROI (multi-ROI) parts
# -----------------------
def hu_preprocess(vol_hu, clip_low=-1000.0, clip_high=2000.0):
    x = np.clip(vol_hu.astype(np.float32), clip_low, clip_high)
    x = (x - clip_low) / (clip_high - clip_low + 1e-6)
    return x


def resize_3d_torch(x, out_dhw):
    x = x.unsqueeze(0)  # (1,1,D,H,W)
    x = F.interpolate(x, size=out_dhw, mode="trilinear", align_corners=False)
    return x.squeeze(0)


def normalize_side(s):
    s = str(s).strip().upper()
    if s in ["R", "RIGHT", "右"]:
        return "R"
    if s in ["L", "LEFT", "左"]:
        return "L"
    return None


def unify_to_right(vol_zyx, side):
    sd = normalize_side(side)
    if sd is None:
        raise ValueError(f"side must be R/L (got {side})")
    if sd == "L":
        return np.flip(vol_zyx, axis=2).copy()
    return vol_zyx


class CTTabResidualDataset(Dataset):
    """
    Returns:
      pid, side, vols, num, sex_id, dis_id, pr_id, y_true, y_tab_hat
    """
    def __init__(self, df, id_col, roi_cols, num_cols, target_cols,
                 scaler, out_dhw, sex_map, dis_map, pr_map, ytab_map):
        self.df = df.reset_index(drop=True)
        self.id_col = id_col
        self.roi_cols = roi_cols
        self.num_cols = num_cols
        self.target_cols = target_cols
        self.scaler = scaler
        self.out_dhw = out_dhw
        self.sex_map = sex_map
        self.dis_map = dis_map
        self.pr_map = pr_map
        self.ytab_map = ytab_map  # dict pid -> y_tab_hat (np array)

    def __len__(self):
        return len(self.df)

    def _load_one(self, roi_path, side):
        vol = np.load(roi_path)  # (Z,Y,X)
        vol = unify_to_right(vol, side)
        vol = hu_preprocess(vol)
        vol_t = torch.from_numpy(vol).unsqueeze(0)      # (1,Z,Y,X)
        vol_t = resize_3d_torch(vol_t, self.out_dhw)    # (1,D,H,W)
        return vol_t

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = str(row[self.id_col])
        side = row["side"]

        vols = []
        for c in self.roi_cols:
            vols.append(self._load_one(str(row[c]), side))
        vols_t = torch.stack(vols, dim=0)  # (R,1,D,H,W)

        num = row[self.num_cols].to_numpy(dtype=np.float32).reshape(1, -1)
        num = self.scaler.transform(num).astype(np.float32).reshape(-1)
        num_t = torch.from_numpy(num)

        sex_id = self.sex_map.get(str(row["sex"]), self.sex_map.get("__UNK__", 0))
        dis_id = self.dis_map.get(str(row["disease"]), self.dis_map.get("__UNK__", 0))
        pr_id  = self.pr_map.get(str(row["primary_or_recur"]), self.pr_map.get("__UNK__", 0))

        sex_t = torch.tensor(sex_id, dtype=torch.long)
        dis_t = torch.tensor(dis_id, dtype=torch.long)
        pr_t  = torch.tensor(pr_id,  dtype=torch.long)

        y_true = row[self.target_cols].to_numpy(dtype=np.float32)
        y_true_t = torch.from_numpy(y_true)

        y_tab = self.ytab_map[pid].astype(np.float32)
        y_tab_t = torch.from_numpy(y_tab)

        return pid, side, vols_t, num_t, sex_t, dis_t, pr_t, y_true_t, y_tab_t


class Small3DCNN(nn.Module):
    def __init__(self, in_ch=1, base=16, emb_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, base, 3, padding=1), nn.BatchNorm3d(base), nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(base, base*2, 3, padding=1), nn.BatchNorm3d(base*2), nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(base*2, base*4, 3, padding=1), nn.BatchNorm3d(base*4), nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(base*4, base*8, 3, padding=1), nn.BatchNorm3d(base*8), nn.ReLU(True),
        )
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(base*8, emb_dim)

    def forward(self, x):
        x = self.net(x)
        x = self.gap(x).flatten(1)
        return self.fc(x)


class MultiROIPool(nn.Module):
    def __init__(self, emb_dim, mode="attn"):
        super().__init__()
        self.mode = mode
        if mode == "attn":
            self.score = nn.Linear(emb_dim, 1)
        elif mode == "mean":
            self.score = None
        else:
            raise ValueError("mode must be 'mean' or 'attn'")

    def forward(self, x):  # (B,R,E)
        if self.mode == "mean":
            return x.mean(dim=1)
        w = self.score(x).squeeze(-1)
        w = torch.softmax(w, dim=1)
        return (x * w.unsqueeze(-1)).sum(1)


class ResidualROIModel(nn.Module):
    """
    Outputs delta (same dim as targets)
    """
    def __init__(self, num_dim, n_sex, n_dis, n_pr, out_dim, n_rois,
                 roi_pool="attn", img_emb=128, sex_emb=8, dis_emb=8, pr_emb=8, hidden=256):
        super().__init__()
        self.encoder = Small3DCNN(in_ch=1, base=16, emb_dim=img_emb)
        self.pool = MultiROIPool(img_emb, mode=roi_pool)

        self.sex_emb = nn.Embedding(n_sex, sex_emb)
        self.dis_emb = nn.Embedding(n_dis, dis_emb)
        self.pr_emb  = nn.Embedding(n_pr,  pr_emb)

        in_dim = img_emb + num_dim + sex_emb + dis_emb + pr_emb
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, vols, num, sex_id, dis_id, pr_id):
        B, R = vols.shape[0], vols.shape[1]
        x = vols.view(B*R, *vols.shape[2:])
        e = self.encoder(x).view(B, R, -1)
        img_e = self.pool(e)

        x = torch.cat([img_e, num,
                       self.sex_emb(sex_id),
                       self.dis_emb(dis_id),
                       self.pr_emb(pr_id)], dim=1)
        return self.head(x)  # delta


def run_epoch_resid(model, loader, optim, device, train=True, use_amp=True, scaler=None, delta_l2=0.0):
    model.train(train)
    losses = []
    for _, _, vols, num, sex_id, dis_id, pr_id, y_true, y_tab in loader:
        vols = vols.to(device, non_blocking=True)
        num  = num.to(device, non_blocking=True)
        sex_id = sex_id.to(device, non_blocking=True)
        dis_id = dis_id.to(device, non_blocking=True)
        pr_id  = pr_id.to(device, non_blocking=True)
        y_true = y_true.to(device, non_blocking=True)
        y_tab  = y_tab.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            with get_autocast(device, enabled=(use_amp and device.type == "cuda")):
                delta = model(vols, num, sex_id, dis_id, pr_id)
                y_hat = y_tab + delta
                loss = F.smooth_l1_loss(y_hat, y_true)
                if delta_l2 > 0:
                    loss = loss + delta_l2 * torch.mean(delta * delta)

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
def predict_resid(model, loader, device, use_amp=True):
    model.eval()
    ids, sides, preds, ys = [], [], [], []
    for pid, side, vols, num, sex_id, dis_id, pr_id, y_true, y_tab in loader:
        vols = vols.to(device, non_blocking=True)
        num  = num.to(device, non_blocking=True)
        sex_id = sex_id.to(device, non_blocking=True)
        dis_id = dis_id.to(device, non_blocking=True)
        pr_id  = pr_id.to(device, non_blocking=True)
        y_tab  = y_tab.to(device, non_blocking=True)

        with get_autocast(device, enabled=(use_amp and device.type == "cuda")):
            delta = model(vols, num, sex_id, dis_id, pr_id)
            y_hat = y_tab + delta

        ids.extend(list(pid))
        sides.extend(list(side))
        preds.append(y_hat.detach().cpu().numpy())
        ys.append(y_true.numpy())
    return ids, sides, np.concatenate(preds, 0), np.concatenate(ys, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--id_col", default="id")
    ap.add_argument("--fold_col", default="fold")
    ap.add_argument("--roi_cols", required=True)
    ap.add_argument("--roi_pool", default="attn", choices=["attn", "mean"])
    ap.add_argument("--out_dhw", default="160,192,192")

    # tabular baseline training
    ap.add_argument("--tab_epochs", type=int, default=300)
    ap.add_argument("--tab_batch", type=int, default=32)
    ap.add_argument("--tab_lr", type=float, default=1e-3)

    # residual ROI training
    ap.add_argument("--roi_epochs", type=int, default=150)
    ap.add_argument("--roi_batch", type=int, default=2)
    ap.add_argument("--roi_lr", type=float, default=1e-3)
    ap.add_argument("--delta_l2", type=float, default=0.0, help="L2 penalty on delta (encourage small corrections)")

    ap.add_argument("--val_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--start_fold", type=int, default=1)
    ap.add_argument("--skip_trained_folds", action="store_true")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--out_prefix", default="RESID_ROI_on_TAB_fixedfold")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = pd.read_csv(args.csv)
    inferred_num_cols, pre_mode = infer_tabular_num_cols(df)
    # mutate num_cols for downstream use

    target_cols = parse_list_csv(args.target)
    roi_cols = parse_list_csv(args.roi_cols)
    out_dhw = tuple(int(x) for x in parse_list_csv(args.out_dhw))

    # ---- AB-aware pre-op PTA features (safe) ----

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

        # last resort: pick any columns that look like pre_PTA_* (robust to naming)

        cand = [c for c in df.columns if c.startswith("pre_PTA_")]

        cand = [c for c in cand if any(k in c for k in ["0.5k","1k","2k","3k"])]

        if len(cand) == 0:

            raise ValueError("Cannot infer pre-op PTA columns. Expected legacy pre_PTA_* or new pre_PTA_*_A (and optionally *_B).")

        pre_cols = sorted(cand)

        pre_mode = "fallback_pre_PTA_*"


    num_cols = ["age"] + list(pre_cols)

    print(f"[INFO] Using pre-op feature mode: {pre_mode}; num_cols={num_cols}")

    # ---- end AB-aware block ----
    required = [args.id_col, args.fold_col, "side", "sex", "disease", "primary_or_recur"] + num_cols + target_cols + roi_cols
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # ensure ROI paths exist
    ok = np.ones(len(df), dtype=bool)
    for c in roi_cols:
        ok &= df[c].apply(lambda p: isinstance(p, str) and os.path.exists(p)).to_numpy()
    df = df.loc[ok].dropna(subset=required).reset_index(drop=True)
    print("[INFO] rows used:", len(df), " unique IDs:", df[args.id_col].nunique())

    folds = sorted(pd.unique(df[args.fold_col].astype(int)))
    print(f"[INFO] Using fixed folds from '{args.fold_col}': {folds}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp)

    fold_metrics = []
    oof_rows = []

    for fold in folds:
        if fold < args.start_fold:
            continue

        # checkpoints
        tab_ckpt = f"{args.out_prefix}_TAB_fold{fold}_best.pt"
        roi_ckpt = f"{args.out_prefix}_RESID_fold{fold}_best.pt"

        test_df = df[df[args.fold_col].astype(int) == fold].copy()
        train_df = df[df[args.fold_col].astype(int) != fold].copy()
        if len(test_df) == 0 or len(train_df) == 0:
            print(f"[WARN] Fold {fold}: empty split. skip.")
            continue

        # optional skip
        if args.skip_trained_folds and os.path.exists(tab_ckpt) and os.path.exists(roi_ckpt):
            print(f"[INFO] Fold {fold}: using existing checkpoints (skip training)")
        else:
            # inner split
            tr_df, va_df = train_test_split(train_df, test_size=args.val_size, random_state=args.seed)

            # fit scaler & maps on tr_df (tabular baseline)
            scaler = StandardScaler()
            scaler.fit(tr_df[num_cols].to_numpy(dtype=np.float32))
            sex_map = build_cat_map(tr_df["sex"].values)
            dis_map = build_cat_map(tr_df["disease"].values)
            pr_map  = build_cat_map(tr_df["primary_or_recur"].values)

            pin = torch.cuda.is_available()
            tr_tab = DataLoader(TabularDataset(tr_df, args.id_col, num_cols, target_cols, scaler, sex_map, dis_map, pr_map),
                                batch_size=args.tab_batch, shuffle=True, num_workers=0, pin_memory=pin)
            va_tab = DataLoader(TabularDataset(va_df, args.id_col, num_cols, target_cols, scaler, sex_map, dis_map, pr_map),
                                batch_size=args.tab_batch, shuffle=False, num_workers=0, pin_memory=pin)

            tab_model = TabularMLP(len(num_cols), len(sex_map), len(dis_map), len(pr_map), len(target_cols)).to(device)
            tab_opt = torch.optim.Adam(tab_model.parameters(), lr=args.tab_lr)
            tab_scaler_amp = get_grad_scaler(device, enabled=(use_amp and device.type == "cuda"))

            best_val = 1e18
            for ep in range(1, args.tab_epochs + 1):
                tr_loss = run_epoch_tab(tab_model, tr_tab, tab_opt, device, train=True, use_amp=use_amp, scaler=tab_scaler_amp)
                va_loss = run_epoch_tab(tab_model, va_tab, tab_opt, device, train=False, use_amp=use_amp, scaler=tab_scaler_amp)
                if ep == 1 or ep % 20 == 0 or ep == args.tab_epochs:
                    print(f"[TAB] Fold {fold} Ep {ep:03d} | train {tr_loss:.4f} | val {va_loss:.4f}")
                if va_loss < best_val:
                    best_val = va_loss
                    torch.save({
                        "model": tab_model.state_dict(),
                        "scaler_mean": scaler.mean_,
                        "scaler_scale": scaler.scale_,
                        "num_cols": num_cols,
                        "sex_map": sex_map,
                        "dis_map": dis_map,
                        "pr_map": pr_map,
                        "target_cols": target_cols,
                        "fold": fold,
                        "torch_version": torch.__version__,
                    }, tab_ckpt)

            # build ytab_map for train/val/test from best tab model
            ck = torch.load(tab_ckpt, map_location=device, weights_only=False)
            scaler = StandardScaler()
            scaler.mean_ = np.asarray(ck["scaler_mean"], dtype=np.float32)
            scaler.scale_ = np.asarray(ck["scaler_scale"], dtype=np.float32)
            sex_map, dis_map, pr_map = ck["sex_map"], ck["dis_map"], ck["pr_map"]

            tab_model.load_state_dict(ck["model"])
            tab_model.eval()

            def make_ytab_map(dfx):
                ld = DataLoader(TabularDataset(dfx, args.id_col, num_cols, target_cols, scaler, sex_map, dis_map, pr_map),
                                batch_size=args.tab_batch, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
                ids, pred, _ = predict_tab(tab_model, ld, device, use_amp=use_amp)
                return {pid: pred[i] for i, pid in enumerate(ids)}

            ytab_train = make_ytab_map(tr_df)
            ytab_val   = make_ytab_map(va_df)
            ytab_test  = make_ytab_map(test_df)

            # train residual ROI model
            tr_res = DataLoader(CTTabResidualDataset(tr_df, args.id_col, roi_cols, num_cols, target_cols,
                                                    scaler, out_dhw, sex_map, dis_map, pr_map, ytab_train),
                                batch_size=args.roi_batch, shuffle=True, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
            va_res = DataLoader(CTTabResidualDataset(va_df, args.id_col, roi_cols, num_cols, target_cols,
                                                    scaler, out_dhw, sex_map, dis_map, pr_map, ytab_val),
                                batch_size=args.roi_batch, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
            te_res = DataLoader(CTTabResidualDataset(test_df, args.id_col, roi_cols, num_cols, target_cols,
                                                    scaler, out_dhw, sex_map, dis_map, pr_map, ytab_test),
                                batch_size=args.roi_batch, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

            resid_model = ResidualROIModel(len(num_cols), len(sex_map), len(dis_map), len(pr_map),
                                           len(target_cols), len(roi_cols), roi_pool=args.roi_pool).to(device)
            resid_opt = torch.optim.Adam(resid_model.parameters(), lr=args.roi_lr)
            resid_scaler_amp = get_grad_scaler(device, enabled=(use_amp and device.type == "cuda"))

            best_val = 1e18
            for ep in range(1, args.roi_epochs + 1):
                tr_loss = run_epoch_resid(resid_model, tr_res, resid_opt, device, train=True,
                                          use_amp=use_amp, scaler=resid_scaler_amp, delta_l2=args.delta_l2)
                va_loss = run_epoch_resid(resid_model, va_res, resid_opt, device, train=False,
                                          use_amp=use_amp, scaler=resid_scaler_amp, delta_l2=args.delta_l2)
                if ep == 1 or ep % 10 == 0 or ep == args.roi_epochs:
                    print(f"[RESID] Fold {fold} Ep {ep:03d} | train {tr_loss:.4f} | val {va_loss:.4f}")
                if va_loss < best_val:
                    best_val = va_loss
                    torch.save({
                        "model": resid_model.state_dict(),
                        "tab_ckpt": tab_ckpt,
                        "scaler_mean": ck["scaler_mean"],
                        "scaler_scale": ck["scaler_scale"],
                        "num_cols": num_cols,
                        "sex_map": sex_map,
                        "dis_map": dis_map,
                        "pr_map": pr_map,
                        "target_cols": target_cols,
                        "roi_cols": roi_cols,
                        "roi_pool": args.roi_pool,
                        "out_dhw": out_dhw,
                        "fold": fold,
                        "torch_version": torch.__version__,
                    }, roi_ckpt)

        # ---- evaluation / OOF (always) ----
        # load tab and resid ckpts
        tab = torch.load(tab_ckpt, map_location=device, weights_only=False)
        roi = torch.load(roi_ckpt, map_location=device, weights_only=False)

        scaler = StandardScaler()
        scaler.mean_ = np.asarray(tab["scaler_mean"], dtype=np.float32)
        scaler.scale_ = np.asarray(tab["scaler_scale"], dtype=np.float32)
        sex_map, dis_map, pr_map = tab["sex_map"], tab["dis_map"], tab["pr_map"]

        tab_model = TabularMLP(len(num_cols), len(sex_map), len(dis_map), len(pr_map), len(target_cols)).to(device)
        tab_model.load_state_dict(tab["model"])
        tab_model.eval()

        # ytab for test fold
        te_tab = DataLoader(TabularDataset(test_df, args.id_col, num_cols, target_cols, scaler, sex_map, dis_map, pr_map),
                            batch_size=args.tab_batch, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
        ids_tab, ytab_pred, y_true_tab = predict_tab(tab_model, te_tab, device, use_amp=use_amp)
        ytab_map_test = {pid: ytab_pred[i] for i, pid in enumerate(ids_tab)}

        te_res = DataLoader(CTTabResidualDataset(test_df, args.id_col, roi_cols, num_cols, target_cols,
                                                scaler, tuple(roi.get("out_dhw", (160,192,192))),
                                                sex_map, dis_map, pr_map, ytab_map_test),
                            batch_size=args.roi_batch, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

        resid_model = ResidualROIModel(len(num_cols), len(sex_map), len(dis_map), len(pr_map),
                                       len(target_cols), len(roi_cols),
                                       roi_pool=roi.get("roi_pool", args.roi_pool)).to(device)
        resid_model.load_state_dict(roi["model"])
        resid_model.eval()

        ids, sides, y_pred, y_true = predict_resid(resid_model, te_res, device, use_amp=use_amp)

        metrics = evaluate_metrics(y_true, y_pred, target_cols)
        metrics.insert(0, "fold", fold)
        fold_metrics.append(metrics)

        oof_df = pd.DataFrame({args.id_col: ids, "side": sides, "fold": fold})
        for k, name in enumerate(target_cols):
            oof_df["true_" + name] = y_true[:, k]
            oof_df["pred_" + name] = y_pred[:, k]
            oof_df["err_" + name]  = y_pred[:, k] - y_true[:, k]
            oof_df["pred_tab_" + name] = ytab_map_test[ids[0]].shape[0]  # placeholder overwritten below
        # set pred_tab columns properly
        # align by ids order
        for k, name in enumerate(target_cols):
            oof_df["pred_tab_" + name] = np.stack([ytab_map_test[pid][k] for pid in ids], axis=0)

        oof_rows.append(oof_df)

        print(f"Fold {fold} metrics:\n{metrics}")

    # save outputs
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
