import os
import math
import random
import warnings
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

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

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---- suppress noisy warnings (incl. autocast future warnings) ----
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------
# Config
# -----------------------------
@dataclass
class CFG:
    csv_path: str = "df_final_fixed.csv"
    out_dir: str = "runs_gated_resid_fixedfold"
    seed: int = 42

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = True

    id_col: str = "id"
    fold_col: str = "fold"

    # targets (4 outputs)
    target_cols: Tuple[str, ...] = ("post_PTA_0.5k", "post_PTA_1k", "post_PTA_2k", "post_PTA_3k")

    # tab inputs (RFと同条件)
    num_cols: Tuple[str, ...] = ("age",)  # inferred at runtime; see infer_tabular_num_cols()cat_cols: Tuple[str, ...] = ("sex", "disease", "primary_or_recur")

    # ROI paths
    roi_cols: Tuple[str, ...] = ("roi_path_25", "roi_path_40", "roi_path_60")

    # ROI preprocessing (あなたの既存RESIDコードに合わせる)
    out_dhw: Tuple[int, int, int] = (96, 96, 96)   # 既存コードと同じ値に合わせてください（不明なら96^3推奨）
    hu_min: int = -1000
    hu_max: int = 2000

    # training
    batch_size_tab: int = 32
    batch_size_roi: int = 6
    num_workers: int = 2

    epochs_tab: int = 200
    epochs_roi: int = 200
    lr_tab: float = 1e-3
    lr_roi: float = 1e-4
    weight_decay: float = 1e-4
    early_stop: int = 20

    # model sizes
    tab_hidden: int = 128
    tab_drop: float = 0.15

    img_emb: int = 128
    roi_pool: str = "attn"  # "mean" or "attn"
    resid_hidden: int = 256
    gate_hidden: int = 128
    resid_drop: float = 0.10

    # gating options
    gate_use_ytab: bool = True  # ゲートがy_tabも見る（推奨）

    # regularization
    lambda_gate_l1: float = 1e-3
    lambda_delta_l2: float = 1e-4


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# ROI helpers (既存RESIDコード互換の想定)
# -----------------------------
def unify_to_right(vol_zyx: np.ndarray, side: str) -> np.ndarray:
    """
    右耳基準にそろえる（例：左耳なら左右反転）。
    既存コードの挙動に合わせたいので、最小実装としてX方向反転。
    side が "L" / "R" 以外ならそのまま。
    """
    s = str(side).upper()
    if s.startswith("L"):
        # flip X axis (last axis)
        return vol_zyx[..., ::-1].copy()
    return vol_zyx

def hu_preprocess(vol_zyx: np.ndarray, hu_min=-1000, hu_max=2000) -> np.ndarray:
    """
    HUクリップして0..1に正規化（既存RESIDの一般的実装に合わせる）
    """
    v = vol_zyx.astype(np.float32)
    v = np.clip(v, hu_min, hu_max)
    v = (v - hu_min) / (hu_max - hu_min + 1e-6)
    return v

def resize_3d_torch(vol_1zyx: torch.Tensor, out_dhw: Tuple[int, int, int]) -> torch.Tensor:
    """
    vol_1zyx: (1,Z,Y,X) -> (1,D,H,W) via trilinear
    """
    # torch expects (N,C,D,H,W)
    x = vol_1zyx.unsqueeze(0)  # (1,1,Z,Y,X)
    x = F.interpolate(x, size=out_dhw, mode="trilinear", align_corners=False)
    return x.squeeze(0)  # (1,D,H,W)


# -----------------------------
# Category maps
# -----------------------------
def build_map(series: pd.Series) -> Dict[str, int]:
    vals = series.astype(str).fillna("__NA__").unique().tolist()
    vals = sorted(vals)
    mp = {"__UNK__": 0}
    for v in vals:
        if v not in mp:
            mp[v] = len(mp)
    return mp

def map_cat(series: pd.Series, mp: Dict[str, int]) -> np.ndarray:
    return series.astype(str).fillna("__NA__").map(lambda x: mp.get(x, mp["__UNK__"])).astype(np.int64).to_numpy()


# -----------------------------
# Dataset
# -----------------------------
class TabDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cfg: CFG, scaler: StandardScaler,
                 sex_map, dis_map, pr_map):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.scaler = scaler
        self.sex_map = sex_map
        self.dis_map = dis_map
        self.pr_map  = pr_map

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = str(row[self.cfg.id_col])

        num = row[list(self.cfg.num_cols)].to_numpy(dtype=np.float32).reshape(1, -1)
        num = self.scaler.transform(num).astype(np.float32).reshape(-1)
        num_t = torch.from_numpy(num)

        sex_id = self.sex_map.get(str(row["sex"]), self.sex_map["__UNK__"])
        dis_id = self.dis_map.get(str(row["disease"]), self.dis_map["__UNK__"])
        pr_id  = self.pr_map.get(str(row["primary_or_recur"]), self.pr_map["__UNK__"])

        cats = {
            "sex": torch.tensor(sex_id, dtype=torch.long),
            "disease": torch.tensor(dis_id, dtype=torch.long),
            "primary_or_recur": torch.tensor(pr_id, dtype=torch.long),
        }

        y = row[list(self.cfg.target_cols)].to_numpy(dtype=np.float32)
        y_t = torch.from_numpy(y)

        return pid, num_t, cats, y_t


class ResidDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cfg: CFG, scaler: StandardScaler,
                 sex_map, dis_map, pr_map,
                 ytab_map: Dict[str, np.ndarray]):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.scaler = scaler
        self.sex_map = sex_map
        self.dis_map = dis_map
        self.pr_map  = pr_map
        self.ytab_map = ytab_map

    def __len__(self): return len(self.df)

    def _load_one(self, roi_path: str, side: str) -> torch.Tensor:
        vol = np.load(roi_path)  # (Z,Y,X)
        vol = unify_to_right(vol, side)
        vol = hu_preprocess(vol, self.cfg.hu_min, self.cfg.hu_max)
        vol_t = torch.from_numpy(vol).unsqueeze(0)            # (1,Z,Y,X)
        vol_t = resize_3d_torch(vol_t, self.cfg.out_dhw)      # (1,D,H,W)
        return vol_t

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = str(row[self.cfg.id_col])
        side = row["side"]

        vols = []
        for c in self.cfg.roi_cols:
            vols.append(self._load_one(str(row[c]), side))
        vols_t = torch.stack(vols, dim=0)  # (R,1,D,H,W)

        num = row[list(self.cfg.num_cols)].to_numpy(dtype=np.float32).reshape(1, -1)
        num = self.scaler.transform(num).astype(np.float32).reshape(-1)
        num_t = torch.from_numpy(num)

        sex_id = self.sex_map.get(str(row["sex"]), self.sex_map["__UNK__"])
        dis_id = self.dis_map.get(str(row["disease"]), self.dis_map["__UNK__"])
        pr_id  = self.pr_map.get(str(row["primary_or_recur"]), self.pr_map["__UNK__"])

        cats = {
            "sex": torch.tensor(sex_id, dtype=torch.long),
            "disease": torch.tensor(dis_id, dtype=torch.long),
            "primary_or_recur": torch.tensor(pr_id, dtype=torch.long),
        }

        y = row[list(self.cfg.target_cols)].to_numpy(dtype=np.float32)
        y_t = torch.from_numpy(y)

        ytab = self.ytab_map.get(pid, None)
        if ytab is None:
            raise KeyError(f"y_tab not found for id={pid}")
        ytab_t = torch.from_numpy(ytab.astype(np.float32))

        return pid, num_t, cats, vols_t, ytab_t, y_t


# -----------------------------
# Models
# -----------------------------
class TabMLP(nn.Module):
    def __init__(self, num_dim: int, n_sex: int, n_dis: int, n_pr: int,
                 hidden=128, drop=0.15, out_dim=4):
        super().__init__()
        self.sex_emb = nn.Embedding(n_sex, 8)
        self.dis_emb = nn.Embedding(n_dis, 8)
        self.pr_emb  = nn.Embedding(n_pr,  8)
        in_dim = num_dim + 8 + 8 + 8
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(True), nn.Dropout(drop),
            nn.Linear(hidden, hidden), nn.ReLU(True), nn.Dropout(drop),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, num, cats):
        x = torch.cat([num,
                       self.sex_emb(cats["sex"]),
                       self.dis_emb(cats["disease"]),
                       self.pr_emb(cats["primary_or_recur"])], dim=1)
        return self.net(x)


class Small3DCNN(nn.Module):
    def __init__(self, in_ch=1, base=16, emb_dim=128):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv3d(in_ch, base, 3, padding=1), nn.ReLU(True), nn.MaxPool3d(2))
        self.conv2 = nn.Sequential(nn.Conv3d(base, base*2, 3, padding=1), nn.ReLU(True), nn.MaxPool3d(2))
        self.conv3 = nn.Sequential(nn.Conv3d(base*2, base*4, 3, padding=1), nn.ReLU(True), nn.AdaptiveAvgPool3d(1))
        self.fc = nn.Linear(base*4, emb_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x).flatten(1)
        return self.fc(x)


class MultiROIPool(nn.Module):
    def __init__(self, emb_dim: int, mode="attn"):
        super().__init__()
        self.mode = mode
        if mode == "attn":
            self.attn = nn.Sequential(
                nn.Linear(emb_dim, emb_dim // 2),
                nn.Tanh(),
                nn.Linear(emb_dim // 2, 1)
            )
        else:
            self.attn = None

    def forward(self, e):  # (B,R,E)
        if self.mode == "mean":
            return e.mean(dim=1)
        w = self.attn(e).squeeze(-1)          # (B,R)
        a = torch.softmax(w, dim=1)           # (B,R)
        return (e * a.unsqueeze(-1)).sum(dim=1)  # (B,E)


class ResidualGatedROIModel(nn.Module):
    """
    y_hat = y_tab + gate ⊙ delta
    gate sees (ROI + clinical [+ y_tab])  ※ clinical+ROIを見る設計
    """
    def __init__(self, num_dim, n_sex, n_dis, n_pr, out_dim=4,
                 img_emb=128, roi_pool="attn",
                 resid_hidden=256, gate_hidden=128, drop=0.1,
                 gate_use_ytab=True):
        super().__init__()
        self.encoder = Small3DCNN(in_ch=1, base=16, emb_dim=img_emb)
        self.pool = MultiROIPool(img_emb, mode=roi_pool)

        self.sex_emb = nn.Embedding(n_sex, 8)
        self.dis_emb = nn.Embedding(n_dis, 8)
        self.pr_emb  = nn.Embedding(n_pr,  8)

        self.gate_use_ytab = gate_use_ytab

        base_in = img_emb + num_dim + 8 + 8 + 8
        gate_in = base_in + (out_dim if gate_use_ytab else 0)

        self.delta_head = nn.Sequential(
            nn.Linear(base_in, resid_hidden), nn.ReLU(True), nn.Dropout(drop),
            nn.Linear(resid_hidden, out_dim)
        )
        self.gate_head = nn.Sequential(
            nn.Linear(gate_in, gate_hidden), nn.ReLU(True), nn.Dropout(drop),
            nn.Linear(gate_hidden, out_dim)
        )

    def forward(self, vols, num, cats, y_tab=None):
        # vols: (B,R,1,D,H,W)
        B, R = vols.shape[0], vols.shape[1]
        x = vols.view(B*R, *vols.shape[2:])     # (B*R,1,D,H,W)
        e = self.encoder(x).view(B, R, -1)      # (B,R,E)
        img_e = self.pool(e)                   # (B,E)

        tab_e = torch.cat([num,
                           self.sex_emb(cats["sex"]),
                           self.dis_emb(cats["disease"]),
                           self.pr_emb(cats["primary_or_recur"])], dim=1)

        base = torch.cat([img_e, tab_e], dim=1)
        delta = self.delta_head(base)

        if self.gate_use_ytab:
            if y_tab is None:
                raise ValueError("gate_use_ytab=True but y_tab is None")
            gate_logits = self.gate_head(torch.cat([base, y_tab], dim=1))
        else:
            gate_logits = self.gate_head(base)

        gate = torch.sigmoid(gate_logits)
        return delta, gate


# -----------------------------
# Train / Eval
# -----------------------------
def autocast_ctx(cfg: CFG):
    return torch.amp.autocast(device_type="cuda", enabled=(cfg.use_amp and cfg.device.startswith("cuda")))

def train_tab_epoch(model, loader, opt, scaler, cfg: CFG):
    model.train()
    total = 0.0
    n = 0
    for _, num, cats, y in loader:
        num = num.to(cfg.device)
        y = y.to(cfg.device)
        cats = {k: v.to(cfg.device) for k, v in cats.items()}

        opt.zero_grad(set_to_none=True)
        with autocast_ctx(cfg):
            pred = model(num, cats)
            loss = F.smooth_l1_loss(pred, y)

        if scaler is not None and cfg.use_amp and cfg.device.startswith("cuda"):
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()

        total += loss.item() * num.size(0)
        n += num.size(0)
    return total / max(n, 1)

@torch.no_grad()
def eval_tab(model, loader, cfg: CFG):
    model.eval()
    total = 0.0
    n = 0
    ids, ys, preds = [], [], []
    for pid, num, cats, y in loader:
        num = num.to(cfg.device)
        y = y.to(cfg.device)
        cats = {k: v.to(cfg.device) for k, v in cats.items()}

        with autocast_ctx(cfg):
            pred = model(num, cats)
            loss = F.smooth_l1_loss(pred, y)

        total += loss.item() * num.size(0)
        n += num.size(0)
        ids.extend(list(pid))
        ys.append(y.detach().cpu().numpy())
        preds.append(pred.detach().cpu().numpy())

    if len(preds) == 0:
        # empty fold guard
        return float("nan"), [], np.empty((0, len(cfg.target_cols))), np.empty((0, len(cfg.target_cols)))

    ys = np.concatenate(ys, axis=0)
    preds = np.concatenate(preds, axis=0)
    return total / max(n, 1), ids, ys, preds


def train_resid_epoch(model, loader, opt, scaler, cfg: CFG):
    model.train()
    total = 0.0
    n = 0
    for _, num, cats, vols, ytab, y in loader:
        num = num.to(cfg.device)
        vols = vols.to(cfg.device)
        ytab = ytab.to(cfg.device)
        y = y.to(cfg.device)
        cats = {k: v.to(cfg.device) for k, v in cats.items()}

        opt.zero_grad(set_to_none=True)
        with autocast_ctx(cfg):
            delta, gate = model(vols, num, cats, y_tab=ytab if cfg.gate_use_ytab else None)
            yhat = ytab + gate * delta
            loss = F.smooth_l1_loss(yhat, y)

            if cfg.lambda_gate_l1 > 0:
                loss = loss + cfg.lambda_gate_l1 * gate.abs().mean()
            if cfg.lambda_delta_l2 > 0:
                loss = loss + cfg.lambda_delta_l2 * (delta ** 2).mean()

        if scaler is not None and cfg.use_amp and cfg.device.startswith("cuda"):
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()

        total += loss.item() * num.size(0)
        n += num.size(0)

    return total / max(n, 1)

@torch.no_grad()
def eval_resid(model, loader, cfg: CFG):
    model.eval()
    total = 0.0
    n = 0
    ids, ys, ytabs, yhats, deltas, gates = [], [], [], [], [], []
    for pid, num, cats, vols, ytab, y in loader:
        num = num.to(cfg.device)
        vols = vols.to(cfg.device)
        ytab = ytab.to(cfg.device)
        y = y.to(cfg.device)
        cats = {k: v.to(cfg.device) for k, v in cats.items()}

        with autocast_ctx(cfg):
            delta, gate = model(vols, num, cats, y_tab=ytab if cfg.gate_use_ytab else None)
            yhat = ytab + gate * delta
            loss = F.smooth_l1_loss(yhat, y)

        total += loss.item() * num.size(0)
        n += num.size(0)

        ids.extend(list(pid))
        ys.append(y.detach().cpu().numpy())
        ytabs.append(ytab.detach().cpu().numpy())
        yhats.append(yhat.detach().cpu().numpy())
        deltas.append(delta.detach().cpu().numpy())
        gates.append(gate.detach().cpu().numpy())

    if len(yhats) == 0:
        return float("nan"), [], None, None, None, None, None

    ys = np.concatenate(ys, axis=0)
    ytabs = np.concatenate(ytabs, axis=0)
    yhats = np.concatenate(yhats, axis=0)
    deltas = np.concatenate(deltas, axis=0)
    gates = np.concatenate(gates, axis=0)
    return total / max(n, 1), ids, ys, ytabs, yhats, deltas, gates


def parse_args():
    parser = argparse.ArgumentParser(description="K-fold fixed-fold OOF training for gated residual (ROI on tabular).")
    parser.add_argument("--csv", dest="csv_path", default=None, help="Input CSV path (must contain id, fold, ROI cols, features, targets).")
    parser.add_argument("--out_dir", default=None, help="Output directory to save fold checkpoints and OOF predictions.")
    parser.add_argument("--roi_cols", type=str, default=None, help="Comma-separated ROI npy path columns (default: auto-detect roi_path_*_sphere)")
    parser.add_argument("--epochs_roi", type=int, default=150, help="Epochs for ROI model")
    parser.add_argument("--batch_size_roi", type=int, default=6, help="Batch size for ROI model")

    parser.add_argument("--target", default=None, help="Comma-separated target column names. Example: post_PTA_0.5k_A,post_PTA_1k_A,...,post_PTA_3k_B")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--device", default=None, help="Device override: cuda or cpu.")
    parser.add_argument("--no_amp", action="store_true", help="Disable AMP even if CUDA is available.")
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = CFG()

    # ---- overrides from CLI ----
    if args.csv_path is not None:
        cfg.csv_path = args.csv_path
    if args.out_dir is not None:
        cfg.out_dir = args.out_dir
    if args.seed is not None:
        cfg.seed = args.seed
    if args.device is not None:
        cfg.device = args.device
    if args.no_amp:
        cfg.use_amp = False
    if args.target is not None:
        tcols = [t.strip() for t in args.target.split(",") if t.strip()]
        if len(tcols) == 0:
            raise ValueError("--target was provided but empty after parsing.")
        cfg.target_cols = tuple(tcols)



        if getattr(args, "roi_cols", None) is not None:
            rcols = [r.strip() for r in args.roi_cols.split(",") if r.strip()]
            if len(rcols) == 0:
                raise ValueError("--roi_cols was provided but empty after parsing.")
            cfg.roi_cols = tuple(rcols)
    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    df = pd.read_csv(cfg.csv_path)
    inferred_num_cols, pre_mode = infer_tabular_num_cols(df)
    # mutate cfg.num_cols for downstream use
    cfg.num_cols = tuple(inferred_num_cols)
    print(f"[INFO] Using pre-op feature mode: {pre_mode}; num_cols={list(cfg.num_cols)}")
    # 必須列チェック
    need = [cfg.id_col, cfg.fold_col, "side", *cfg.num_cols, *getattr(cfg,'cat_cols',tuple()), *cfg.target_cols, *cfg.roi_cols]
    for c in need:
        if c not in df.columns:
            raise KeyError(f"required column missing: {c}")

    # target欠損は除外（比較条件を揃えるならここは必須）
    df = df.dropna(subset=list(cfg.target_cols)).reset_index(drop=True)

    # --- fold をCSVに合わせてそのまま使う（完全比較条件） ---
    folds = sorted(df[cfg.fold_col].unique().tolist())
    print("[INFO] folds in CSV:", folds)
    if len(folds) < 2:
        raise ValueError("folds are not valid (need >=2). Check df_final_fixed.csv fold column.")

    oof_rows = []

    for fold in folds:
        df_tr = df[df[cfg.fold_col] != fold].reset_index(drop=True)
        df_va = df[df[cfg.fold_col] == fold].reset_index(drop=True)

        if len(df_va) == 0 or len(df_tr) == 0:
            print(f"[WARN] Fold {fold}: empty train/val. skipping.")
            continue

        # --- category maps built on train fold only ---
        sex_map = build_map(df_tr["sex"])
        dis_map = build_map(df_tr["disease"])
        pr_map  = build_map(df_tr["primary_or_recur"])

        # --- scaler (train fold) ---
        scaler = StandardScaler()
        scaler.fit(df_tr[list(cfg.num_cols)].astype(np.float32).to_numpy())

        # --- TAB datasets ---
        ds_tr_tab = TabDataset(df_tr, cfg, scaler, sex_map, dis_map, pr_map)
        ds_va_tab = TabDataset(df_va, cfg, scaler, sex_map, dis_map, pr_map)
        ld_tr_tab = DataLoader(ds_tr_tab, batch_size=cfg.batch_size_tab, shuffle=True,
                               num_workers=cfg.num_workers, pin_memory=True)
        ld_va_tab = DataLoader(ds_va_tab, batch_size=cfg.batch_size_tab, shuffle=False,
                               num_workers=cfg.num_workers, pin_memory=True)

        tab_model = TabMLP(
            num_dim=len(cfg.num_cols),
            n_sex=len(sex_map), n_dis=len(dis_map), n_pr=len(pr_map),
            hidden=cfg.tab_hidden, drop=cfg.tab_drop, out_dim=len(cfg.target_cols)
        ).to(cfg.device)

        opt_tab = torch.optim.Adam(tab_model.parameters(), lr=cfg.lr_tab, weight_decay=cfg.weight_decay)
        scaler_amp = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and cfg.device.startswith("cuda")))

        best = float("inf")
        best_state = None
        patience = 0

        for ep in range(1, cfg.epochs_tab + 1):
            tr_loss = train_tab_epoch(tab_model, ld_tr_tab, opt_tab, scaler_amp, cfg)
            va_loss, va_ids, va_y, va_pred = eval_tab(tab_model, ld_va_tab, cfg)
            if not math.isfinite(va_loss):
                # foldが空など
                break

            if va_loss < best - 1e-6:
                best = va_loss
                best_state = {k: v.detach().cpu().clone() for k, v in tab_model.state_dict().items()}
                patience = 0
            else:
                patience += 1
            if patience >= cfg.early_stop:
                break

        tab_model.load_state_dict(best_state)

        # --- build ytab_map: train uses TRAIN predictions (no leak), val uses VAL predictions (OOF) ---
        ld_tr_tab_ns = DataLoader(ds_tr_tab, batch_size=cfg.batch_size_tab, shuffle=False,
                                  num_workers=cfg.num_workers, pin_memory=True)

        _, tr_ids, _, tr_pred = eval_tab(tab_model, ld_tr_tab_ns, cfg)
        _, va_ids, _, va_pred = eval_tab(tab_model, ld_va_tab, cfg)

        ytab_map_train = {pid: tr_pred[i] for i, pid in enumerate(tr_ids)}
        ytab_map_val   = {pid: va_pred[i] for i, pid in enumerate(va_ids)}

        # --- ROI file existence filter (あなたの従来運用と同様) ---
        def roi_ok(dfx: pd.DataFrame) -> np.ndarray:
            ok = np.ones(len(dfx), dtype=bool)
            for c in cfg.roi_cols:
                ok &= dfx[c].astype(str).map(os.path.exists).to_numpy()
            return ok

        ok_tr = roi_ok(df_tr)
        ok_va = roi_ok(df_va)
        df_tr2 = df_tr[ok_tr].reset_index(drop=True)
        df_va2 = df_va[ok_va].reset_index(drop=True)

        if len(df_tr2) == 0 or len(df_va2) == 0:
            print(f"[WARN] Fold {fold}: after ROI-exist filter, empty train/val. skipping.")
            continue

        ds_tr_res = ResidDataset(df_tr2, cfg, scaler, sex_map, dis_map, pr_map, ytab_map_train)
        ds_va_res = ResidDataset(df_va2, cfg, scaler, sex_map, dis_map, pr_map, ytab_map_val)
        ld_tr_res = DataLoader(ds_tr_res, batch_size=cfg.batch_size_roi, shuffle=True,
                               num_workers=cfg.num_workers, pin_memory=True)
        ld_va_res = DataLoader(ds_va_res, batch_size=cfg.batch_size_roi, shuffle=False,
                               num_workers=cfg.num_workers, pin_memory=True)

        resid_model = ResidualGatedROIModel(
            num_dim=len(cfg.num_cols),
            n_sex=len(sex_map), n_dis=len(dis_map), n_pr=len(pr_map),
            out_dim=len(cfg.target_cols),
            img_emb=cfg.img_emb, roi_pool=cfg.roi_pool,
            resid_hidden=cfg.resid_hidden, gate_hidden=cfg.gate_hidden,
            drop=cfg.resid_drop, gate_use_ytab=cfg.gate_use_ytab
        ).to(cfg.device)

        opt_roi = torch.optim.Adam(resid_model.parameters(), lr=cfg.lr_roi, weight_decay=cfg.weight_decay)
        scaler_amp2 = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and cfg.device.startswith("cuda")))

        best = float("inf")
        best_state = None
        patience = 0

        for ep in range(1, cfg.epochs_roi + 1):
            tr_loss = train_resid_epoch(resid_model, ld_tr_res, opt_roi, scaler_amp2, cfg)
            va_loss, *_ = eval_resid(resid_model, ld_va_res, cfg)
            if not math.isfinite(va_loss):
                break

            if va_loss < best - 1e-6:
                best = va_loss
                best_state = {k: v.detach().cpu().clone() for k, v in resid_model.state_dict().items()}
                patience = 0
            else:
                patience += 1
            if patience >= cfg.early_stop:
                break

        resid_model.load_state_dict(best_state)

        # save fold model
        fold_dir = os.path.join(cfg.out_dir, f"fold{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        torch.save(best_state, os.path.join(fold_dir, "best_resid_gated.pt"))

        # OOF on this fold
        va_loss, ids, y_true, y_tab, y_hat, delta, gate = eval_resid(resid_model, ld_va_res, cfg)
        print(f"[Fold {fold}] val_n={len(ids)} (ROI-ok only) | val_loss={va_loss:.4f}")

        for i, pid in enumerate(ids):
            row = {cfg.id_col: pid, cfg.fold_col: fold}
            for k, t in enumerate(cfg.target_cols):
                row[t] = float(y_true[i, k])
                row[f"pred_{t}"] = float(y_hat[i, k])
                row[f"y_tab_{t}"] = float(y_tab[i, k])
                row[f"delta_{t}"] = float(delta[i, k])
                row[f"gate_{t}"] = float(gate[i, k])
                row[f"err_{t}"] = float(y_hat[i, k] - y_true[i, k])
                row[f"abs_err_{t}"] = float(abs(y_hat[i, k] - y_true[i, k]))
            oof_rows.append(row)

    oof = pd.DataFrame(oof_rows)
    out_oof = os.path.join(cfg.out_dir, f"GATED_RESID_ROI_on_TAB_out{len(cfg.target_cols)}_fixedfold_oof_predictions.csv")
    oof.to_csv(out_oof, index=False, encoding="utf-8-sig")
    print("\n[SAVED]", out_oof)


if __name__ == "__main__":
    main()
