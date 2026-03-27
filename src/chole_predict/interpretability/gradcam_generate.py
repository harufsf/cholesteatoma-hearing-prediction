#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_gradcam_for_case_list_fixedfold.py

Generate 3D Grad-CAM for each case in case_list.csv using the *actual model used for that case*,
i.e., the model corresponding to the patient's fixed fold (fold1..fold5).

It imports your CANON_v4 training scripts to match model definitions:
- kfold_train_residual_roi_on_tabular_fixedfold_abaware_CANON_v4.py  (RESID)
- kfold_train_gated_residual_roi_on_tabular_fixedfold_out8_cli_abaware_CANON_v4.py  (GATED)

TAB has no image -> no Grad-CAM (we use TAB predictions from per_patient_results.csv as y_tab).

Output:
  out_dir/
    fold<k>/
      <id>/
        resid_cam_yhat.npy
        resid_overlay_yhat_ax.png / cor / sag
        (optional) resid_cam_delta.npy + overlays if --also_save_resid_delta_cam
        gated_cam.npy
        gate.npy
        gated_overlay_ax.png / cor / sag
        meta.json

Run from chole_predict/ (same folder where the training scripts exist).

Example:
python make_gradcam_for_case_list_fixedfold.py ^
  --case_list_csv experiments\20260218_143700_CroppedROI_out8_AB\case_list.csv ^
  --base_csv df_fixed_with_cropped_roi.csv ^
  --per_patient_csv experiments\20260218_143700_CroppedROI_out8_AB\per_patient_results.csv ^
  --exp_root experiments\20260218_143700_CroppedROI_out8_AB ^
  --roi_col roi_path_40_sphere ^
  --out_dir GradCAM_cases_fixedfold ^
  --out_dhw 96,96,96 ^
  --objective gated_term_ac_mean ^
  --also_save_resid_delta_cam
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F


# -------------------------
# PyTorch 2.6+ safe loading helper
# -------------------------
def torch_load_ckpt(path, map_location):
    """
    PyTorch 2.6 changed torch.load default weights_only=True.
    Your checkpoints may contain numpy objects / metadata -> require weights_only=False.
    We explicitly set weights_only=False when supported.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # older torch without weights_only kwarg
        return torch.load(path, map_location=map_location)
# Import model definitions to guarantee architectural match
import kfold_train_residual_roi_on_tabular_fixedfold_abaware_CANON_v4 as resid_train
import kfold_train_gated_residual_roi_on_tabular_fixedfold_out8_cli_abaware_CANON_v4 as gated_train


def normalize_side(s):
    s = str(s).strip().upper()
    if s in ["R", "RIGHT", "右"]:
        return "R"
    if s in ["L", "LEFT", "左"]:
        return "L"
    return None


def unify_to_right(vol_zyx: np.ndarray, side: str) -> np.ndarray:
    sd = normalize_side(side)
    if sd is None:
        raise ValueError(f"side must be R/L (got {side})")
    if sd == "L":
        return vol_zyx[..., ::-1].copy()
    return vol_zyx


def hu_preprocess(vol_zyx: np.ndarray, hu_min=-1000, hu_max=2000) -> np.ndarray:
    v = vol_zyx.astype(np.float32)
    v = np.clip(v, hu_min, hu_max)
    v = (v - hu_min) / (hu_max - hu_min + 1e-6)
    return v


def resize_3d(vol_zyx: np.ndarray, out_dhw) -> np.ndarray:
    t = torch.from_numpy(vol_zyx).float()[None, None]  # (1,1,Z,Y,X)
    t = F.interpolate(t, size=out_dhw, mode="trilinear", align_corners=False)
    return t[0, 0].cpu().numpy()

def upsample_cam_to(vol_zyx_shape, cam_dhw: np.ndarray) -> np.ndarray:
    """
    Upsample CAM (D,H,W) to match original ROI shape (Z,Y,X) using trilinear interpolation.
    Returns (Z,Y,X) float32 in [0,1].
    """
    Z, Y, X = vol_zyx_shape
    t = torch.from_numpy(cam_dhw).float()[None, None]  # (1,1,D,H,W)
    t = F.interpolate(t, size=(Z, Y, X), mode="trilinear", align_corners=False)
    cam = t[0, 0].cpu().numpy().astype(np.float32)
    cam = np.clip(cam, 0.0, 1.0)
    return cam



class GradCAM3D:
    # Grad-CAM for 3D conv nets without module backward hooks.
    # We capture activations in a forward hook and read gradients from activation.grad after backward.
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.h1 = target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out
        try:
            self.activations.retain_grad()
        except Exception:
            pass

    def remove(self):
        self.h1.remove()

    def __call__(self, score):
        self.model.zero_grad(set_to_none=True)
        score.backward(retain_graph=True)

        A = self.activations
        if A is None:
            raise RuntimeError('Activations not captured. Check target_layer selection.')
        G = A.grad
        if G is None:
            raise RuntimeError('Activation gradients not available. Ensure backward ran and retain_grad worked.')

        w = G.mean(dim=(2, 3, 4), keepdim=True)
        cam = (w * A).sum(dim=1)  # (B,D,H,W)

        # Post-process
        if getattr(self, "cam_mode", "relu") == "relu":
            cam = F.relu(cam)
            cam_min = cam.amin(dim=(1, 2, 3), keepdim=True)
            cam_max = cam.amax(dim=(1, 2, 3), keepdim=True)
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-6)
        elif self.cam_mode == "abs":
            cam = cam.abs()
            cam_min = cam.amin(dim=(1, 2, 3), keepdim=True)
            cam_max = cam.amax(dim=(1, 2, 3), keepdim=True)
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-6)
        else:  # signed
            denom = cam.abs().amax(dim=(1, 2, 3), keepdim=True) + 1e-6
            cam = cam / denom
            cam = (cam + 1.0) * 0.5  # [-1,1] -> [0,1]

        return cam.detach().cpu().numpy()[0]



def disable_inplace_relu(model: torch.nn.Module):
    # Set all nn.ReLU(inplace=True) to inplace=False to avoid autograd hook issues.
    for m in model.modules():
        if isinstance(m, torch.nn.ReLU) and getattr(m, 'inplace', False):
            m.inplace = False


def get_layer_by_name(model: torch.nn.Module, name: str):
    if not name:
        return None
    for n, m in model.named_modules():
        if n == name:
            return m
    raise KeyError(f"Layer name not found: {name}")


def list_named_modules(model, pattern: str = "", max_items: int = 500):
    """Print model.named_modules() filtered by substring pattern."""
    items = []
    for name, mod in model.named_modules():
        if name == "":
            continue
        if pattern and (pattern not in name):
            continue
        items.append((name, mod.__class__.__name__))
        if len(items) >= max_items:
            break
    if not items:
        print(f"[LIST_LAYERS] No modules matched pattern='{pattern}'")
        return
    print(f"[LIST_LAYERS] matched={len(items)} (showing up to {max_items}) pattern='{pattern}'")
    for name, cls in items:
        print(f"  {name} ({cls})")

def find_last_conv3d(model: torch.nn.Module):
    last = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv3d):
            last = m
    if last is None:
        raise RuntimeError("No Conv3d layer found in model. Cannot compute Grad-CAM.")
    return last



def build_cam_mask(vol01_zyx: np.ndarray,
                   hu_min: float,
                   hu_max: float,
                   mode: str = "nonzero",
                   eps: float = 1e-3,
                   hu_low: float | None = None,
                   hu_high: float | None = None) -> np.ndarray:
    """Return boolean mask (Z,Y,X) where CAM should be considered for normalization/overlay.

    vol01_zyx: HU-normalized volume in [0,1] (after hu_preprocess).
    mode:
      - none: all True
      - nonzero: vol01 > eps  (robustly removes padding=0)
      - air / soft / bone: based on reconstructed HU (approx)
      - range: use hu_low/hu_high
    """
    if mode is None:
        mode = "nonzero"
    mode = str(mode).lower()
    if mode in ("none", "all"):
        return np.ones_like(vol01_zyx, dtype=bool)

    base = vol01_zyx > float(eps)

    hu = vol01_zyx * (hu_max - hu_min) + hu_min

    if mode == "nonzero":
        return base
    if mode == "air":
        # typically: air / aeration spaces (-1200 .. -300 HU)
        lo = -1200.0 if hu_low is None else float(hu_low)
        hi = -300.0 if hu_high is None else float(hu_high)
        return base & (hu >= lo) & (hu <= hi)
    if mode == "soft":
        lo = -300.0 if hu_low is None else float(hu_low)
        hi = 300.0 if hu_high is None else float(hu_high)
        return base & (hu >= lo) & (hu <= hi)
    if mode == "bone":
        lo = 300.0 if hu_low is None else float(hu_low)
        return base & (hu >= lo)

    if mode in ("range", "hu_range"):
        if hu_low is None or hu_high is None:
            # fall back to nonzero
            return base
        return base & (hu >= float(hu_low)) & (hu <= float(hu_high))

    # fallback
    return base


def normalize_cam_for_viz(cam_zyx: np.ndarray,
                          mask_zyx: np.ndarray | None = None,
                          clip_percentile: float = 99.0,
                          border_crop: int = 0) -> np.ndarray:
    """Normalize CAM to [0,1] using robust percentile scaling inside mask."""
    cam = np.asarray(cam_zyx, dtype=np.float32)

    if border_crop and border_crop > 0:
        b = int(border_crop)
        cam[:b, :, :] = 0
        cam[-b:, :, :] = 0
        cam[:, :b, :] = 0
        cam[:, -b:, :] = 0
        cam[:, :, :b] = 0
        cam[:, :, -b:] = 0

    if mask_zyx is None:
        m = np.ones_like(cam, dtype=bool)
    else:
        m = mask_zyx.astype(bool)
        if m.shape != cam.shape:
            m = np.ones_like(cam, dtype=bool)

    vals = cam[m]
    if vals.size == 0:
        # fallback
        vmin, vmax = float(cam.min()), float(cam.max())
    else:
        vmax = float(np.percentile(vals, float(clip_percentile)))
        vmin = float(vals.min())

    if vmax <= vmin + 1e-8:
        return np.zeros_like(cam, dtype=np.float32)

    cam = np.clip((cam - vmin) / (vmax - vmin), 0.0, 1.0)
    cam[~m] = 0.0
    return cam


def save_overlay_png(vol_for_show_zyx: np.ndarray,
                     cam_for_show_zyx: np.ndarray,
                     out_prefix: str,
                     axial_only: bool = False,
                     *,
                     hu_min: float = -1024.0,
                     hu_max: float = 3071.0,
                     axial_slice_mode: str = "mid",
                     axial_center_frac: float = 0.35,
                     axial_air_hu_thr: float = None,
                     axial_z_window: int = -1,
                     axial_score_method: str = "sum",
                     axial_top_pct: float = 0.01,
                     axial_min_count: int = 0,
                     mask_mode: str = "nonzero",
                     mask_eps: float = 1e-3,
                     mask_hu_low: float | None = None,
                     mask_hu_high: float | None = None,
                     cam_clip_percentile: float = 99.0,
                     cam_border_crop: int = 0):
    """Save overlay images.

    vol_for_show_zyx, cam_for_show_zyx: (Z,Y,X) arrays at the SAME resolution.
      - vol_for_show_zyx is expected to be HU-normalized [0,1] (after hu_preprocess)
      - cam_for_show_zyx can be any real-valued CAM map (signed/abs/relu). We'll normalize for visualization.

    Key idea:
      1) Build a robust mask (default removes padding==0)
      2) Normalize CAM inside mask by percentile scaling (reduces edge artefacts and makes local signals visible)
      3) Optionally crop borders (kills the typical "padding ring" CAM)
    """
    import matplotlib.pyplot as plt

    Z, Y, X = cam_for_show_zyx.shape
    # default view centers
    z, y, x = Z // 2, Y // 2, X // 2

    # base mask used both for CAM normalization and for autoslice scoring
    cam_mask_zyx = build_cam_mask(
        vol_for_show_zyx,
        hu_min=hu_min,
        hu_max=hu_max,
        mode=mask_mode,
        eps=mask_eps,
        hu_low=mask_hu_low,
        hu_high=mask_hu_high,
    )

    # optional: choose an axial slice where CAM is strongest near the ROI center (or within mask)
    if axial_only and axial_slice_mode != "mid":
        cam_abs = np.abs(cam_for_show_zyx)
        # base region mask per-slice
        score_mask_zyx = cam_mask_zyx.copy()
        if axial_slice_mode == "maxcam_center":
            # center window (fraction of H/W)
            frac = float(axial_center_frac)
            frac = max(0.05, min(1.0, frac))
            y0 = int(round((1.0 - frac) * 0.5 * Y))
            y1 = int(round((1.0 + frac) * 0.5 * Y))
            x0 = int(round((1.0 - frac) * 0.5 * X))
            x1 = int(round((1.0 + frac) * 0.5 * X))
            center_mask = np.zeros((Y, X), dtype=bool)
            center_mask[y0:y1, x0:x1] = True
            score_mask_zyx &= center_mask[None, :, :]
        elif axial_slice_mode == "maxcam_mask":
            # already masked by cam_mask_zyx
            pass
        else:
            # unknown mode -> fallback
            pass

        if axial_air_hu_thr is not None:
            # reconstruct HU from normalized volume in [0,1]
            hu = vol_for_show_zyx * (hu_max - hu_min) + hu_min
            air_mask = hu < float(axial_air_hu_thr)
            score_mask_zyx &= air_mask

        # score each z by CAM intensity within score mask
        # Optionally restrict search to mid-z ± axial_z_window slices to avoid obviously off slices.
        z_mid = Z // 2
        if axial_z_window is not None and int(axial_z_window) > 0:
            K = int(axial_z_window)
            z_lo = max(0, z_mid - K)
            z_hi = min(Z, z_mid + K + 1)
            z_range = range(z_lo, z_hi)
        else:
            z_range = range(Z)

        def _score_vals(vals: np.ndarray) -> float:
            if vals.size == 0:
                return 0.0
            if axial_min_count and vals.size < int(axial_min_count):
                return 0.0
            mth = str(axial_score_method).lower()
            if mth in ("sum", "l1", "abs_sum"):
                return float(vals.sum())
            if mth in ("mean", "avg"):
                return float(vals.mean())
            if mth in ("top_pct_mean", "topmean", "top_p"):
                pct = float(axial_top_pct) if axial_top_pct is not None else 0.01
                pct = max(1e-4, min(1.0, pct))
                k = max(1, int(round(pct * vals.size)))
                # take top-k values (vals are already non-negative cam_abs)
                part = np.partition(vals, -k)[-k:]
                return float(part.mean())
            # fallback
            return float(vals.sum())

        scores = np.zeros(Z, dtype=np.float32)
        for zz in z_range:
            msk = score_mask_zyx[zz]
            if np.any(msk):
                scores[zz] = _score_vals(cam_abs[zz][msk])
            else:
                scores[zz] = 0.0

        if float(scores.max()) > 0:
            z = int(scores.argmax())


    mask = cam_mask_zyx

    cam_viz = normalize_cam_for_viz(cam_for_show_zyx, mask_zyx=mask,
                                    clip_percentile=cam_clip_percentile,
                                    border_crop=cam_border_crop)

    # slices
    ax_title = "axial (mid-z)" if axial_slice_mode == "mid" else f"axial (auto-z: {axial_slice_mode}, z={z})"
    slices = [(ax_title, vol_for_show_zyx[z, :, :], cam_viz[z, :, :], f"{out_prefix}_ax.png")]
    if not axial_only:
        slices += [
            ("coronal (mid-y)", vol_for_show_zyx[:, y, :], cam_viz[:, y, :], f"{out_prefix}_cor.png"),
            ("sagittal (mid-x)", vol_for_show_zyx[:, :, x], cam_viz[:, :, x], f"{out_prefix}_sag.png"),
        ]

    for title, v2d, c2d, out_png in slices:
        plt.figure(figsize=(6, 6))
        plt.imshow(v2d, cmap="gray", vmin=0, vmax=1)
        # alpha proportional to cam intensity, but capped
        alpha = np.clip(c2d, 0, 1) * 0.65
        plt.imshow(c2d, cmap="jet", alpha=alpha, vmin=0, vmax=1)
        plt.axis("off")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_png, dpi=220)
        plt.close()

def build_ytab_map(per_patient_csv: str, id_col="id"):
    df = pd.read_csv(per_patient_csv)
    df[id_col] = df[id_col].astype(str)

    targets = [
        "post_PTA_0.5k_A", "post_PTA_1k_A", "post_PTA_2k_A", "post_PTA_3k_A",
        "post_PTA_0.5k_B", "post_PTA_1k_B", "post_PTA_2k_B", "post_PTA_3k_B",
    ]

    prefixes = ["pred_tab_", "pred_TAB_", "pred_tab", "pred_TAB", "pred_"]

    def find_col(target):
        for p in prefixes:
            c = f"{p}{target}"
            if c in df.columns:
                return c
        for c in df.columns:
            cl = c.lower()
            if "tab" in cl and target.lower() in cl:
                return c
        return None

    tab_cols = []
    for t in targets:
        c = find_col(t)
        if c is None:
            raise KeyError(f"Cannot find TAB prediction column for target={t} in {per_patient_csv}")
        tab_cols.append(c)

    ytab = {}
    for _, r in df.iterrows():
        pid = str(r[id_col])
        ytab[pid] = r[tab_cols].to_numpy(dtype=np.float32)
    return ytab, targets


def extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        for k in ["model", "state_dict"]:
            if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
                return ckpt_obj[k]
        return ckpt_obj
    return ckpt_obj


def build_maps_and_numcols(base_df, ckpt, which: str):
    if which == "resid":
        num_cols, _ = resid_train.infer_tabular_num_cols(base_df)
        sex_map = ckpt.get("sex_map", resid_train.build_cat_map(base_df["sex"].astype(str).values))
        dis_map = ckpt.get("dis_map", resid_train.build_cat_map(base_df["disease"].astype(str).values))
        pr_map  = ckpt.get("pr_map",  resid_train.build_cat_map(base_df["primary_or_recur"].astype(str).values))
        maps = {"sex": sex_map, "disease": dis_map, "primary_or_recur": pr_map}
    else:
        num_cols, _ = gated_train.infer_tabular_num_cols(base_df)
        sex_map = ckpt.get("sex_map", gated_train.build_map(base_df["sex"]))
        dis_map = ckpt.get("dis_map", gated_train.build_map(base_df["disease"]))
        pr_map  = ckpt.get("pr_map",  gated_train.build_map(base_df["primary_or_recur"]))
        maps = {"sex": sex_map, "disease": dis_map, "primary_or_recur": pr_map}

    scaler = None
    if isinstance(ckpt, dict) and "scaler_mean" in ckpt and "scaler_scale" in ckpt:
        scaler = {"mean": np.asarray(ckpt["scaler_mean"], dtype=np.float32),
                  "scale": np.asarray(ckpt["scaler_scale"], dtype=np.float32)}
    return num_cols, maps, scaler


def make_num_tensor(row, num_cols, scaler, device):
    x = row[list(num_cols)].to_numpy(dtype=np.float32)
    if scaler is not None:
        x = (x - scaler["mean"]) / (scaler["scale"] + 1e-12)
    return torch.from_numpy(x.reshape(1, -1)).to(device)


def make_cat_tensors(row, maps, device, unk_key="__UNK__"):
    out = {}
    for k, mp in maps.items():
        val = str(row[k])
        idx = mp.get(val, mp.get(unk_key, 0)) if isinstance(mp, dict) else 0
        out[k] = torch.tensor([idx], dtype=torch.long, device=device)
    return out


class FoldModelCache:
    def __init__(self, base_df, targets, device, resid_ckpt_paths, gated_ckpt_paths):
        self.base_df = base_df
        self.targets = targets
        self.device = device
        self.resid_ckpt_paths = resid_ckpt_paths
        self.gated_ckpt_paths = gated_ckpt_paths
        self._cache = {}

    def get(self, fold: int, cam_mode: str = 'relu', target_layer_name: str = '', target_layer_name_resid: str = '', target_layer_name_gated: str = ''):
        fold = int(fold)
        if fold in self._cache:
            return self._cache[fold]

        resid_ckpt = torch_load_ckpt(self.resid_ckpt_paths[fold], map_location=self.device)
        resid_sd = extract_state_dict(resid_ckpt)
        resid_num_cols, resid_maps, resid_scaler = build_maps_and_numcols(self.base_df, resid_ckpt, "resid")

        resid_model = resid_train.ResidualROIModel(
            num_dim=len(resid_num_cols),
            n_sex=len(resid_maps["sex"]),
            n_dis=len(resid_maps["disease"]),
            n_pr=len(resid_maps["primary_or_recur"]),
            out_dim=len(self.targets),
            n_rois=1,
            roi_pool="attn",
            img_emb=128,
            hidden=256,
        ).to(self.device)
        resid_model.load_state_dict(resid_sd, strict=False)
        resid_model.eval()
        disable_inplace_relu(resid_model)
        tl_name = target_layer_name_resid or target_layer_name
        tl = get_layer_by_name(resid_model, tl_name) if tl_name else None
        resid_cam = GradCAM3D(resid_model, tl if tl is not None else find_last_conv3d(resid_model))
        resid_cam.cam_mode = cam_mode

        gated_ckpt = torch_load_ckpt(self.gated_ckpt_paths[fold], map_location=self.device)
        gated_sd = extract_state_dict(gated_ckpt)
        gated_num_cols, gated_maps, gated_scaler = build_maps_and_numcols(self.base_df, gated_ckpt, "gated")

        gated_model = gated_train.ResidualGatedROIModel(
            num_dim=len(gated_num_cols),
            n_sex=len(gated_maps["sex"]),
            n_dis=len(gated_maps["disease"]),
            n_pr=len(gated_maps["primary_or_recur"]),
            out_dim=len(self.targets),
            img_emb=128,
            roi_pool="attn",
            resid_hidden=256,
            gate_hidden=128,
            drop=0.10,
            gate_use_ytab=True,
        ).to(self.device)
        gated_model.load_state_dict(gated_sd, strict=False)
        gated_model.eval()
        disable_inplace_relu(gated_model)
        tlg_name = target_layer_name_gated or target_layer_name
        tlg = get_layer_by_name(gated_model, tlg_name) if tlg_name else None
        gated_cam = GradCAM3D(gated_model, tlg if tlg is not None else find_last_conv3d(gated_model))
        gated_cam.cam_mode = cam_mode

        pack = {
            "resid": {"model": resid_model, "cam": resid_cam, "num_cols": resid_num_cols, "maps": resid_maps, "scaler": resid_scaler},
            "gated": {"model": gated_model, "cam": gated_cam, "num_cols": gated_num_cols, "maps": gated_maps, "scaler": gated_scaler},
        }
        self._cache[fold] = pack
        return pack


def find_ckpt_resid(exp_root: Path, fold: int) -> Path:
    fold = int(fold)
    cands = []
    p1 = exp_root / "RESID" / f"fold{fold}"
    if p1.exists():
        cands += sorted(p1.glob("best*.pt"))
        cands += sorted(p1.glob("*best*.pt"))
    cands += sorted(exp_root.glob(f"RESID_TAB_fold{fold}*best*.pt"))
    cands += sorted(exp_root.glob(f"*RESID*fold{fold}*best*.pt"))
    if not cands:
        raise FileNotFoundError(f"Could not find RESID ckpt for fold{fold} under {exp_root}")
    return cands[0]


def find_ckpt_gated(exp_root: Path, fold: int) -> Path:
    fold = int(fold)
    p = exp_root / "GATED" / f"fold{fold}"
    if p.exists():
        brg = p / "best_resid_gated.pt"
        if brg.exists():
            return brg
        cands = sorted(p.glob("best*.pt")) + sorted(p.glob("*best*.pt"))
        if cands:
            return cands[0]
    cands = sorted(exp_root.glob(f"*GATED*fold{fold}*best*.pt"))
    if not cands:
        raise FileNotFoundError(f"Could not find GATED ckpt for fold{fold} under {exp_root}")
    return cands[0]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case_list_csv", required=True)
    ap.add_argument("--base_csv", required=True)
    ap.add_argument("--per_patient_csv", required=True)
    ap.add_argument("--exp_root", required=True)

    ap.add_argument("--roi_col", default="roi_path_40_sphere")
    ap.add_argument("--id_col", default="id")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--out_dir", default="GradCAM_cases_fixedfold")
    ap.add_argument("--out_dhw", default="96,96,96")
    ap.add_argument("--hu_min", type=int, default=-1000)
    ap.add_argument("--hu_max", type=int, default=2000)

    ap.add_argument(
        "--objective",
        type=str,
        default="gated_term_ac_mean",
        choices=[
            # canonical names
            "yhat_ac_mean",
            "delta_ac_mean",
            "gate_ac_mean",
            "gated_term_ac_mean",
            # aliases (more convenient)
            "yhat",
            "delta",
            "gate",
            "gated_term",
            "delta_yhat_from_gate",
        ],
        help=(
            "Objective used to backprop for Grad-CAM (computed on the AC band mean). "
            "Aliases: yhat->yhat_ac_mean, delta->delta_ac_mean, gate->gate_ac_mean, "
            "gated_term/delta_yhat_from_gate->gated_term_ac_mean."
        ),
    )
    ap.add_argument("--also_save_resid_delta_cam", action="store_true")
    ap.add_argument("--viz_mode", default="original",
                    choices=["model", "original"],
                    help="Visualization resolution. 'model' shows out_dhw. 'original' overlays upsampled CAM on original ROI resolution.")
    ap.add_argument("--axial_only", action="store_true",
                    help="If set, save axial only (skip coronal/sagittal) to reduce clutter.")
    ap.add_argument("--cam_mode", default="relu",
                    choices=["relu", "abs", "signed"],
                    help="How to post-process CAM. relu=positive only, abs=|contribution|, signed=keep sign and normalize to [-1,1].")
    
    ap.add_argument("--cam_clip_percentile", type=float, default=99.0,
                        help="Normalize CAM using this percentile inside mask (helps suppress edge artefacts).")
    ap.add_argument("--cam_border_crop", type=int, default=0,
                        help="Zero-out CAM within this many voxels from each border before normalization (kills padding-ring artefacts).")
    ap.add_argument("--cam_mask_mode", type=str, default="nonzero",
                        choices=["none", "nonzero", "air", "soft", "bone", "range"],
                        help="Mask used for CAM normalization/overlay. nonzero=vol>eps (removes padding). air/soft/bone use HU thresholds.")
    ap.add_argument("--cam_mask_eps", type=float, default=1e-3,
                        help="Threshold for nonzero mask (on HU-normalized [0,1] volume).")
    ap.add_argument("--cam_mask_hu_low", type=float, default=None,
                        help="Lower HU threshold for cam_mask_mode=range/air/soft (optional override).")
    ap.add_argument("--cam_mask_hu_high", type=float, default=None,
                        help="Upper HU threshold for cam_mask_mode=range/air/soft (optional override).")
    ap.add_argument("--target_layer_name", default="",
                    help="Optional: explicitly select conv layer by name for CAM (e.g., 'encoder.conv2.0'). If empty, auto-pick last Conv3d.")
    ap.add_argument("--target_layer_name_resid", default="",
                    help="Optional: Conv layer name for RESID CAM (overrides --target_layer_name for RESID only), e.g., 'encoder.net.8'.")
    ap.add_argument("--target_layer_name_gated", default="",
                    help="Optional: Conv layer name for GATED CAM (overrides --target_layer_name for GATED only), e.g., 'encoder.conv2.0'.")

    # Debug / introspection: list model layers
    ap.add_argument("--list_layers", action="store_true",
                    help="If set, list named_modules of the selected model and exit.")
    ap.add_argument("--list_layers_fold", type=int, default=1,
                    help="Which fold to load when --list_layers is set.")
    ap.add_argument("--list_layers_model", type=str, default="gated",
                    choices=["gated", "resid"],
                    help="Which model to inspect when --list_layers is set.")
    ap.add_argument("--list_layers_pattern", type=str, default="",
                    help="Substring filter for layer names when listing layers (e.g., 'encoder').")
    ap.add_argument("--list_layers_max", type=int, default=500,
                    help="Maximum number of layers to print.")
    

    # Axial slice selection for visualization
    ap.add_argument(
        "--axial_slice_mode",
        choices=["mid", "maxcam_center", "maxcam_mask"],
        default="mid",
        help="Which axial slice to visualize: 'mid' uses mid-z; 'maxcam_center' selects z with maximum |CAM| within a central window; 'maxcam_mask' selects z with maximum |CAM| within the CAM mask.",
    )
    ap.add_argument(
        "--axial_center_frac",
        type=float,
        default=0.35,
        help="Central window size (fraction of H/W) used by --axial_slice_mode=maxcam_center. 0.35 means center 35%% of the slice.",
    )
    ap.add_argument(
        "--axial_air_hu_thr",
        type=float,
        default=None,
        help="Optional HU threshold to further restrict the slice-scoring region to 'air-like' voxels (e.g., -300). Uses HU reconstructed from --hu_min/--hu_max.",
    )

    ap.add_argument(
        "--axial_z_window",
        type=int,
        default=-1,
        help="If >0, restrict auto-z search to mid-z ± this many slices (safety window). Example: 15.",
    )
    ap.add_argument(
        "--axial_score_method",
        type=str,
        default="sum",
        choices=["sum", "mean", "top_pct_mean"],
        help="How to score each axial slice when selecting auto-z: sum/mean/top_pct_mean of |CAM| within the scoring mask.",
    )
    ap.add_argument(
        "--axial_top_pct",
        type=float,
        default=0.01,
        help="For axial_score_method=top_pct_mean: fraction of top voxels to average (e.g., 0.01 = top 1%).",
    )
    ap.add_argument(
        "--axial_min_count",
        type=int,
        default=0,
        help="Minimum number of voxels in the scoring mask for a slice to be considered (0 disables).",
    )
    return ap.parse_args()


def main():
    args = parse_args()

    # Normalize objective aliases to canonical names
    _obj_alias = {
        "yhat": "yhat_ac_mean",
        "delta": "delta_ac_mean",
        "gate": "gate_ac_mean",
        "gated_term": "gated_term_ac_mean",
        "delta_yhat_from_gate": "gated_term_ac_mean",
    }
    if args.objective in _obj_alias:
        print(f"[INFO] objective alias '{args.objective}' -> '{_obj_alias[args.objective]}'")
        args.objective = _obj_alias[args.objective]
    exp_root = Path(args.exp_root)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    out_dhw = tuple(int(x) for x in args.out_dhw.split(","))

    case_df = pd.read_csv(args.case_list_csv)
    base_df = pd.read_csv(args.base_csv)

    case_df[args.id_col] = case_df[args.id_col].astype(str)
    base_df[args.id_col] = base_df[args.id_col].astype(str)

    if "fold" not in base_df.columns:
        raise ValueError("base_csv must contain 'fold' column (fixedfold).")
    if "side" not in base_df.columns:
        raise ValueError("base_csv must contain 'side' column.")

    ytab_map, targets = build_ytab_map(args.per_patient_csv, id_col=args.id_col)

    ac_targets = ["post_PTA_0.5k_A", "post_PTA_1k_A", "post_PTA_2k_A", "post_PTA_3k_A"]
    ac_idx = [targets.index(t) for t in ac_targets]

    device = torch.device(args.device)

    # Resolve fold assignment robustly: case_list may already contain fold; base_csv provides fallback
    merged = case_df.copy()
    if "fold" not in merged.columns:
        merged = merged.merge(base_df[[args.id_col, "fold"]], on=args.id_col, how="left")
    else:
        merged = merged.merge(base_df[[args.id_col, "fold"]], on=args.id_col, how="left", suffixes=("", "_base"))
        if "fold_base" in merged.columns:
            merged["fold"] = merged["fold"].fillna(merged["fold_base"])
            merged = merged.drop(columns=["fold_base"])
    if "fold" not in merged.columns or merged["fold"].isna().any():
        missing = merged.loc[merged.get("fold").isna() if "fold" in merged.columns else slice(None), args.id_col].tolist()
        raise ValueError(f"Some ids missing fold (case_list/base_csv): {missing[:20]} ...")
    folds = sorted(set(int(x) for x in merged["fold"].unique()))
    resid_ckpt_paths = {}
    gated_ckpt_paths = {}
    for f in folds:
        resid_ckpt_paths[f] = find_ckpt_resid(exp_root, f)
        gated_ckpt_paths[f] = find_ckpt_gated(exp_root, f)
        print(f"[CKPT] fold{f} RESID={resid_ckpt_paths[f]}")
        print(f"[CKPT] fold{f} GATED={gated_ckpt_paths[f]}")

    cache = FoldModelCache(base_df, targets, device, resid_ckpt_paths, gated_ckpt_paths)

    # Optional: list model layers and exit (useful to pick --target_layer_name_* correctly)
    if args.list_layers:
        f = int(args.list_layers_fold)
        pack = cache.get(
            f,
            cam_mode=args.cam_mode,
            target_layer_name=args.target_layer_name,
            target_layer_name_resid=getattr(args, "target_layer_name_resid", ""),
            target_layer_name_gated=getattr(args, "target_layer_name_gated", ""),
        )
        model = pack["gated"]["model"] if args.list_layers_model == "gated" else pack["resid"]["model"]
        print(f"[LIST_LAYERS] fold={f} model={args.list_layers_model}")
        list_named_modules(model, pattern=args.list_layers_pattern, max_items=int(args.list_layers_max))
        return

    for _, rr in merged.iterrows():
        pid = str(rr[args.id_col])
        fold = int(rr["fold"])

        bdf = base_df.loc[base_df[args.id_col] == pid]
        if len(bdf) == 0:
            print(f"[WARN] id not found in base_csv: {pid}")
            continue
        b = bdf.iloc[0]

        if pid not in ytab_map:
            print(f"[WARN] id not found in per_patient_csv (ytab): {pid}")
            continue

        roi_path = str(b[args.roi_col])
        side = b["side"]

        out_case = out_root / f"fold{fold}" / pid
        out_case.mkdir(parents=True, exist_ok=True)

        vol = np.load(roi_path)          # (Z,Y,X)
        vol = unify_to_right(vol, side)
        vol = hu_preprocess(vol, args.hu_min, args.hu_max)   # (Z,Y,X) in [0,1]
        vol_orig_zyx = vol  # keep for visualization
        vol_dhw = resize_3d(vol, out_dhw)  # model input (D,H,W)
        vols = torch.from_numpy(vol_dhw).float()[None, None, None].to(device)  # (1,1,1,D,H,W)

        pack = cache.get(fold, cam_mode=args.cam_mode, target_layer_name=args.target_layer_name, target_layer_name_resid=getattr(args,'target_layer_name_resid',''), target_layer_name_gated=getattr(args,'target_layer_name_gated',''))
        resid = pack["resid"]
        gated = pack["gated"]

        ytab = torch.from_numpy(ytab_map[pid].reshape(1, -1)).to(device)

        num_res = make_num_tensor(b, resid["num_cols"], resid["scaler"], device)
        cats_res = make_cat_tensors(b, resid["maps"], device)
        sex_id, dis_id, pr_id = cats_res["sex"], cats_res["disease"], cats_res["primary_or_recur"]

        num_g = make_num_tensor(b, gated["num_cols"], gated["scaler"], device)
        cats_g = make_cat_tensors(b, gated["maps"], device)

        # RESID
        delta_res = resid["model"](vols, num_res, sex_id, dis_id, pr_id)
        yhat_res = ytab + delta_res
        score_res_yhat = yhat_res[:, ac_idx].mean()
        cam_res_yhat = resid["cam"](score_res_yhat)
        np.save(out_case / "resid_cam_yhat.npy", cam_res_yhat)
        cam_res_yhat_show = upsample_cam_to(vol_orig_zyx.shape, cam_res_yhat) if args.viz_mode=="original" else cam_res_yhat
        vol_show = vol_orig_zyx if args.viz_mode=="original" else vol_dhw
        save_overlay_png(vol_show, cam_res_yhat_show, str(out_case / "resid_overlay_yhat"), axial_only=args.axial_only,
                     hu_min=args.hu_min, hu_max=args.hu_max,
                     mask_mode=args.cam_mask_mode, mask_eps=args.cam_mask_eps,
                     mask_hu_low=args.cam_mask_hu_low, mask_hu_high=args.cam_mask_hu_high,
                     cam_clip_percentile=args.cam_clip_percentile,
                     cam_border_crop=args.cam_border_crop,
                     axial_slice_mode=args.axial_slice_mode,
                     axial_center_frac=args.axial_center_frac,
                     axial_air_hu_thr=args.axial_air_hu_thr,
                     axial_z_window=getattr(args,'axial_z_window',-1),
                     axial_score_method=getattr(args,'axial_score_method','sum'),
                     axial_top_pct=getattr(args,'axial_top_pct',0.01),
                     axial_min_count=getattr(args,'axial_min_count',0)
                     )
        if args.viz_mode=="original":
            np.save(out_case / "resid_cam_yhat_upsampled.npy", cam_res_yhat_show)

        if args.also_save_resid_delta_cam:
            score_res_delta = delta_res[:, ac_idx].mean()
            cam_res_delta = resid["cam"](score_res_delta)
            np.save(out_case / "resid_cam_delta.npy", cam_res_delta)
            cam_res_delta_show = upsample_cam_to(vol_orig_zyx.shape, cam_res_delta) if args.viz_mode=="original" else cam_res_delta
            vol_show = vol_orig_zyx if args.viz_mode=="original" else vol_dhw
            save_overlay_png(vol_show, cam_res_delta_show, str(out_case / "resid_overlay_delta"), axial_only=args.axial_only,
                     hu_min=args.hu_min, hu_max=args.hu_max,
                     mask_mode=args.cam_mask_mode, mask_eps=args.cam_mask_eps,
                     mask_hu_low=args.cam_mask_hu_low, mask_hu_high=args.cam_mask_hu_high,
                     cam_clip_percentile=args.cam_clip_percentile,
                     cam_border_crop=args.cam_border_crop,
                     axial_slice_mode=args.axial_slice_mode,
                     axial_center_frac=args.axial_center_frac,
                     axial_air_hu_thr=args.axial_air_hu_thr,
                     axial_z_window=getattr(args,'axial_z_window',-1),
                     axial_score_method=getattr(args,'axial_score_method','sum'),
                     axial_top_pct=getattr(args,'axial_top_pct',0.01),
                     axial_min_count=getattr(args,'axial_min_count',0)
                     )
            if args.viz_mode=="original":
                np.save(out_case / "resid_cam_delta_upsampled.npy", cam_res_delta_show)

        # GATED
        delta_g, gate_g = gated["model"](vols, num_g, cats_g, y_tab=ytab)
        yhat_g = ytab + gate_g * delta_g

        if args.objective == "yhat_ac_mean":
            score_g = yhat_g[:, ac_idx].mean()
        elif args.objective == "delta_ac_mean":
            score_g = delta_g[:, ac_idx].mean()
        elif args.objective == "gate_ac_mean":
            score_g = gate_g[:, ac_idx].mean()
        elif args.objective == "gated_term_ac_mean":
            score_g = (gate_g * delta_g)[:, ac_idx].mean()
        else:
            raise ValueError(args.objective)

        cam_g = gated["cam"](score_g)
        np.save(out_case / "gated_cam.npy", cam_g)
        np.save(out_case / "gate.npy", gate_g.detach().cpu().numpy()[0])
        cam_g_show = upsample_cam_to(vol_orig_zyx.shape, cam_g) if args.viz_mode=="original" else cam_g
        vol_show = vol_orig_zyx if args.viz_mode=="original" else vol_dhw
        save_overlay_png(vol_show, cam_g_show, str(out_case / "gated_overlay"), axial_only=args.axial_only,
                     hu_min=args.hu_min, hu_max=args.hu_max,
                     mask_mode=args.cam_mask_mode, mask_eps=args.cam_mask_eps,
                     mask_hu_low=args.cam_mask_hu_low, mask_hu_high=args.cam_mask_hu_high,
                     cam_clip_percentile=args.cam_clip_percentile,
                     cam_border_crop=args.cam_border_crop,
                     axial_slice_mode=args.axial_slice_mode,
                     axial_center_frac=args.axial_center_frac,
                     axial_air_hu_thr=args.axial_air_hu_thr,
                     axial_z_window=getattr(args,'axial_z_window',-1),
                     axial_score_method=getattr(args,'axial_score_method','sum'),
                     axial_top_pct=getattr(args,'axial_top_pct',0.01),
                     axial_min_count=getattr(args,'axial_min_count',0)
                     )
        if args.viz_mode=="original":
            np.save(out_case / "gated_cam_upsampled.npy", cam_g_show)

        meta = {
            "id": pid,
            "fold": fold,
            "roi_path": roi_path,
            "side": str(side),
            "roi_col": args.roi_col,
            "objective_gated": args.objective,
            "viz_mode": args.viz_mode,
            "axial_only": bool(args.axial_only),
            "ac_idx": ac_idx,
            "ytab_ac_mean": float(ytab[0, ac_idx].mean().detach().cpu().item()),
            "yhat_res_ac_mean": float(yhat_res[0, ac_idx].mean().detach().cpu().item()),
            "yhat_gated_ac_mean": float(yhat_g[0, ac_idx].mean().detach().cpu().item()),
            "gate_ac_mean": float(gate_g[0, ac_idx].mean().detach().cpu().item()),
            "resid_ckpt": str(resid_ckpt_paths[fold]),
            "gated_ckpt": str(gated_ckpt_paths[fold]),
        }
        with open(out_case / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"[OK] fold{fold} {pid} -> {out_case}")

    print("[DONE] All cases processed.")


if __name__ == "__main__":
    main()