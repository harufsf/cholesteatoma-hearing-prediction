from __future__ import annotations

import csv
import os
import random
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from chole_predict.data.case_schema import CaseInfo
from chole_predict.models.roi_localizer import Tiny3DUNet, gaussian_heatmap, soft_argmax_zyx
from chole_predict.training.roi_autogen_data import prepare_case_inputs
from chole_predict.roi.geometry import map_point_input_to_crop

def train_one_fold(train_cases: List[CaseInfo],
                   dev_cases: List[CaseInfo],
                   iso_mm: float,
                   crop_size_mm: Tuple[float,float,float],
                   input_shape: Tuple[int,int,int],
                   click_keys: Optional[List[str]],
                   sigma_vox: float,
                   clip_high: float,
                   crop_y_front_mm: Optional[float],
                   crop_y_back_mm: Optional[float],
                   gt_keys: Optional[List[str]],
                   device: str,
                   vw_points_space: str,
                   epochs: int = 10,
                   batch_size: int = 2,
                   lr: float = 1e-3,
                   seed: int = 1337,
                   # CT-only anchor
                   anchor_method: str = "bbox_frac",
                   anchor_x_frac: float = 0.70,
                   anchor_y_shift_mm: float = 0.0,
                   anchor_z_shift_mm: float = 0.0,
                   # Anterior mask (anterior=-Y)
                   anterior_mask_y_mm: float = 0.0,
                   anterior_mask_alpha: float = 0.2,
                   anterior_mask_ramp_mm: float = 10.0,
                   anterior_mask_mode: str = "attenuate",
                   
                   best_epoch_metric: str = "val_kl",
                   best_epoch_patience: int = 0,
                   best_epoch_min_delta: float = 0.0,
                   best_epoch_ema: float = 0.0,
                   ) -> Tuple[nn.Module, int]:
    """Train on one CV fold with **CT-only preprocessing** (no click-based crop/mask)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = Tiny3DUNet(base=16).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def make_batch(cases_batch: List[CaseInfo], return_gt_in: bool = False):
        xs = []
        ys = []
        gtins = []
        for c in cases_batch:
            inp = prepare_case_inputs(
                case=c, iso_mm=iso_mm, crop_size_mm=crop_size_mm, input_shape=input_shape,
                click_keys=click_keys,
                gt_keys=gt_keys, vw_points_space=vw_points_space,
                crop_y_front_mm=crop_y_front_mm, crop_y_back_mm=crop_y_back_mm,
                anchor_method=anchor_method, anchor_x_frac=anchor_x_frac, anchor_y_shift_mm=anchor_y_shift_mm, anchor_z_shift_mm=anchor_z_shift_mm,
                anterior_mask_y_mm=anterior_mask_y_mm, anterior_mask_alpha=anterior_mask_alpha,
                anterior_mask_ramp_mm=anterior_mask_ramp_mm, anterior_mask_mode=anterior_mask_mode,
            )
            x = inp["vol_in_zyx"][None, ...]  # (1,Z,Y,X)
            x = np.clip(x, -1024.0, float(clip_high))
            x = (x - (-1024.0)) / (float(clip_high) - (-1024.0) + 1e-6)
            x = (x * 2.0) - 1.0
            xs.append(x.astype(np.float32))

            # Target distribution in input grid (Gaussian around gt_in)
            gt_in = inp["gt_in_zyx"].astype(np.float32)
            if return_gt_in:
                gtins.append(gt_in)
            y = gaussian_heatmap(input_shape, gt_in, float(sigma_vox)).astype(np.float32)
            ys.append(y[None, ...])  # (1,Z,Y,X)

        X = torch.from_numpy(np.stack(xs, axis=0)).to(device)  # (B,1,Z,Y,X)
        Y = torch.from_numpy(np.stack(ys, axis=0)).to(device)  # (B,1,Z,Y,X)
        return X, Y, (np.stack(gtins, axis=0) if return_gt_in else None)

    best_val = float("inf")
    best_state = None
    best_epoch = -1
    bad_epochs = 0

    metric_s = None  # optional smoothed metric for best-epoch selection

    for ep in range(int(epochs)):
        model.train()
        random.shuffle(train_cases)
        losses = []
        for i in range(0, len(train_cases), int(batch_size)):
            batch = train_cases[i:i+int(batch_size)]
            Xb, Yb, _ = make_batch(batch)
            opt.zero_grad(set_to_none=True)
            logits = model(Xb)  # (B,1,Z,Y,X)

            # KL( target || pred )
            logp = torch.log_softmax(logits.flatten(2), dim=-1).view_as(logits)
            # normalize target
            yt = Yb / (Yb.sum(dim=(2,3,4), keepdim=True) + 1e-12)
            loss = (yt * (torch.log(yt + 1e-12) - logp)).sum(dim=(2,3,4)).mean()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))

        # dev monitoring (NO-LEAK: dev is drawn only from non-val folds by caller)
        model.eval()
        vlosses = []
        vdist = []
        if dev_cases:
            want_dist = (best_epoch_metric == "dev_dist")
            vz = float(crop_size_mm[0]) / float(input_shape[0])
            vy = float(crop_size_mm[1]) / float(input_shape[1])
            vx = float(crop_size_mm[2]) / float(input_shape[2])
            Z, Y, X = int(input_shape[0]), int(input_shape[1]), int(input_shape[2])
            for i in range(0, len(dev_cases), int(batch_size)):
                batch = dev_cases[i:i+int(batch_size)]
                Xb, Yb, gt_inb = make_batch(batch, return_gt_in=want_dist)
                with torch.no_grad():
                    logits = model(Xb)
                    logp = torch.log_softmax(logits.flatten(2), dim=-1).view_as(logits)
                    yt = Yb / (Yb.sum(dim=(2,3,4), keepdim=True) + 1e-12)
                    vloss = (yt * (torch.log(yt + 1e-12) - logp)).sum(dim=(2,3,4)).mean()
                vlosses.append(float(vloss.detach().cpu().item()))

                if want_dist and gt_inb is not None:
                    flat = logits.reshape(logits.shape[0], -1)
                    idxs = torch.argmax(flat, dim=1).detach().cpu().numpy().astype(np.int64)
                    zz = idxs // (Y * X)
                    rem = idxs % (Y * X)
                    yy = rem // X
                    xx = rem % X
                    pr = np.stack([zz, yy, xx], axis=1).astype(np.float32)
                    gt = gt_inb.astype(np.float32)
                    dz = (pr[:, 0] - gt[:, 0]) * vz
                    dy = (pr[:, 1] - gt[:, 1]) * vy
                    dx = (pr[:, 2] - gt[:, 2]) * vx
                    dist = np.sqrt(dz*dz + dy*dy + dx*dx)
                    vdist.extend(dist.tolist())

        tr = float(np.mean(losses)) if losses else float("nan")
        va = float(np.mean(vlosses)) if vlosses else float("nan")
        vd = float(np.mean(vdist)) if vdist else float("nan")

        if best_epoch_metric == "dev_dist" and vd == vd:
            metric = vd
            metric_name = "dev_dist_mm"
        else:
            metric = va
            metric_name = "val_KL"

        # Optional EMA smoothing for more stable best-epoch selection
        if best_epoch_ema and float(best_epoch_ema) > 0.0 and float(best_epoch_ema) < 1.0 and metric == metric:
            if metric_s is None:
                metric_s = float(metric)
            else:
                metric_s = float(best_epoch_ema) * float(metric_s) + (1.0 - float(best_epoch_ema)) * float(metric)
        else:
            metric_s = float(metric) if metric == metric else float('nan')

        extra = f" metric_s={metric_s:.4f}" if (best_epoch_ema and float(best_epoch_ema)>0.0 and float(best_epoch_ema)<1.0 and metric_s==metric_s) else ""
        print(f"[epoch {ep:03d}] train_KL={tr:.6f}  val_KL={va:.6f}  dev_dist_mm={(vd if vd==vd else float('nan')):.3f}  best_by={metric_name}{extra}")

        if metric_s == metric_s and (metric_s < best_val - float(best_epoch_min_delta)):
            best_val = float(metric_s)
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if best_epoch_patience and bad_epochs >= int(best_epoch_patience):
            print(f"[early_stop] patience={best_epoch_patience} reached at epoch={ep}, best_epoch={best_epoch} best_metric={best_val} (min_delta={best_epoch_min_delta}, ema={best_epoch_ema})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, int(best_epoch)
