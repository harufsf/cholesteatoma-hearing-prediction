from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from chole_predict.utils.torch_utils import get_autocast


def run_resid_epoch(model, loader, optim, device: torch.device, train: bool = True, use_amp: bool = True, scaler=None, delta_l2: float = 0.0) -> float:
    model.train(train)
    losses = []
    for _, num, cats, vols, y_tab, y in loader:
        num = num.to(device)
        vols = vols.to(device)
        cats = {k: v.to(device) for k, v in cats.items()}
        y_tab = y_tab.to(device)
        y = y.to(device)
        with torch.set_grad_enabled(train):
            with get_autocast(device, enabled=(use_amp and device.type == "cuda")):
                delta = model(vols, num, cats)
                y_hat = y_tab + delta
                loss = F.smooth_l1_loss(y_hat, y)
                if delta_l2 > 0:
                    loss = loss + delta_l2 * (delta ** 2).mean()
            if train:
                optim.zero_grad(set_to_none=True)
                if use_amp and device.type == "cuda" and scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optim)
                    scaler.update()
                else:
                    loss.backward()
                    optim.step()
        losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def predict_residual(model, loader, device: torch.device, use_amp: bool = True):
    model.eval()
    ids, y_pred, y_true = [], [], []
    for pid, num, cats, vols, y_tab, y in loader:
        num = num.to(device)
        vols = vols.to(device)
        cats = {k: v.to(device) for k, v in cats.items()}
        y_tab = y_tab.to(device)
        with get_autocast(device, enabled=(use_amp and device.type == "cuda")):
            delta = model(vols, num, cats)
            pred = y_tab + delta
        ids.extend(list(pid))
        y_pred.append(pred.detach().cpu().numpy())
        y_true.append(y.numpy())
    return ids, np.concatenate(y_pred, 0), np.concatenate(y_true, 0)
