from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from chole_predict.utils.torch_utils import get_autocast


def run_tab_epoch(model, loader, optim, device: torch.device, train: bool = True, use_amp: bool = True, scaler=None) -> float:
    model.train(train)
    losses = []
    for _, num, cats, y in loader:
        num = num.to(device, non_blocking=True)
        cats = {k: v.to(device, non_blocking=True) for k, v in cats.items()}
        y = y.to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            with get_autocast(device, enabled=(use_amp and device.type == "cuda")):
                pred = model(num, cats)
                loss = F.smooth_l1_loss(pred, y)
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
def predict_tabular(model, loader, device: torch.device, use_amp: bool = True):
    model.eval()
    ids, preds, ys = [], [], []
    for pid, num, cats, y in loader:
        num = num.to(device, non_blocking=True)
        cats = {k: v.to(device, non_blocking=True) for k, v in cats.items()}
        with get_autocast(device, enabled=(use_amp and device.type == "cuda")):
            pred = model(num, cats)
        ids.extend(list(pid))
        preds.append(pred.detach().cpu().numpy())
        ys.append(y.numpy())
    return ids, np.concatenate(preds, 0), np.concatenate(ys, 0)
