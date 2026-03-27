from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from chole_predict.utils.torch_utils import get_autocast


def train_gated_epoch(model, loader, opt, device: torch.device, use_amp: bool = True, scaler=None, gate_use_ytab: bool = True, lambda_gate_l1: float = 0.0, lambda_delta_l2: float = 0.0) -> float:
    model.train()
    total = 0.0
    n = 0
    for _, num, cats, vols, y_tab, y in loader:
        num = num.to(device)
        vols = vols.to(device)
        y_tab = y_tab.to(device)
        y = y.to(device)
        cats = {k: v.to(device) for k, v in cats.items()}
        opt.zero_grad(set_to_none=True)
        with get_autocast(device, enabled=(use_amp and device.type == "cuda")):
            delta, gate = model(vols, num, cats, y_tab=y_tab if gate_use_ytab else None)
            y_hat = y_tab + gate * delta
            loss = F.smooth_l1_loss(y_hat, y)
            if lambda_gate_l1 > 0:
                loss = loss + lambda_gate_l1 * gate.abs().mean()
            if lambda_delta_l2 > 0:
                loss = loss + lambda_delta_l2 * (delta ** 2).mean()
        if use_amp and device.type == "cuda" and scaler is not None:
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
def eval_gated(model, loader, device: torch.device, use_amp: bool = True, gate_use_ytab: bool = True):
    model.eval()
    total = 0.0
    n = 0
    ids, ys, ytabs, yhats, deltas, gates = [], [], [], [], [], []
    for pid, num, cats, vols, y_tab, y in loader:
        num = num.to(device)
        vols = vols.to(device)
        y_tab = y_tab.to(device)
        y = y.to(device)
        cats = {k: v.to(device) for k, v in cats.items()}
        with get_autocast(device, enabled=(use_amp and device.type == "cuda")):
            delta, gate = model(vols, num, cats, y_tab=y_tab if gate_use_ytab else None)
            y_hat = y_tab + gate * delta
            loss = F.smooth_l1_loss(y_hat, y)
        total += loss.item() * num.size(0)
        n += num.size(0)
        ids.extend(list(pid))
        ys.append(y.detach().cpu().numpy())
        ytabs.append(y_tab.detach().cpu().numpy())
        yhats.append(y_hat.detach().cpu().numpy())
        deltas.append(delta.detach().cpu().numpy())
        gates.append(gate.detach().cpu().numpy())
    if not yhats:
        return float("nan"), [], None, None, None, None, None
    return total / max(n, 1), ids, np.concatenate(ys, 0), np.concatenate(ytabs, 0), np.concatenate(yhats, 0), np.concatenate(deltas, 0), np.concatenate(gates, 0)
