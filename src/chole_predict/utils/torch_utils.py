from __future__ import annotations

import torch


def get_autocast(device: torch.device, enabled: bool):
    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", enabled=enabled)
    return torch.amp.autocast(device_type="cpu", enabled=False)


def get_grad_scaler(device: torch.device, enabled: bool):
    if device.type == "cuda":
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.cuda.amp.GradScaler(enabled=enabled)
    try:
        return torch.amp.GradScaler("cpu", enabled=False)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=False)
