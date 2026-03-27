from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Tiny3DUNet(nn.Module):
    def __init__(self, base=16):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv3d(1, base, 3, padding=1), nn.ReLU(inplace=True),
                                  nn.Conv3d(base, base, 3, padding=1), nn.ReLU(inplace=True))
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = nn.Sequential(nn.Conv3d(base, base*2, 3, padding=1), nn.ReLU(inplace=True),
                                  nn.Conv3d(base*2, base*2, 3, padding=1), nn.ReLU(inplace=True))
        self.pool2 = nn.MaxPool3d(2)
        self.bott = nn.Sequential(nn.Conv3d(base*2, base*4, 3, padding=1), nn.ReLU(inplace=True),
                                  nn.Conv3d(base*4, base*4, 3, padding=1), nn.ReLU(inplace=True))
        self.up2 = nn.ConvTranspose3d(base*4, base*2, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv3d(base*4, base*2, 3, padding=1), nn.ReLU(inplace=True),
                                  nn.Conv3d(base*2, base*2, 3, padding=1), nn.ReLU(inplace=True))
        self.up1 = nn.ConvTranspose3d(base*2, base, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv3d(base*2, base, 3, padding=1), nn.ReLU(inplace=True),
                                  nn.Conv3d(base, base, 3, padding=1), nn.ReLU(inplace=True))
        self.out = nn.Conv3d(base, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b  = self.bott(self.pool2(e2))
        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        return self.out(d1)  # logits

def gaussian_heatmap(shape: Tuple[int,int,int], center_zyx: np.ndarray, sigma: float) -> np.ndarray:
    z, y, x = shape
    cz, cy, cx = [float(v) for v in center_zyx]
    zz = np.arange(z, dtype=np.float32)[:, None, None]
    yy = np.arange(y, dtype=np.float32)[None, :, None]
    xx = np.arange(x, dtype=np.float32)[None, None, :]
    d2 = (zz - cz)**2 + (yy - cy)**2 + (xx - cx)**2
    hm = np.exp(-0.5 * d2 / max(1e-6, sigma**2)).astype(np.float32)
    s = float(hm.sum())
    if s > 0:
        hm /= s
    return hm

def soft_argmax_zyx(p: np.ndarray) -> np.ndarray:
    """p is probability distribution (Z,Y,X) sum=1."""
    z, y, x = p.shape
    zz = np.arange(z, dtype=np.float32)[:, None, None]
    yy = np.arange(y, dtype=np.float32)[None, :, None]
    xx = np.arange(x, dtype=np.float32)[None, None, :]
    cz = float((p * zz).sum())
    cy = float((p * yy).sum())
    cx = float((p * xx).sum())
    return np.array([cz, cy, cx], dtype=np.float32)

def confidence_from_logits(logits: np.ndarray) -> Dict[str, Any]:
    a = logits.reshape(-1).astype(np.float64)
    a = a - a.max()
    expa = np.exp(a)
    p = expa / (expa.sum() + 1e-12)
    ent = -float(np.sum(p * np.log(p + 1e-12)))
    ent_norm = ent / math.log(len(p) + 1e-12)
    part = np.partition(p, -2)
    top2v = float(part[-2])
    top1v = float(part[-1])
    ratio = float(top1v / (top2v + 1e-12))
    return {
        "top1_value": top1v,
        "top2_value": top2v,
        "top1_top2_ratio": ratio,
        "softmax_entropy_norm": float(ent_norm),
    }
