from __future__ import annotations

import numpy as np


def save_center_qa_png(vol_zyx: np.ndarray, center_zyx: tuple[int, int, int], out_png: str, wl: float = 400.0, ww: float = 4000.0, title: str | None = None) -> None:
    import matplotlib.pyplot as plt

    cz, cy, cx = center_zyx
    axial = vol_zyx[cz, :, :]
    coronal = vol_zyx[:, cy, :]
    sagittal = vol_zyx[:, :, cx]
    vmin = wl - ww / 2.0
    vmax = wl + ww / 2.0
    fig = plt.figure(figsize=(12, 4))
    if title:
        fig.suptitle(title)
    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(axial, cmap="gray", vmin=vmin, vmax=vmax)
    ax1.axhline(cy, color="lime", linewidth=0.8)
    ax1.axvline(cx, color="lime", linewidth=0.8)
    ax1.axis("off")
    ax2 = plt.subplot(1, 3, 2)
    ax2.imshow(coronal, cmap="gray", vmin=vmin, vmax=vmax)
    ax2.axhline(cz, color="lime", linewidth=0.8)
    ax2.axvline(cx, color="lime", linewidth=0.8)
    ax2.axis("off")
    ax3 = plt.subplot(1, 3, 3)
    ax3.imshow(sagittal, cmap="gray", vmin=vmin, vmax=vmax)
    ax3.axhline(cz, color="lime", linewidth=0.8)
    ax3.axvline(cy, color="lime", linewidth=0.8)
    ax3.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
