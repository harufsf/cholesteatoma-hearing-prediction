from __future__ import annotations

import os
from typing import Optional

import numpy as np

def save_qa_montage(vol_zyx: np.ndarray, gt_zyx: np.ndarray, pr_zyx: np.ndarray, click_zyx: Optional[np.ndarray], out_png: str, title: str = "") -> None:
    if not HAS_MPL:
        return
    vol = vol_zyx
    gt = np.asarray(gt_zyx, dtype=np.float32)
    pr = np.asarray(pr_zyx, dtype=np.float32)

    z = int(round(gt[0])); y = int(round(gt[1])); x = int(round(gt[2]))
    z = max(0, min(vol.shape[0]-1, z))
    y = max(0, min(vol.shape[1]-1, y))
    x = max(0, min(vol.shape[2]-1, x))

    fig = plt.figure(figsize=(10, 6))
    fig.suptitle(title, fontsize=10)

    ax1 = fig.add_subplot(2,3,1); ax1.set_title("Axial@GT(z)")
    ax1.imshow(vol[z,:,:], cmap="gray"); ax1.scatter([gt[2]],[gt[1]], c="cyan", s=30, marker="x")
    if click_zyx is not None:
        ck=np.asarray(click_zyx,dtype=np.float32); ax1.scatter([ck[2]],[ck[1]], c="lime", s=20, marker="+")
    ax1.scatter([pr[2]],[pr[1]], c="orange", s=30, marker="o"); ax1.axis("off")

    ax2 = fig.add_subplot(2,3,2); ax2.set_title("Coronal@GT(y)")
    ax2.imshow(vol[:,y,:], cmap="gray"); ax2.scatter([gt[2]],[gt[0]], c="cyan", s=30, marker="x")
    if click_zyx is not None:
        ck=np.asarray(click_zyx,dtype=np.float32); ax2.scatter([ck[2]],[ck[0]], c="lime", s=20, marker="+")
    ax2.scatter([pr[2]],[pr[0]], c="orange", s=30, marker="o"); ax2.axis("off")

    ax3 = fig.add_subplot(2,3,3); ax3.set_title("Sagittal@GT(x)")
    ax3.imshow(vol[:,:,x], cmap="gray"); ax3.scatter([gt[1]],[gt[0]], c="cyan", s=30, marker="x")
    if click_zyx is not None:
        ck=np.asarray(click_zyx,dtype=np.float32); ax3.scatter([ck[1]],[ck[0]], c="lime", s=20, marker="+")
    ax3.scatter([pr[1]],[pr[0]], c="orange", s=30, marker="o"); ax3.axis("off")

    z2 = int(round(pr[0])); y2 = int(round(pr[1])); x2 = int(round(pr[2]))
    z2 = max(0, min(vol.shape[0]-1, z2))
    y2 = max(0, min(vol.shape[1]-1, y2))
    x2 = max(0, min(vol.shape[2]-1, x2))

    ax4 = fig.add_subplot(2,3,4); ax4.set_title("Axial@PR(z)")
    ax4.imshow(vol[z2,:,:], cmap="gray"); ax4.scatter([gt[2]],[gt[1]], c="cyan", s=30, marker="x")
    if click_zyx is not None:
        ck=np.asarray(click_zyx,dtype=np.float32); ax4.scatter([ck[2]],[ck[1]], c="lime", s=20, marker="+")
    ax4.scatter([pr[2]],[pr[1]], c="orange", s=30, marker="o"); ax4.axis("off")

    ax5 = fig.add_subplot(2,3,5); ax5.set_title("Coronal@PR(y)")
    ax5.imshow(vol[:,y2,:], cmap="gray"); ax5.scatter([gt[2]],[gt[0]], c="cyan", s=30, marker="x")
    if click_zyx is not None:
        ck=np.asarray(click_zyx,dtype=np.float32); ax5.scatter([ck[2]],[ck[0]], c="lime", s=20, marker="+")
    ax5.scatter([pr[2]],[pr[0]], c="orange", s=30, marker="o"); ax5.axis("off")

    ax6 = fig.add_subplot(2,3,6); ax6.set_title("Sagittal@PR(x)")
    ax6.imshow(vol[:,:,x2], cmap="gray"); ax6.scatter([gt[1]],[gt[0]], c="cyan", s=30, marker="x")
    if click_zyx is not None:
        ck=np.asarray(click_zyx,dtype=np.float32); ax6.scatter([ck[1]],[ck[0]], c="lime", s=20, marker="+")
    ax6.scatter([pr[1]],[pr[0]], c="orange", s=30, marker="o"); ax6.axis("off")

    ensure_dir(os.path.dirname(out_png))
    fig.tight_layout(rect=[0,0,1,0.95])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
