from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from chole_predict.data.case_schema import CaseInfo
from chole_predict.models.roi_localizer import confidence_from_logits, soft_argmax_zyx
from chole_predict.qa.roi_autogen_qa import save_qa_montage
from chole_predict.roi.geometry import map_point_input_to_crop, map_point_crop_to_input
from chole_predict.roi.sphere_crop import extract_spherical_roi
from chole_predict.training.roi_autogen_data import prepare_case_inputs
from chole_predict.io.json_io import save_json

def infer_center_for_case(model: nn.Module,
                         case: CaseInfo,
                         iso_mm: float,
                         crop_size_mm: Tuple[float,float,float],
                         input_shape: Tuple[int,int,int],
                         click_keys: Optional[List[str]],
                         gt_keys: Optional[List[str]],
                         device: str,
                         vw_points_space: str,
                         clip_high: float = 3071.0,
                         crop_y_front_mm: Optional[float] = None,
                         crop_y_back_mm: Optional[float] = None,
                         # CT-only anchor
                         anchor_method: str = "bbox_frac",
                         anchor_x_frac: float = 0.70,
                         anchor_y_shift_mm: float = 0.0,
                         anchor_z_shift_mm: float = 0.0,
                         # Anterior mask in canonical cube (anterior = -Y)
                         anterior_mask_y_mm: float = 0.0,
                         anterior_mask_alpha: float = 0.2,
                         anterior_mask_ramp_mm: float = 10.0,
                         anterior_mask_mode: str = "attenuate",
                         # Metal-aware attenuation (CT-only)
                         metal_thr: float = 3000.0,
                         metal_alpha: float = 0.90,
                         metal_attenuate_on_guard: bool = False,
                         metal_attenuate_always: bool = False,
                         # Uncertainty guard (CT-only)
                         guard_top1_top2_lt: float = 1.05,
                         guard_entropy_gt: float = 0.62,
                         ) -> Dict[str, Any]:
    model.eval()
    inp = prepare_case_inputs(
        case=case, iso_mm=iso_mm, crop_size_mm=crop_size_mm, input_shape=input_shape,
        click_keys=click_keys,
                gt_keys=gt_keys, vw_points_space=vw_points_space,
        crop_y_front_mm=crop_y_front_mm, crop_y_back_mm=crop_y_back_mm,
        anchor_method=anchor_method, anchor_x_frac=anchor_x_frac, anchor_y_shift_mm=anchor_y_shift_mm, anchor_z_shift_mm=anchor_z_shift_mm,
        anterior_mask_y_mm=anterior_mask_y_mm, anterior_mask_alpha=anterior_mask_alpha, anterior_mask_ramp_mm=anterior_mask_ramp_mm, anterior_mask_mode=anterior_mask_mode,
    )

    x = inp["vol_in_zyx"][None, None, ...].astype(np.float32)
    x = np.clip(x, -1024.0, float(clip_high))
    x = (x - (-1024.0)) / (float(clip_high) - (-1024.0) + 1e-6)
    x = (x * 2.0) - 1.0

    X = torch.from_numpy(x).to(device)
    with torch.no_grad():
        logits = model(X)[0,0].detach().cpu().numpy().astype(np.float32)  # (Z,Y,X)
        # probability distribution over voxels
        a = logits.reshape(-1).astype(np.float64)
        a = a - a.max()
        expa = np.exp(a)
        p = (expa / (expa.sum() + 1e-12)).astype(np.float32).reshape(input_shape)
        pred_in = soft_argmax_zyx(p)  # float ZYX in input space

    # input -> crop -> full(canon)
    pred_crop = map_point_input_to_crop(pred_in, inp["scale_crop_to_in"], inp["shift_in"])
    pred_full_canon = pred_crop + inp["origin_full_zyx"].astype(np.float32)

    gt_canon = inp["gt_canon_zyx"]

    dz_mm = float((pred_full_canon[0] - gt_canon[0]) * iso_mm)
    dy_mm = float((pred_full_canon[1] - gt_canon[1]) * iso_mm)
    dx_mm = float((pred_full_canon[2] - gt_canon[2]) * iso_mm)
    dist_mm = float(math.sqrt(dz_mm*dz_mm + dy_mm*dy_mm + dx_mm*dx_mm))

    conf = confidence_from_logits(logits)

    # ---------------------------------
    # CT-only guard & metal-aware attenuation
    # ---------------------------------
    low_conf = (conf.get("top1_top2_ratio", 999.0) < float(guard_top1_top2_lt)) or (conf.get("softmax_entropy_norm", 0.0) > float(guard_entropy_gt))
    guard_triggered = bool(low_conf)

    used_metal_attenuation = False
    if (metal_attenuate_always or (metal_attenuate_on_guard and guard_triggered)):
        try:
            vol_in_hu = inp["vol_in_zyx"].astype(np.float32)  # (Z,Y,X) on input grid, before clip/normalize
            metal_mask = (vol_in_hu > float(metal_thr)).astype(np.float32)
            p2 = p * (1.0 - float(metal_alpha) * metal_mask)
            s = float(p2.sum())
            if s > 1e-12:
                p2 = (p2 / s).astype(np.float32)
                pred_in2 = soft_argmax_zyx(p2)
                pred_crop2 = map_point_input_to_crop(pred_in2, inp["scale_crop_to_in"], inp["shift_in"])
                pred_full_canon2 = pred_crop2 + inp["origin_full_zyx"].astype(np.float32)

                pred_in = pred_in2
                pred_crop = pred_crop2
                pred_full_canon = pred_full_canon2

                dz_mm = float((pred_full_canon[0] - gt_canon[0]) * iso_mm)
                dy_mm = float((pred_full_canon[1] - gt_canon[1]) * iso_mm)
                dx_mm = float((pred_full_canon[2] - gt_canon[2]) * iso_mm)
                dist_mm = float(math.sqrt(dz_mm*dz_mm + dy_mm*dy_mm + dx_mm*dx_mm))
                used_metal_attenuation = True
        except Exception:
            used_metal_attenuation = False

    return {
        "inp": inp,
        "pred_center_full_canon_f": pred_full_canon.astype(np.float32),
        "gt_center_full_canon_f": gt_canon.astype(np.float32),
        "eval": {"dist_mm": dist_mm, "dz_mm": dz_mm, "dy_mm": dy_mm, "dx_mm": dx_mm},
        "confidence": conf,
        "guards": {
            "low_conf": bool(low_conf),
            "triggered": bool(guard_triggered),
            "used_metal_attenuation": bool(used_metal_attenuation),
        },
    }
