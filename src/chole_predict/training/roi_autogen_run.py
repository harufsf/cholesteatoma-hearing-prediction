#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import math
import os
import random
from datetime import datetime
from typing import List

import numpy as np
import torch
import torch.nn as nn

from chole_predict.analysis.roi_eval import eval_root
from chole_predict.data.case_loader import load_cases_from_csv
from chole_predict.data.case_schema import CaseInfo
from chole_predict.io.json_io import load_json
from chole_predict.qa.roi_autogen_qa import save_qa_montage
from chole_predict.roi.sphere_crop import extract_spherical_roi
from chole_predict.training.roi_autogen_data import prepare_case_inputs
from chole_predict.training.roi_autogen_infer import infer_center_for_case
from chole_predict.training.roi_autogen_train import train_one_fold
from chole_predict.utils.reproducibility import enable_determinism

DEFAULT_ISO_MM = 0.5
DEFAULT_HU_PAD = -1000.0
DEFAULT_INPUT_SHAPE = (64, 64, 64)
DEFAULT_SIGMA_VOX = 2.5
DEFAULT_CROP_MM = (200.0, 160.0, 160.0)
DEFAULT_ROI_SIZES_MM = [25, 40, 60]

def reqa_missing(args: argparse.Namespace) -> None:
    """
    Regenerate QA images ONLY (no training, no inference).
    Typical use:
      - after a CV run, some cases may have *_pred_center.json but missing QA png.
      - this scans fold output dirs and recreates QA for cases with dist >= threshold,
        optionally only when QA file is missing.

    It reloads the volume and GT/click from vw_point.json (same pipeline as run),
    and reads pred_center from *_pred_center.json.
    """
    root = os.path.abspath(args.root)
    run_dir = args.run_dir
    if not os.path.isabs(run_dir):
        run_dir = os.path.join(root, run_dir)
    if not os.path.isdir(run_dir):
        raise SystemExit(f"run_dir not found: {run_dir}")

    gt_csv = os.path.join(root, args.gt_csv)
    point_json_dir = os.path.join(root, args.point_json_dir)

    cases = load_cases_from_csv(gt_csv, point_json_dir, root=root, dicom_root=args.dicom_root)
    case_map = {c.pid: c for c in cases}

    # which folds to scan
    fold_dirs = []
    if args.folds:
        foldset = set([int(x) for x in re.split(r"[,\s]+", args.folds.strip()) if x.strip()])
        for f in sorted(foldset):
            fold_dirs.append((f, os.path.join(run_dir, f"fold_{f}")))
    else:
        for d in sorted(glob.glob(os.path.join(run_dir, "fold_*"))):
            bn = os.path.basename(d)
            try:
                f = int(bn.split("_")[1])
            except Exception:
                continue
            fold_dirs.append((f, d))

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "cpu" else args.device)
    print("Device:", device)
    click_keys = args.click_keys.split(",") if getattr(args,'click_keys',"") else None
    click_keys = args.click_keys.split(",") if getattr(args,'click_keys',"") else None
    gt_keys = args.gt_keys.split(",") if args.gt_keys else None

    n_total = 0
    n_done = 0
    n_skip = 0
    n_fail = 0

    for fold, fold_out in fold_dirs:
        if not os.path.isdir(fold_out):
            print(f"[reqa] fold dir missing: {fold_out}")
            continue

        pred_files = sorted(glob.glob(os.path.join(fold_out, "**", "*_pred_center.json"), recursive=True))
        if args.ids:
            idset = set([normalize_pid(x) for x in re.split(r"[,\s]+", args.ids.strip()) if x.strip()])
            pred_files = [p for p in pred_files if normalize_pid(os.path.basename(p).split("_")[0]) in idset]

        print("\n" + "="*80)
        print(f"[reqa] fold={fold} pred_files={len(pred_files)}")
        for pf in pred_files:
            n_total += 1
            try:
                res = load_json(pf)
                pid = normalize_pid(str(res.get("patient_id") or os.path.basename(pf).split("_")[0]))
                c = case_map.get(pid)
                if c is None:
                    raise RuntimeError(f"Case not found in gt_csv for pid={pid}")

                # Determine QA output path (keep consistent with run mode)
                qa_png = os.path.join(fold_out, f"{pid}_qa.png")
                if args.missing_only:
                    if os.path.exists(qa_png):
                        n_skip += 1
                        continue

                # distance threshold check
                dist = None
                if isinstance(res.get("eval"), dict) and ("dist_mm" in res["eval"]):
                    dist = float(res["eval"]["dist_mm"])
                if dist is None:
                    # compute from stored points if eval missing
                    gt = np.asarray(res.get("gt_center_zyx_canon_f", [np.nan]*3), dtype=np.float32)
                    pr = np.asarray(res.get("pred_center_zyx_canon_f", [np.nan]*3), dtype=np.float32)
                    dist = float(np.linalg.norm((pr-gt) * float(args.iso_mm)))

                if dist < float(args.dist_ge):
                    n_skip += 1
                    continue

                # Rebuild inputs (canonical full volume + canonical gt/click)
                inp = prepare_case_inputs(
                    case=c,
                    iso_mm=args.iso_mm,
                    crop_size_mm=tuple(args.crop_mm),
                    input_shape=tuple(args.input_shape),
                    click_keys=click_keys,
                    gt_keys=gt_keys,
                    vw_points_space=args.vw_points_space,
                    crop_y_front_mm=args.crop_y_front_mm,
                    crop_y_back_mm=args.crop_y_back_mm,
                    anchor_method=args.anchor_method,
                    anchor_x_frac=args.anchor_x_frac,
                    anchor_y_shift_mm=args.anchor_y_shift_mm,
                    anchor_z_shift_mm=args.anchor_z_shift_mm,
                    anterior_mask_y_mm=args.anterior_mask_y_mm,
                    anterior_mask_alpha=args.anterior_mask_alpha,
                    anterior_mask_ramp_mm=args.anterior_mask_ramp_mm,
                    anterior_mask_mode=args.anterior_mask_mode,
                )
                gt_canon = np.asarray(res.get("gt_center_zyx_canon_f", inp["gt_canon_zyx"].tolist()), dtype=np.float32)
                pr_canon = np.asarray(res.get("pred_center_zyx_canon_f", None), dtype=np.float32)
                if pr_canon is None or (np.any(~np.isfinite(pr_canon))):
                    raise RuntimeError("pred_center_zyx_canon_f missing or invalid in pred_center.json")

                if not HAS_MPL:
                    raise RuntimeError("matplotlib is not available; cannot generate QA images.")
                title = (f"{pid} dist={dist:.2f}mm fold={fold} side={c.side_rl} "
                         f"flipped={inp['flipped_lr']} gt_key={inp['point_keys'].get('gt_key_used','')}")
                save_qa_montage(inp["vol_full_canon_zyx"], gt_canon, pr_canon, None, qa_png, title=title)
                n_done += 1
            except Exception as e:
                n_fail += 1
                print(f"[reqa][NG] {os.path.basename(pf)}: {e}")

    print("\n[reqa] Done.")
    print(f"[reqa] total={n_total}  regenerated={n_done}  skipped={n_skip}  failed={n_fail}")

def run_cv(args: argparse.Namespace) -> None:
    root = os.path.abspath(args.root)
    gt_csv = os.path.join(root, args.gt_csv)
    point_json_dir = os.path.join(root, args.point_json_dir)

    cases = load_cases_from_csv(gt_csv, point_json_dir, root=root, dicom_root=args.dicom_root)
    if args.ids:
        idset = set([normalize_pid(x) for x in re.split(r"[,\s]+", args.ids.strip()) if x.strip()])
        cases = [c for c in cases if c.pid in idset]
    if not cases:
        raise SystemExit("No cases found after filtering.")

    folds = sorted(set([c.fold for c in cases]))
    if args.folds:
        foldset = set([int(x) for x in re.split(r"[,\s]+", args.folds.strip()) if x.strip()])
        folds = [f for f in folds if f in foldset]

    out_dir = os.path.join(root, args.out_dir if args.out_dir else f"AutoROI_ClickCenter_CANON_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    ensure_dir(out_dir)
    print("Output:", out_dir)

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "cpu" else args.device)
    print("Device:", device)
    click_keys = args.click_keys.split(",") if getattr(args,'click_keys',"") else None
    gt_keys = args.gt_keys.split(",") if args.gt_keys else None

    for fold in folds:
        fold_out = os.path.join(out_dir, f"fold_{fold}")
        ensure_dir(fold_out)

        train_cases = [c for c in cases if c.fold != fold]
        val_cases   = [c for c in cases if c.fold == fold]

        print("\n" + "="*80)
        print(f"[Fold {fold}] train={len(train_cases)}  val={len(val_cases)}")

        # -------------------------
        # NO-LEAK guarantee:
        #  - val_cases (this fold) must never be used for training nor for early-stopping/validation.
        #  - we create dev_cases ONLY from train_cases (other folds).
        # -------------------------
        val_pids = set([c.pid for c in val_cases])
        train_pids = set([c.pid for c in train_cases])
        inter = sorted(list(val_pids.intersection(train_pids)))
        if inter:
            raise RuntimeError(f"[Fold {fold}] DATA LEAKAGE: train/val overlap: {inter[:10]} (and more)" if len(inter)>10 else f"[Fold {fold}] DATA LEAKAGE: train/val overlap: {inter}")

        # Dev split from training folds (for monitoring only). This keeps fold-{fold} completely untouched.
        rng = np.random.RandomState(int(args.seed) + int(fold) * 10007)
        train_cases_shuf = list(train_cases)
        rng.shuffle(train_cases_shuf)
        dev_cases = []
        train_cases_used = train_cases_shuf
        if args.dev_fraction > 0 and len(train_cases_shuf) >= 3:
            n_dev = int(round(len(train_cases_shuf) * float(args.dev_fraction)))
            n_dev = max(1, min(n_dev, len(train_cases_shuf)-2))  # keep >=2 for training
            dev_cases = train_cases_shuf[:n_dev]
            train_cases_used = train_cases_shuf[n_dev:]
        print(f"[Fold {fold}] dev(from-train)={len(dev_cases)}  train_used={len(train_cases_used)}")

        # If train set is empty (e.g., running with --ids only 1 case), skip training.
        # Still allow QA generation to verify GT/click alignment.
        if len(train_cases) == 0:
            print(f"[Fold {fold}] WARNING: train set is empty. Skipping training and running QA-only for val cases.")
            for c in val_cases:
                try:
                    inp = prepare_case_inputs(
                    case=c,
                    iso_mm=args.iso_mm,
                    crop_size_mm=tuple(args.crop_mm),
                    input_shape=tuple(args.input_shape),
                    click_keys=click_keys,
                    gt_keys=gt_keys,
                    vw_points_space=args.vw_points_space,
                    crop_y_front_mm=args.crop_y_front_mm,
                    crop_y_back_mm=args.crop_y_back_mm,
                    anchor_method=args.anchor_method,
                    anchor_x_frac=args.anchor_x_frac,
                    anchor_y_shift_mm=args.anchor_y_shift_mm,
                    anchor_z_shift_mm=args.anchor_z_shift_mm,
                    anterior_mask_y_mm=args.anterior_mask_y_mm,
                    anterior_mask_alpha=args.anterior_mask_alpha,
                    anterior_mask_ramp_mm=args.anterior_mask_ramp_mm,
                    anterior_mask_mode=args.anterior_mask_mode,
                )
                    # Always save a small json for debugging even if matplotlib is unavailable
                    qa_dir = os.path.join(fold_out, "qa")
                    ensure_dir(qa_dir)
                    qa_json = os.path.join(qa_dir, f"{c.pid}_qa_only.json")
                    save_json({
                        "patient_id": c.pid,
                        "fold": int(fold),
                        "side_rl": c.side_rl,
                        "dicom_dir": c.dicom_dir,
                        "point_json": c.point_json,
                        "vw_points_space": args.vw_points_space,
                        "iso_mm": float(args.iso_mm),
                        "flipped_lr": bool(inp.get("flipped_lr", False)),
                        "anchor_canon_zyx_f": [float(v) for v in inp["anchor_canon_zyx"].tolist()],
                        "gt_canon_zyx_f": [float(v) for v in inp["gt_canon_zyx"].tolist()],
                        "point_keys": inp.get("point_keys", {}),
                    }, qa_json)

                    if args.save_qa and HAS_MPL:
                        qa_png = os.path.join(qa_dir, f"{c.pid}_gt_click_qa.png")
                        save_qa_montage(
                            inp["vol_full_canon_zyx"],
                            inp["gt_canon_zyx"],
                            inp["gt_canon_zyx"],  # pred same as gt for QA-only
                            inp["anchor_canon_zyx"],
                            qa_png,
                            title=f"{c.pid} fold{fold} QA-only (train=0) flipped={inp.get('flipped_lr', False)}",
                        )
                        print(f"[Fold {fold}] QA-only OK: {c.pid} -> {os.path.basename(qa_png)}")
                    else:
                        if args.save_qa and (not HAS_MPL):
                            print(f"[Fold {fold}] QA-only note: matplotlib not available; wrote {os.path.basename(qa_json)}")
                        else:
                            print(f"[Fold {fold}] QA-only OK: {c.pid} -> {os.path.basename(qa_json)}")
                except Exception as e:
                    print(f"[Fold {fold}] QA-only failed for {c.pid}: {e}")
            continue


        model, best_epoch = train_one_fold(
            train_cases=train_cases_used, dev_cases=dev_cases,
            iso_mm=args.iso_mm, crop_size_mm=tuple(args.crop_mm),
            input_shape=tuple(args.input_shape), click_keys=click_keys, sigma_vox=args.sigma_vox,
            clip_high=float(args.clip_high),
            crop_y_front_mm=args.crop_y_front_mm, crop_y_back_mm=args.crop_y_back_mm,
            gt_keys=gt_keys,
            device=device, vw_points_space=args.vw_points_space, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed + fold,
            anchor_method=args.anchor_method, anchor_x_frac=args.anchor_x_frac, anchor_y_shift_mm=args.anchor_y_shift_mm, anchor_z_shift_mm=args.anchor_z_shift_mm,
            anterior_mask_y_mm=args.anterior_mask_y_mm, anterior_mask_alpha=args.anterior_mask_alpha, anterior_mask_ramp_mm=args.anterior_mask_ramp_mm, anterior_mask_mode=args.anterior_mask_mode,
            best_epoch_metric=args.best_epoch_metric, best_epoch_patience=args.best_epoch_patience, best_epoch_min_delta=getattr(args,'best_epoch_min_delta',0.0), best_epoch_ema=getattr(args,'best_epoch_ema',0.0)
        )

        ckpt_path = os.path.join(fold_out, "model.pt")
        torch.save(model.state_dict(), ckpt_path)
        with open(os.path.join(fold_out, "best_epoch.txt"), "w", encoding="utf-8") as f:
            f.write(str(best_epoch) + "\n")


        for c in val_cases:
            try:
                res = infer_center_for_case(
                    model=model, case=c, iso_mm=args.iso_mm, crop_size_mm=tuple(args.crop_mm),
                    input_shape=tuple(args.input_shape),
                    click_keys=click_keys,
                    gt_keys=gt_keys, device=device, vw_points_space=args.vw_points_space,
                    clip_high=float(args.clip_high),
                    crop_y_front_mm=args.crop_y_front_mm, crop_y_back_mm=args.crop_y_back_mm,
                    anchor_method=args.anchor_method, anchor_x_frac=args.anchor_x_frac, anchor_y_shift_mm=args.anchor_y_shift_mm, anchor_z_shift_mm=args.anchor_z_shift_mm,
                    anterior_mask_y_mm=args.anterior_mask_y_mm, anterior_mask_alpha=args.anterior_mask_alpha, anterior_mask_ramp_mm=args.anterior_mask_ramp_mm, anterior_mask_mode=args.anterior_mask_mode,
                    metal_thr=float(args.metal_thr), metal_alpha=float(args.metal_alpha),
                    metal_attenuate_on_guard=bool(args.metal_attenuate_on_guard),
                    metal_attenuate_always=bool(args.metal_attenuate_always),
                    guard_top1_top2_lt=float(args.guard_top1_top2_lt),
                    guard_entropy_gt=float(args.guard_entropy_gt),
                )
                inp = res["inp"]
                pred_canon = res["pred_center_full_canon_f"]
                gt_canon = res["gt_center_full_canon_f"]

                qa_png = ""
                if args.save_qa and HAS_MPL and (res["eval"]["dist_mm"] >= args.qa_only_if_dist_ge):
                    qa_png = os.path.join(fold_out, f"{c.pid}_qa.png")
                    title = (f"{c.pid} dist={res['eval']['dist_mm']:.2f}mm fold={fold} "
                             f"side={c.side_rl} flipped={inp['flipped_lr']} "
                             f"ctonly={inp['point_keys']['gt_key_used']} gt_key={inp['point_keys']['gt_key_used']}")
                    save_qa_montage(inp["vol_full_canon_zyx"], gt_canon, pred_canon, inp.get("anchor_canon_zyx", None), qa_png, title=title)

                roi_files = {}
                if args.save_rois:
                    roi_dir = os.path.join(fold_out, "rois")
                    ensure_dir(roi_dir)
                    for rmm in args.roi_sizes_mm:
                        roi = extract_spherical_roi(inp["vol_full_canon_zyx"], pred_canon, float(rmm), args.iso_mm, pad_hu=DEFAULT_HU_PAD)
                        npy_path = os.path.join(roi_dir, f"{c.pid}_pred_{int(rmm)}mm.npy")
                        np.save(npy_path, roi)
                        roi_files[f"pred_{int(rmm)}mm_npy"] = npy_path

                pred_meta = {
                    "patient_id": c.pid,
                    "fold": int(fold),
                    "side_rl": c.side_rl,
                    "flipped_lr": bool(inp["flipped_lr"]),
                    "dicom_dir": c.dicom_dir,
                    "point_json": c.point_json,
                    "point_keys": inp["point_keys"],
                    "qa_png": qa_png,
                    "confidence": res["confidence"],
                    "eval": {
                        "iso_spacing_mm": float(args.iso_mm),
                        "dist_mm": float(res["eval"]["dist_mm"]),
                        "dz_mm": float(res["eval"]["dz_mm"]),
                        "dy_mm": float(res["eval"]["dy_mm"]),
                        "dx_mm": float(res["eval"]["dx_mm"]),
                    },
                    "gt_center_zyx_canon_f": [float(v) for v in gt_canon.tolist()],
                    "pred_center_zyx_canon_f": [float(v) for v in pred_canon.tolist()],
                    "gt_center_zyx": [int(round(v)) for v in gt_canon.tolist()],
                    "pred_center_zyx": [int(round(v)) for v in pred_canon.tolist()],
                    "roi_files": roi_files,
                }

                out_json = os.path.join(fold_out, f"{c.pid}_pred_center.json")
                save_json(pred_meta, out_json)
                print(f"[OK] {c.pid} dist={res['eval']['dist_mm']:.2f}mm")

            except Exception as e:
                print(f"[NG] {c.pid}: {e}")

    print("\nAll Done. Output dir:", out_dir)
    if args.auto_eval:
        print("\nRunning internal eval...")
        eval_root(out_dir, iso_mm=args.iso_mm, out_csv=os.path.join(out_dir, "pred_center_errors.csv"))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_run = sub.add_parser("run")
    ap_run.add_argument("--root", required=True)
    ap_run.add_argument("--gt_csv", default="df_final_fixed.csv")
    ap_run.add_argument("--point_json_dir", default="vw_roi")
    ap_run.add_argument("--vw_points_space", default="auto", choices=["auto","raw","iso","iso_canon"],
                    help="Interpret vw_point.json coords: raw=DICOM grid, iso=after iso-resample, iso_canon=after iso-resample + LR-canonical. auto=heuristic")

    ap_run.add_argument("--dicom_root", default="DICOM")

    ap_run.add_argument("--folds", default="", help="comma/space separated folds. empty=all")
    ap_run.add_argument("--ids", default="", help="optional: ids to run")
    ap_run.add_argument("--out_dir", default="", help="output dir name under root")
    ap_run.add_argument("--run_dir", default="", dest="out_dir", help="alias of --out_dir")

    ap_run.add_argument("--iso_mm", type=float, default=DEFAULT_ISO_MM)
    ap_run.add_argument("--crop_mm", type=float, nargs=3, default=list(DEFAULT_CROP_MM))
    ap_run.add_argument("--crop_y_front_mm", type=float, default=None,
                    help="Optional asymmetric crop on Y: mm to keep to +Y (front/anterior) from anchor center. If set, reduces anterior FOV.")
    ap_run.add_argument("--crop_y_back_mm", type=float, default=None,
                    help="Optional asymmetric crop on Y: mm to keep to -Y (back/posterior) from anchor center. If set with crop_y_front_mm, enables asymmetric crop.")
    
    ap_run.add_argument("--clip_high", type=float, default=3071.0,
                    help="Upper HU clip value before normalization (e.g., 2000/2500/3000). Lower may suppress metal/teeth artifacts.")

    # CT-only crop anchor (deterministic; no click/GT)
    ap_run.add_argument("--anchor_method", type=str, default="bbox_frac", choices=["bbox_frac","volume_center"],
                    help="How to estimate crop center from CT only (after LR canonicalization).")
    ap_run.add_argument("--anchor_x_frac", type=float, default=0.70,
                    help="In bbox_frac anchor: x = xmin + frac*(xmax-xmin). >0.5 biases to RIGHT (target ear).")
    ap_run.add_argument("--anchor_y_shift_mm", type=float, default=0.0, help="Additional +Y shift (mm) applied to anchor.")
    ap_run.add_argument("--anchor_z_shift_mm", type=float, default=0.0, help="Additional +Z shift (mm) applied to anchor.")

    # Conditional anterior mask in canonical crop cube (anterior assumed -Y)
    ap_run.add_argument("--anterior_mask_y_mm", type=float, default=0.0,
                    help="Start suppressing anterior region at this distance (mm) from crop center toward -Y. 0 disables.")
    ap_run.add_argument("--anterior_mask_alpha", type=float, default=0.2,
                    help="Suppression strength: smaller=stronger. Used in attenuate mode.")
    ap_run.add_argument("--anterior_mask_ramp_mm", type=float, default=10.0,
                    help="Ramp width (mm) for smooth transition of anterior suppression.")
    ap_run.add_argument("--anterior_mask_mode", type=str, default="attenuate", choices=["attenuate","zero"],
                    help="attenuate: blend HU toward pad; zero: set to pad HU.")
    ap_run.add_argument("--metal_thr", type=float, default=3000.0,
                    help="HU threshold to define metal mask on the *input grid* (after crop+resize). Used for heatmap attenuation.")
    ap_run.add_argument("--metal_alpha", type=float, default=0.90,
                    help="Attenuation strength in metal regions: p *= (1 - metal_alpha*metal_mask).")
    ap_run.add_argument("--metal_attenuate_on_guard", action="store_true",
                    help="When guard triggers, attenuate probability inside metal mask and recompute center (no second forward pass).")
    ap_run.add_argument("--metal_attenuate_always", action="store_true",
                    help="Always attenuate inside metal mask (may hurt normal cases). Prefer --metal_attenuate_on_guard.")
    
    ap_run.add_argument("--guard_top1_top2_lt", type=float, default=1.05,
                    help="Low-confidence guard: triggers if top1/top2 ratio is below this.")
    ap_run.add_argument("--guard_entropy_gt", type=float, default=0.62,
                    help="Low-confidence guard: triggers if softmax entropy (normalized) is above this.")
    ap_run.add_argument("--input_shape", type=int, nargs=3, default=list(DEFAULT_INPUT_SHAPE))
    ap_run.add_argument("--sigma_vox", type=float, default=DEFAULT_SIGMA_VOX)

    ap_run.add_argument("--epochs", type=int, default=10)
    ap_run.add_argument("--best_epoch_metric", type=str, default="val_kl", choices=["val_kl","dev_dist","dev_p90","dev_max","dev_count_gt15"],
                        help="Select best epoch by this metric: val_kl (default) or dev_dist (mean distance in mm on dev set).")
    ap_run.add_argument("--best_epoch_patience", type=int, default=0,
                        help="Early stopping patience based on best_epoch_metric. 0 disables early stopping.")
    ap_run.add_argument("--best_epoch_min_delta", type=float, default=0.0,
                        help="Minimum improvement required to update best epoch (applied to selected metric).")
    ap_run.add_argument("--best_epoch_ema", type=float, default=0.0,
                        help="EMA smoothing factor for best-epoch metric in [0,1). 0 disables smoothing. Example: 0.9.")
    ap_run.add_argument("--batch_size", type=int, default=2)
    ap_run.add_argument("--lr", type=float, default=1e-3)
    ap_run.add_argument("--seed", type=int, default=1337)
    ap_run.add_argument("--deterministic", action="store_true",
                        help="Enable deterministic/reproducible mode (cudnn/cublas/tf32/seed/series sort).")
    ap_run.add_argument("--deterministic_strict", action="store_true", help="Deterministic mode but fail on non-deterministic ops.")

    ap_run.add_argument("--dev_fraction", type=float, default=0.1,
                        help="Fraction of training-fold cases used as dev set for monitoring. (NO-LEAK: dev is drawn only from non-val folds). Set 0 to disable.")

    ap_run.add_argument("--device", default="auto", choices=["auto","cpu","cuda"])
    ap_run.add_argument("--save_qa", action="store_true")
    ap_run.add_argument("--qa_only_if_dist_ge", type=float, default=0.0)
    ap_run.add_argument("--save_rois", action="store_true")
    ap_run.add_argument("--roi_sizes_mm", type=int, nargs="+", default=DEFAULT_ROI_SIZES_MM)
    ap_run.add_argument("--auto_eval", action="store_true", default=True,
                     help="Run evaluation after prediction (default: on).")
    ap_run.add_argument("--no_auto_eval", action="store_false", dest="auto_eval",
                     help="Disable automatic evaluation step.")

    ap_run.add_argument("--click_keys", default="", help="override click keys (comma-separated)")
    ap_run.add_argument("--gt_keys", default="", help="override gt keys (comma-separated)")

    ap_eval = sub.add_parser("eval")

    ap_eval.add_argument("--root", required=True)
    ap_eval.add_argument("--run_dir", required=True, help="Existing CV output dir (absolute or under --root).")
    ap_eval.add_argument("--iso_mm", type=float, default=DEFAULT_ISO_MM)
    ap_eval.add_argument("--seed", type=int, default=1337)
    ap_eval.add_argument("--deterministic", action="store_true",
                        help="Enable deterministic/reproducible mode (cudnn/cublas/tf32/seed/series sort).")
    ap_eval.add_argument("--deterministic_strict", action="store_true", help="Deterministic mode but fail on non-deterministic ops.")
    ap_eval.add_argument("--out_csv", default="pred_center_errors_eval.csv", help="Output CSV filename (under run_dir).")


    ap_reqa = sub.add_parser("reqa", aliases=["repa"])
    ap_reqa.add_argument("--root", required=True)
    ap_reqa.add_argument("--run_dir", required=True, help="Existing CV output dir (absolute or under --root).")
    ap_reqa.add_argument("--gt_csv", default="df_final_fixed.csv")
    ap_reqa.add_argument("--point_json_dir", default="vw_roi")
    ap_reqa.add_argument("--dicom_root", default="DICOM")
    ap_reqa.add_argument("--vw_points_space", default="auto", choices=["auto","raw","iso","iso_canon"])
    ap_reqa.add_argument("--iso_mm", type=float, default=DEFAULT_ISO_MM)
    ap_reqa.add_argument("--seed", type=int, default=1337)
    ap_reqa.add_argument("--deterministic", action="store_true",
                        help="Enable deterministic/reproducible mode (cudnn/cublas/tf32/seed/series sort).")
    ap_reqa.add_argument("--deterministic_strict", action="store_true", help="Deterministic mode but fail on non-deterministic ops.")
    ap_reqa.add_argument("--crop_mm", type=float, nargs=3, default=list(DEFAULT_CROP_MM))
    ap_reqa.add_argument("--input_shape", type=int, nargs=3, default=list(DEFAULT_INPUT_SHAPE))

    # (optional) match CT-only preprocessing used during run
    ap_reqa.add_argument("--anchor_method", type=str, default="bbox_frac", choices=["bbox_frac","volume_center"])
    ap_reqa.add_argument("--anchor_x_frac", type=float, default=0.70)
    ap_reqa.add_argument("--anchor_y_shift_mm", type=float, default=0.0)
    ap_reqa.add_argument("--anchor_z_shift_mm", type=float, default=0.0)
    ap_reqa.add_argument("--anterior_mask_y_mm", type=float, default=0.0)
    ap_reqa.add_argument("--anterior_mask_alpha", type=float, default=0.2)
    ap_reqa.add_argument("--anterior_mask_ramp_mm", type=float, default=10.0)
    ap_reqa.add_argument("--anterior_mask_mode", type=str, default="attenuate", choices=["attenuate","zero"])
    ap_reqa.add_argument("--folds", default="", help="comma/space separated folds to scan. empty=all")
    ap_reqa.add_argument("--ids", default="", help="optional: ids to scan")
    ap_reqa.add_argument("--device", default="auto", choices=["auto","cpu","cuda"])
    ap_reqa.add_argument("--click_keys", default="", help="override click keys (comma-separated)")
    ap_reqa.add_argument("--gt_keys", default="", help="override gt keys (comma-separated)")
    ap_reqa.add_argument("--dist_ge", type=float, default=5.0, help="Regenerate QA only when dist_mm >= this threshold.")
    ap_reqa.add_argument("--missing_only", action="store_true", help="Only regenerate if QA png does not exist.")


    args = ap.parse_args()
    if getattr(args, "deterministic", False):
        enable_determinism(getattr(args, "seed", 1337), verbose=True)
    if args.cmd == "run":
        run_cv(args)
    elif args.cmd == "eval":
        run_dir = args.run_dir
        if not os.path.isabs(run_dir):
            run_dir = os.path.join(args.root, run_dir)
        out_csv_path = os.path.join(run_dir, args.out_csv)
        eval_root(run_dir, iso_mm=args.iso_mm, out_csv=out_csv_path)
    elif args.cmd in ("reqa","repa"):
        reqa_missing(args)
    else:
        raise SystemExit("Unknown cmd")


def run_roi_autogen():
    return main()

if __name__ == "__main__":
    main()
