#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_pipeline_cropped_roi_fixed.py

End-to-end runner:
  0) Inject sphere ROI paths into CSV (cropped_roi, fold subfolders supported)
  1) Train tabular baseline (OOF)
  2) Train residual ROI model (OOF)
  3) Train gated residual ROI model out8 (OOF)
  4) Merge OOFs -> per_patient_results.csv and run analysis_utils outputs

Assumptions:
- You run this from the project root (the folder containing analysis_utils.py).
- df_fixed.csv contains: id, fold, and target columns (A/B 8 targets).
- cropped_roi contains sphere npy files created by make_sphere_npy_and_qa... under fold_* subfolders.

Usage:
  python run_pipeline_cropped_roi_fixed.py ^
    --in_csv df_fixed.csv ^
    --roi_dir cropped_roi ^
    --experiment_name CroppedROI_out8_AB ^
    --sizes 25,40,60
"""

import os
import sys
import glob
import datetime
import subprocess
import shutil
import argparse

import analysis_utils

def run_command(cmd_list, log_file):
    print(f"Running: {' '.join(cmd_list)}")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n\n{'='*30}\nRunning: {' '.join(cmd_list)}\n{'='*30}\n")
        p = subprocess.Popen(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            print(line, end="")
            f.write(line)
        p.wait()
        if p.returncode != 0:
            print(f"[ERROR] Command failed. Check {log_file}")
            sys.exit(p.returncode)

def find_project_python():
    return sys.executable  # use current env python

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment_name", default="CroppedROI_out8_AB")
    ap.add_argument("--in_csv", default="df_fixed.csv")
    ap.add_argument("--roi_dir", required=True, help="Root folder of cropped ROI (contains fold_* subfolders)")
    ap.add_argument("--out_csv", default="df_fixed_with_cropped_roi.csv")
    ap.add_argument("--sizes", default="25,40,60")
    ap.add_argument("--roi_cols", default="roi_path_25_sphere,roi_path_40_sphere,roi_path_60_sphere")
    ap.add_argument("--id_col", default="id")
    ap.add_argument("--seed", default="0")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    # scripts (relative to project root)
    script_update = "update_csv_paths_cropped_roi_cli.py"
    script_tab = "tabular_only_mlp_fixedfold_oof_abaware_CANON.py"
    script_resid = "kfold_train_residual_roi_on_tabular_fixedfold_abaware_CANON_v4.py"
    script_gated = "kfold_train_gated_residual_roi_on_tabular_fixedfold_out8_cli_abaware_CANON_v4.py"

    # Targets (8 outputs: air A + bone B)
    TARGET = ",".join([
        "post_PTA_0.5k_A","post_PTA_1k_A","post_PTA_2k_A","post_PTA_3k_A",
        "post_PTA_0.5k_B","post_PTA_1k_B","post_PTA_2k_B","post_PTA_3k_B",
    ])

    PY = find_project_python()

    # 0) Update CSV
    print("=== Step 0: Updating CSV paths for CROPPED ROI (sphere npy) ===")
    run_command([PY, script_update,
                 "--in_csv", args.in_csv,
                 "--out_csv", args.out_csv,
                 "--roi_dir", args.roi_dir,
                 "--id_col", args.id_col,
                 "--sizes", args.sizes,
                 "--make_absolute"],
                log_file="pipeline_step0_update.log")

    # 1) Experiment folder
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("experiments", f"{date_str}_{args.experiment_name}")
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "experiment_log.txt")

    print(f"\n=== Experiment Start: {args.experiment_name} ===")
    print(f"Output Directory: {out_dir}")

    roi_cols = args.roi_cols
    common_args = ["--csv", args.out_csv, "--target", TARGET]
    if args.seed is not None:
        common_args += ["--seed", str(args.seed)]
    if args.device:
        common_args += ["--device", args.device]

    # (A) Tabular
    print("\n--- Step 1: Training Tabular Model ---")
    run_command([PY, script_tab] + common_args + [
        "--out_prefix", os.path.join(out_dir, "TAB"),
        "--epochs", "300", "--batch", "32"
    ], log_path)

    # (B) Residual
    print("\n--- Step 2: Training Residual Model (CROPPED ROI) ---")
    run_command([PY, script_resid] + common_args + [
        "--out_prefix", os.path.join(out_dir, "RESID"),
        "--roi_cols", roi_cols,
        "--roi_epochs", "150", "--roi_batch", "6"
    ], log_path)

    # (C) Gated (out8 CLI)
    print("\n--- Step 3: Training Gated Model (CROPPED ROI, out8) ---")
    gated_outdir = os.path.join(out_dir, "GATED")
    os.makedirs(gated_outdir, exist_ok=True)
    gated_cmd = [PY, script_gated,
                 "--csv", args.out_csv,
                 "--target", TARGET,
                 "--out_dir", gated_outdir,
                 "--roi_cols", roi_cols,
                 "--epochs_roi", "150",
                 "--batch_size_roi", "6"]
    if args.seed is not None:
        gated_cmd += ["--seed", str(args.seed)]
    if args.device:
        gated_cmd += ["--device", args.device]
    run_command(gated_cmd, log_path)

    # Normalize gated OOF filename into out_dir root for analysis
    expected_gated_oof = os.path.join(out_dir, "GATED_oof_predictions.csv")
    if not os.path.exists(expected_gated_oof):
        cand = glob.glob(os.path.join(gated_outdir, "*oof_predictions*.csv"))
        if cand:
            # prefer out8
            cand_sorted = sorted(cand, key=lambda p: (("out8" not in os.path.basename(p)), p))
            src = cand_sorted[0]
            print(f"[FIX] Copy gated OOF '{src}' -> '{expected_gated_oof}'")
            shutil.copyfile(src, expected_gated_oof)
        else:
            print(f"[WARNING] No gated OOF predictions found under: {gated_outdir}")

    # 4) Analysis
    print("\n--- Step 4: Analysis ---")
    tab_oof = os.path.join(out_dir, "TAB_oof_predictions.csv")
    resid_oof = os.path.join(out_dir, "RESID_oof_predictions.csv")
    gated_oof = os.path.join(out_dir, "GATED_oof_predictions.csv")

    if os.path.exists(tab_oof) and os.path.exists(resid_oof) and os.path.exists(gated_oof):
        merged_df = analysis_utils.load_and_merge_oofs(args.out_csv, tab_oof, resid_oof, gated_oof)
        merged_df.to_csv(os.path.join(out_dir, "per_patient_results.csv"), index=False, encoding="utf-8-sig")
        analysis_utils.create_summary_plots(merged_df, out_dir)
        analysis_utils.export_statistics(merged_df, out_dir)
        print(f"\n=== All Done! Results are in {out_dir} ===")
    else:
        print("\n[ERROR] One or more OOF files are missing. Cannot proceed with analysis.")
        print("  tab  :", tab_oof, os.path.exists(tab_oof))
        print("  resid:", resid_oof, os.path.exists(resid_oof))
        print("  gated:", gated_oof, os.path.exists(gated_oof))

if __name__ == "__main__":
    main()
