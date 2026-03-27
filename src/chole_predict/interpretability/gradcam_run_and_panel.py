#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
End-to-end runner:
1) Generate RESID Grad-CAM overlays for a pre-built case_list (e.g., case_list_resid_vs_tab.csv)
2) Build a paper-ready RESID Grad-CAM panel figure (PNG+PDF) from those outputs

This script intentionally shells out to your existing scripts, so it will work even if those
scripts live in your project folder and rely on your project's imports.

Typical usage (Windows cmd.exe):

  python run_resid_gradcam_and_panel.py ^
    --gradcam_py make_gradcam_for_case_list_fixedfold_v6_layersplit_patched_masknorm_autoslice_fixed_v6_foldfix.py ^
    --panel_py  make_resid_gradcam_figure_panel.py ^
    --case_list_csv experiments\20260218_143700_CroppedROI_out8_AB\_resid_gradcam_selection\case_list_resid_vs_tab.csv ^
    --manifest_json experiments\20260218_143700_CroppedROI_out8_AB\_resid_gradcam_selection\selection_manifest.json ^
    --base_csv df_fixed_with_cropped_roi.csv ^
    --per_patient_csv experiments\20260218_143700_CroppedROI_out8_AB\per_patient_results.csv ^
    --exp_root experiments\20260218_143700_CroppedROI_out8_AB ^
    --roi_col roi_path_25_sphere ^
    --gradcam_out_dir GradCAM_RESID_delta_net8_thr5_soft ^
    --panel_out_prefix FIG_resid_gradcam_panel_soft ^
    --cols 4 ^
    -- ^
    --objective delta_ac_mean ^
    --viz_mode original ^
    --axial_only ^
    --cam_mode abs ^
    --target_layer_name_resid encoder.net.8 ^
    --cam_mask_mode range --cam_mask_hu_low -300 --cam_mask_hu_high 500 ^
    --cam_clip_percentile 95 ^
    --cam_border_crop 2 ^
    --axial_slice_mode maxcam_mask ^
    --axial_center_frac 0.35 ^
    --axial_z_window 15 ^
    --axial_score_method top_pct_mean ^
    --axial_top_pct 0.05 ^
    --axial_min_count 100

Notes
- Everything after a standalone "--" is forwarded verbatim to the Grad-CAM script.
- If you omit the "-- ...", this runner will use a safe default set of Grad-CAM args
  that tends to suppress bone dominance a bit (soft/range mask + maxcam_mask slice pick).

Outputs
- <gradcam_out_dir>/... (whatever your Grad-CAM script writes)
- <panel_out_prefix>.png / .pdf / _paths.csv (in the current working directory unless you include a path in --panel_out_prefix)
"""

from __future__ import annotations
import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List


def _norm_py(p: str) -> str:
    # Allow passing either an absolute path, a relative path, or just a filename in CWD.
    pp = Path(p)
    if pp.exists():
        return str(pp)
    # Try alongside this runner (if copied together)
    here = Path(__file__).resolve().parent
    cand = here / p
    if cand.exists():
        return str(cand)
    return p  # let subprocess fail with a clear error


def _default_gradcam_forward_args() -> List[str]:
    # Default chosen to reduce "all-bone-all-the-time" tendencies:
    # - abs CAM (stable)
    # - HU range mask focusing on soft tissue / moderate HU
    # - slightly lower clip percentile
    # - choose z by maxcam_mask (after mask) rather than raw CAM
    return [
        "--objective", "delta_ac_mean",
        "--viz_mode", "original",
        "--axial_only",
        "--cam_mode", "abs",
        "--target_layer_name_resid", "encoder.net.8",
        "--cam_mask_mode", "range",
        "--cam_mask_hu_low", "-300",
        "--cam_mask_hu_high", "500",
        "--cam_clip_percentile", "95",
        "--cam_border_crop", "2",
        "--axial_slice_mode", "maxcam_mask",
        "--axial_center_frac", "0.35",
        "--axial_z_window", "15",
        "--axial_score_method", "top_pct_mean",
        "--axial_top_pct", "0.05",
        "--axial_min_count", "100",
    ]


def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--python", default=sys.executable, help="Python executable to use (default: current)")
    ap.add_argument("--gradcam_py", required=True, help="Path to Grad-CAM script (fold-fix version recommended)")
    ap.add_argument("--panel_py", required=True, help="Path to make_resid_gradcam_figure_panel.py")

    # Shared inputs for Grad-CAM
    ap.add_argument("--case_list_csv", required=True, help="case_list_resid_vs_tab.csv")
    ap.add_argument("--base_csv", required=True, help="df_fixed_with_cropped_roi.csv")
    ap.add_argument("--per_patient_csv", required=True, help="per_patient_results.csv")
    ap.add_argument("--exp_root", required=True, help="experiment root (contains checkpoints)")
    ap.add_argument("--roi_col", default="roi_path_25_sphere", help="ROI path column name in base_csv")
    ap.add_argument("--gradcam_out_dir", required=True, help="Output directory name for Grad-CAM outputs")

    # Inputs for panel builder
    ap.add_argument("--manifest_json", default=None, help="selection_manifest.json (recommended). If omitted, case_list_csv is used.")
    ap.add_argument("--panel_out_prefix", default="FIG_resid_gradcam_panel", help="Output prefix for panel figure (png/pdf/csv)")
    ap.add_argument("--cols", type=int, default=4, help="Number of columns per row in the panel")
    ap.add_argument("--show_fold", action="store_true", help="Show fold in panel labels")
    ap.add_argument("--anonymize_panel", action="store_true", help="Anonymize panel labels (PatientA, PatientB, ...) and hide row labels")
    ap.add_argument("--patient_prefix", default="Patient", help="Prefix for anonymized panel labels")
    ap.add_argument("--hide_panel_title", action="store_true", help="Hide panel title")

    # Parse args, but keep everything after "--" to forward to gradcam script.
    if "--" in sys.argv:
        idx = sys.argv.index("--")
        known = sys.argv[1:idx]
        forward = sys.argv[idx + 1:]
        args = ap.parse_args(known)
    else:
        args = ap.parse_args()
        forward = []

    pyexe = args.python
    gradcam_py = _norm_py(args.gradcam_py)
    panel_py = _norm_py(args.panel_py)

    # --- 1) Run Grad-CAM
    gradcam_cmd = [
        pyexe, gradcam_py,
        "--case_list_csv", args.case_list_csv,
        "--base_csv", args.base_csv,
        "--per_patient_csv", args.per_patient_csv,
        "--exp_root", args.exp_root,
        "--roi_col", args.roi_col,
        "--out_dir", args.gradcam_out_dir,
    ]

    if forward:
        gradcam_cmd += forward
    else:
        gradcam_cmd += _default_gradcam_forward_args()

    print("\n[RUN] Grad-CAM command:")
    print(" ".join(shlex.quote(s) for s in gradcam_cmd))
    subprocess.check_call(gradcam_cmd)

    # --- 2) Build panel
    panel_cmd = [
        pyexe, panel_py,
        "--gradcam_root", args.gradcam_out_dir,
        "--out_prefix", args.panel_out_prefix,
        "--cols", str(args.cols),
    ]
    if args.show_fold:
        panel_cmd.append("--show_fold")
    if args.anonymize_panel:
        panel_cmd += ["--anonymize_labels", "--patient_prefix", args.patient_prefix, "--hide_row_labels"]
    if args.hide_panel_title:
        panel_cmd.append("--hide_title")
    if args.manifest_json:
        panel_cmd += ["--manifest_json", args.manifest_json]
    else:
        panel_cmd += ["--case_list_csv", args.case_list_csv]

    print("\n[RUN] Panel command:")
    print(" ".join(shlex.quote(s) for s in panel_cmd))
    subprocess.check_call(panel_cmd)

    print("\n[DONE] End-to-end complete.")
    print(f"  Grad-CAM outputs: {args.gradcam_out_dir}")
    print(f"  Panel figure: {args.panel_out_prefix}.png / .pdf")


if __name__ == "__main__":
    main()
