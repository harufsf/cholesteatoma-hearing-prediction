#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from chole_predict.utils.config import load_yaml
from chole_predict.utils.legacy_runner import run_legacy_script


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=None)
    ap.add_argument('--backend', choices=['legacy','modular'], default='legacy')
    ap.add_argument('command', nargs='?', default='run', choices=['run','eval','reqa'])
    ap.add_argument('--root', default='.')
    ap.add_argument('--gt_csv', default='df_final_fixed.csv')
    ap.add_argument('--dicom_root', default='DICOM')
    ap.add_argument('--point_json_dir', default='vw_roi')
    ap.add_argument('--out_dir', default='AutoROI_CTonly_fix2_anterior_best_p90_allfolds_v2')
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--anchor_method', default='bbox_frac')
    ap.add_argument('--anchor_x_frac', type=float, default=0.70)
    ap.add_argument('--anterior_mask_y_mm', type=float, default=30)
    ap.add_argument('--anterior_mask_alpha', type=float, default=0.15)
    ap.add_argument('--anterior_mask_ramp_mm', type=float, default=10)
    ap.add_argument('--anterior_mask_mode', default='attenuate')
    ap.add_argument('--metal_thr', type=float, default=3000)
    ap.add_argument('--metal_alpha', type=float, default=0.80)
    ap.add_argument('--metal_attenuate_always', action='store_true', default=True)
    ap.add_argument('--guard_top1_top2_lt', type=float, default=1.05)
    ap.add_argument('--guard_entropy_gt', type=float, default=0.62)
    ap.add_argument('--best_epoch_metric', default='dev_p90')
    ap.add_argument('--best_epoch_patience', type=int, default=0)
    ap.add_argument('--seed', type=int, default=1337)
    ap.add_argument('--deterministic', action='store_true', default=True)
    args, unknown = ap.parse_known_args()

    if args.config:
        cfg = load_yaml(args.config)
        for k, v in cfg.items():
            if hasattr(args, k):
                setattr(args, k, v)

    if args.backend == 'modular':
        from chole_predict.training.roi_autogen_run import run_roi_autogen
        return run_roi_autogen()

    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / 'src' / 'chole_predict' / 'legacy' / 'roi_autogen_ctonly_canon_cv_v9_patch_guard_fix2_ctonly_anterior_v8d6.py'
    cli = [args.command,
        '--root', args.root,
        '--gt_csv', args.gt_csv,
        '--dicom_root', args.dicom_root,
        '--point_json_dir', args.point_json_dir,
        '--out_dir', args.out_dir,
        '--epochs', str(args.epochs),
        '--lr', str(args.lr),
        '--anchor_method', args.anchor_method,
        '--anchor_x_frac', str(args.anchor_x_frac),
        '--anterior_mask_y_mm', str(args.anterior_mask_y_mm),
        '--anterior_mask_alpha', str(args.anterior_mask_alpha),
        '--anterior_mask_ramp_mm', str(args.anterior_mask_ramp_mm),
        '--anterior_mask_mode', args.anterior_mask_mode,
        '--metal_thr', str(args.metal_thr),
        '--metal_alpha', str(args.metal_alpha),
        '--guard_top1_top2_lt', str(args.guard_top1_top2_lt),
        '--guard_entropy_gt', str(args.guard_entropy_gt),
        '--best_epoch_metric', args.best_epoch_metric,
        '--best_epoch_patience', str(args.best_epoch_patience),
        '--seed', str(args.seed),
    ]
    if args.metal_attenuate_always:
        cli.append('--metal_attenuate_always')
    if args.deterministic:
        cli.append('--deterministic')
    cli.extend(unknown)
    run_legacy_script(script, cli)


if __name__ == '__main__':
    main()
