#!/usr/bin/env python3
from __future__ import annotations

import argparse

from chole_predict.pipeline.full_pipeline import run_full_pipeline
from chole_predict.utils.config import load_yaml


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run the modular full pipeline with optional YAML configuration."
    )
    ap.add_argument('--config', default=None, help='YAML config path.')
    ap.add_argument('--backend', choices=['modular', 'legacy'], default='modular')
    ap.add_argument('--in_csv', default=None)
    ap.add_argument('--out_csv', default=None)
    ap.add_argument('--roi_dir', default=None)
    ap.add_argument('--experiment_name', default=None)
    ap.add_argument('--sizes', default='25,40,60')
    args = ap.parse_args()

    cfg: dict = {}
    if args.config:
        cfg = load_yaml(args.config)
        for k, v in cfg.items():
            if hasattr(args, k) and getattr(args, k) in (None, '25,40,60'):
                setattr(args, k, v)

    if args.backend == 'legacy':
        raise SystemExit(
            'Legacy backend import execution is not yet wired as function. '
            'Use modular backend or direct legacy script.'
        )

    if not all([args.in_csv, args.out_csv, args.roi_dir, args.experiment_name]):
        raise SystemExit('--in_csv --out_csv --roi_dir --experiment_name are required')

    run_full_pipeline(
        in_csv=args.in_csv,
        out_csv=args.out_csv,
        roi_dir=args.roi_dir,
        experiment_name=args.experiment_name,
        sizes=args.sizes,
        target_cols=cfg.get('target_cols'),
        run_tabular=cfg.get('run_tabular', True),
        run_residual=cfg.get('run_residual', True),
        run_gated=cfg.get('run_gated', True),
        config=cfg,
    )


if __name__ == '__main__':
    main()
