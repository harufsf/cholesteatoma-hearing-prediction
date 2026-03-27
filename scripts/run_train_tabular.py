#!/usr/bin/env python3
from __future__ import annotations

import argparse

from chole_predict.training.train_tabular import run_train_tabular
from chole_predict.utils.parsing import parse_csv_list


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--out_prefix", default="TAB")
    ap.add_argument("--id_col", default="id")
    ap.add_argument("--fold_col", default="fold")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no_amp", action="store_true")
    args = ap.parse_args()
    run_train_tabular(
        csv_path=args.csv,
        target_cols=parse_csv_list(args.target),
        out_prefix=args.out_prefix,
        id_col=args.id_col,
        fold_col=args.fold_col,
        seed=args.seed,
        batch_size=args.batch,
        epochs=args.epochs,
        lr=args.lr,
        val_size=args.val_size,
        num_workers=args.num_workers,
        use_amp=not args.no_amp,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
