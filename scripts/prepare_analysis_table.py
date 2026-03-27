#!/usr/bin/env python3
from __future__ import annotations

import argparse
from chole_predict.data.path_injection import add_roi_paths_to_csv
from chole_predict.utils.parsing import parse_int_csv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--in_csv', required=True)
    ap.add_argument('--out_csv', required=True)
    ap.add_argument('--roi_dir', required=True)
    ap.add_argument('--id_col', default='id')
    ap.add_argument('--sizes', default='25,40,60')
    args = ap.parse_args()
    add_roi_paths_to_csv(args.in_csv, args.out_csv, args.roi_dir, id_col=args.id_col, sizes=parse_int_csv(args.sizes))


if __name__ == '__main__':
    main()
