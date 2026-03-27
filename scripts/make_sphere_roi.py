#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
from pathlib import Path
import shutil

import numpy as np

from chole_predict.io.dicom_io import load_dicom_volume_normalized
from chole_predict.io.json_io import load_json
from chole_predict.qa.roi_qa import save_center_qa_png
from chole_predict.roi.sphere_crop import extract_spherical_roi
from chole_predict.utils.config import load_yaml
from chole_predict.utils.parsing import parse_int_csv, parse_shape_csv


def _infer_id_from_json_path(json_path: str) -> str:
    base = Path(json_path).name
    return base[:-len('_pred_center.json')] if base.endswith('_pred_center.json') else Path(base).stem


def _resolve_dicom_dir(meta: dict, pid: str, dicom_root: str | None) -> str:
    for k in ['dicom_dir', 'dicom_path', 'dicom', 'dicomFolder']:
        if meta.get(k):
            return str(meta[k])
    if not dicom_root:
        raise ValueError('JSON missing dicom_dir and --dicom_root not provided.')
    for cand in [Path(dicom_root) / pid / 'DICOM', Path(dicom_root) / pid]:
        if cand.is_dir():
            return str(cand)
    raise ValueError(f'Cannot resolve DICOM dir for id={pid}')


def _resolve_center_zyx(meta: dict) -> tuple[int, int, int]:
    for key in ['pred_center_zyx', 'pred_center_zyx_canon_f']:
        if meta.get(key) is not None:
            c = meta[key]
            return int(round(c[0])), int(round(c[1])), int(round(c[2]))
    raise ValueError('Missing pred_center_zyx')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=None)
    ap.add_argument('--roi_dir', required=False)
    ap.add_argument('--recursive', action='store_true', default=True)
    ap.add_argument('--dicom_root', default=None)
    ap.add_argument('--sizes', default='25,40,60')
    ap.add_argument('--out_shape', default='64,64,64')
    ap.add_argument('--default_iso_spacing', type=float, default=0.5)
    ap.add_argument('--fill_hu', type=float, default=-1000.0)
    ap.add_argument('--qa', action='store_true')
    ap.add_argument('--qa_dir', default=None)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    if args.config:
        cfg = load_yaml(args.config)
        for k, v in cfg.items():
            if hasattr(args, k):
                setattr(args, k, v)
    if not args.roi_dir:
        raise SystemExit('--roi_dir is required')
    sizes = parse_int_csv(args.sizes)
    out_shape = parse_shape_csv(args.out_shape)
    pattern = str(Path(args.roi_dir) / '**/*_pred_center.json') if args.recursive else str(Path(args.roi_dir) / '*_pred_center.json')
    qa_dir = Path(args.qa_dir) if args.qa_dir else None
    if qa_dir:
        qa_dir.mkdir(parents=True, exist_ok=True)
    for json_path in glob.glob(pattern, recursive=args.recursive):
        meta = load_json(json_path)
        pid = _infer_id_from_json_path(json_path)
        dicom_dir = _resolve_dicom_dir(meta, pid, args.dicom_root)
        center = _resolve_center_zyx(meta)
        iso = float(meta.get('iso_spacing_mm', args.default_iso_spacing))
        vol, spacing = load_dicom_volume_normalized(dicom_dir, iso_spacing_mm=iso)
        for size in sizes:
            out_npy = Path(args.roi_dir) / f'{pid}_{size}mm_sphere.npy'
            if out_npy.exists() and not args.overwrite:
                continue
            roi = extract_spherical_roi(vol, center, cube_mm=float(size), spacing_zyx_mm=spacing, out_shape=out_shape, fill_hu=args.fill_hu)
            np.save(out_npy, roi)
        if args.qa:
            out_png = (qa_dir / f'{pid}_qa.png') if qa_dir else Path(args.roi_dir) / f'{pid}_qa.png'
            if out_png.exists() and not args.overwrite:
                continue
            save_center_qa_png(vol, center, str(out_png), title=pid)


if __name__ == '__main__':
    main()
