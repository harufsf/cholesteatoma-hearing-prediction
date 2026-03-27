
from __future__ import annotations

import os
from typing import List, Optional

import pandas as pd

from chole_predict.data.case_schema import CaseInfo
from chole_predict.data.id_utils import normalize_pid
from chole_predict.io.json_io import load_json
from chole_predict.roi.canonicalize import normalize_side_value


def load_cases_from_csv(gt_csv: str, point_json_dir: str, root: str, dicom_root: str) -> List[CaseInfo]:
    df = pd.read_csv(gt_csv)
    cols_l = {c.lower(): c for c in df.columns}

    def col_optional(*names: str) -> Optional[str]:
        for n in names:
            if n.lower() in cols_l:
                return cols_l[n.lower()]
        return None

    def col_required(name: str) -> str:
        c = col_optional(name)
        if c is None:
            raise KeyError(f"Missing column '{name}' in {gt_csv}. Found: {list(df.columns)}")
        return c

    id_col = col_required('id')
    fold_col = col_required('fold')
    side_col = col_optional('side')
    dicom_col = col_optional('dicom_dir', 'dicom_path', 'ct_dir', 'ct_path', 'series_dir')
    manifest_col = col_optional('roi_manifest_path', 'manifest_path')

    def infer_dicom_dir_from_row(row: pd.Series, pid: str) -> str:
        if dicom_col is not None:
            v = str(row[dicom_col]).strip()
            if v and v.lower() != 'nan' and os.path.isdir(v):
                return v
        if manifest_col is not None:
            mpath = str(row[manifest_col]).strip()
            if mpath and mpath.lower() != 'nan' and os.path.exists(mpath):
                try:
                    mj = load_json(mpath)
                    if isinstance(mj, dict):
                        for k in ['dicom_dir','dicom_path','ct_dir','ct_path','series_dir','dicomFolder','dicom']:
                            if k in mj and isinstance(mj[k], str) and os.path.isdir(mj[k]):
                                return mj[k]
                except Exception:
                    pass
        cand = os.path.join(root, dicom_root, pid)
        if os.path.isdir(cand):
            return cand
        raise FileNotFoundError(f'Cannot infer dicom_dir for id={pid}. Tried manifest and {cand}.')

    cases: List[CaseInfo] = []
    pj_root = point_json_dir if os.path.isabs(point_json_dir) else os.path.join(root, point_json_dir)
    for _, r in df.iterrows():
        pid = normalize_pid(r[id_col])
        if not pid:
            continue
        try:
            fold = int(r[fold_col])
        except Exception:
            continue
        side_rl = normalize_side_value(r[side_col]) if side_col else None
        dicom_dir = infer_dicom_dir_from_row(r, pid)
        pj = os.path.join(pj_root, f'{pid}_vw_point.json')
        cases.append(CaseInfo(pid=pid, dicom_dir=dicom_dir, side_rl=side_rl, fold=fold, point_json=pj))
    return cases
