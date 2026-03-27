
from __future__ import annotations

import csv
import glob
import math
import os

from chole_predict.io.json_io import ensure_dir, load_json


def eval_root(root: str, iso_mm: float, out_csv: str) -> None:
    pred_files = glob.glob(os.path.join(root, '**', '*_pred_center.json'), recursive=True)
    if len(pred_files) == 0:
        raise SystemExit(f'No *_pred_center.json found under: {root}')
    fieldnames = [
        'patient_id','file','status','dist_mm','dz_mm','dy_mm','dx_mm',
        'top1_value','top2_value','top1_top2_ratio','softmax_entropy_norm','qa_png'
    ]
    rows = []
    for fp in pred_files:
        try:
            d = load_json(fp)
        except Exception:
            continue
        pid = d.get('patient_id') or os.path.basename(fp).split('_pred_center.json')[0]
        gt = d.get('gt_center_zyx') or d.get('gt_center_zyx_canon_f')
        pr = d.get('pred_center_zyx') or d.get('pred_center_zyx_canon_f')
        if not (isinstance(gt,(list,tuple)) and len(gt)==3 and isinstance(pr,(list,tuple)) and len(pr)==3):
            rows.append({'patient_id':pid,'file':fp,'status':'missing_keys','dist_mm':'','dz_mm':'','dy_mm':'','dx_mm':'',
                         'top1_value':'','top2_value':'','top1_top2_ratio':'','softmax_entropy_norm':'',
                         'qa_png':d.get('qa_png','')})
            continue
        dz = (float(pr[0]) - float(gt[0])) * iso_mm
        dy = (float(pr[1]) - float(gt[1])) * iso_mm
        dx = (float(pr[2]) - float(gt[2])) * iso_mm
        dist = math.sqrt(dz*dz + dy*dy + dx*dx)
        conf = d.get('confidence', {})
        rows.append({'patient_id':pid,'file':fp,'status':'ok','dist_mm':dist,'dz_mm':dz,'dy_mm':dy,'dx_mm':dx,
                     'top1_value':conf.get('top1_value',''),'top2_value':conf.get('top2_value',''),
                     'top1_top2_ratio':conf.get('top1_top2_ratio',''),'softmax_entropy_norm':conf.get('softmax_entropy_norm',''),
                     'qa_png':d.get('qa_png','')})
    ensure_dir(os.path.dirname(os.path.abspath(out_csv)))
    with open(out_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fieldnames})
    dists = sorted([r['dist_mm'] for r in rows if r['status']=='ok' and isinstance(r['dist_mm'], (int,float))])
    if dists:
        import numpy as np
        print(f'Saved: {out_csv}')
        print(f"Summary: n={len(dists)} median={float(np.median(dists)):.3f} p90={float(np.percentile(dists,90)):.3f} p95={float(np.percentile(dists,95)):.3f} max={max(dists):.3f} over10={sum(x>10.0 for x in dists)}")
