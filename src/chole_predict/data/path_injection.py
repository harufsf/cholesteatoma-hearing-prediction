
from __future__ import annotations

from pathlib import Path
import pandas as pd


def add_roi_paths_to_csv(in_csv: str, out_csv: str, roi_dir: str, id_col: str = 'id', sizes: list[int] | tuple[int, ...] = (25,40,60)) -> pd.DataFrame:
    df = pd.read_csv(in_csv).copy()
    roi_root = Path(roi_dir)
    for size in sizes:
        col = f'roi_path_{int(size)}_sphere'
        paths = []
        for pid in df[id_col].astype(str):
            pid_clean = pid[:-2] if pid.endswith('.0') else pid
            p = roi_root / f'{pid_clean}_{int(size)}mm_sphere.npy'
            paths.append(str(p) if p.exists() else '')
        df[col] = paths
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df
