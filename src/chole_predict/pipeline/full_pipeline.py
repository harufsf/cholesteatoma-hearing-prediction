from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from chole_predict.data.path_injection import add_roi_paths_to_csv
from chole_predict.training.train_gated import run_train_gated
from chole_predict.training.train_residual import run_train_residual
from chole_predict.training.train_tabular import run_train_tabular
from chole_predict.utils.parsing import parse_csv_list, parse_int_csv

DEFAULT_TARGETS = [
    'post_PTA_0.5k_A', 'post_PTA_1k_A', 'post_PTA_2k_A', 'post_PTA_3k_A',
    'post_PTA_0.5k_B', 'post_PTA_1k_B', 'post_PTA_2k_B', 'post_PTA_3k_B',
]


def run_full_pipeline(
    in_csv: str,
    out_csv: str,
    roi_dir: str,
    experiment_name: str,
    sizes: str | Iterable[int] = '25,40,60',
    target_cols: str | Iterable[str] | None = None,
    run_tabular: bool = True,
    run_residual: bool = True,
    run_gated: bool = True,
    config: dict[str, Any] | None = None,
) -> None:
    config = config or {}
    common_cfg = config.get('common', {})
    tab_cfg = config.get('tabular', {})
    resid_cfg = config.get('residual', {})
    gated_cfg = config.get('gated', {})

    out_root = Path('experiments') / experiment_name
    out_root.mkdir(parents=True, exist_ok=True)

    size_list = parse_int_csv(sizes) if isinstance(sizes, str) else [int(x) for x in sizes]
    roi_cols = [f'roi_path_{s}_sphere' for s in size_list]
    tcols = parse_csv_list(target_cols) if target_cols is not None else list(DEFAULT_TARGETS)

    add_roi_paths_to_csv(in_csv=in_csv, out_csv=out_csv, roi_dir=roi_dir, sizes=size_list)

    if run_tabular:
        run_train_tabular(
            csv_path=out_csv,
            target_cols=tcols,
            out_prefix=str(out_root / 'TAB'),
            id_col=common_cfg.get('id_col', 'id'),
            fold_col=common_cfg.get('fold_col', 'fold'),
            seed=common_cfg.get('seed', 42),
            batch_size=tab_cfg.get('batch_size', 32),
            epochs=tab_cfg.get('epochs', 300),
            lr=tab_cfg.get('lr', 1e-3),
            val_size=common_cfg.get('val_size', 0.2),
            num_workers=common_cfg.get('num_workers', 0),
            use_amp=not common_cfg.get('no_amp', False),
        )

    if run_residual:
        run_train_residual(
            csv=out_csv,
            target=tcols,
            roi_cols=roi_cols,
            out_prefix=str(out_root / 'RESID'),
            id_col=common_cfg.get('id_col', 'id'),
            fold_col=common_cfg.get('fold_col', 'fold'),
            side_col=common_cfg.get('side_col', 'side'),
            roi_pool=resid_cfg.get('roi_pool', 'concat'),
            out_dhw=resid_cfg.get('out_dhw', '160,192,192'),
            tab_epochs=resid_cfg.get('tab_epochs', 300),
            tab_batch=resid_cfg.get('tab_batch', 32),
            tab_lr=resid_cfg.get('tab_lr', 1e-3),
            roi_epochs=resid_cfg.get('roi_epochs', 150),
            roi_batch=resid_cfg.get('roi_batch', 2),
            roi_lr=resid_cfg.get('roi_lr', 1e-3),
            delta_l2=resid_cfg.get('delta_l2', 0.0),
            val_size=common_cfg.get('val_size', 0.2),
            seed=common_cfg.get('seed', 42),
            start_fold=resid_cfg.get('start_fold', 1),
            num_workers=common_cfg.get('num_workers', 0),
            no_amp=common_cfg.get('no_amp', False),
        )

    if run_gated:
        run_train_gated(
            csv=out_csv,
            target=tcols,
            roi_cols=roi_cols,
            out_prefix=str(out_root / 'GATED'),
            id_col=common_cfg.get('id_col', 'id'),
            fold_col=common_cfg.get('fold_col', 'fold'),
            side_col=common_cfg.get('side_col', 'side'),
            roi_pool=gated_cfg.get('roi_pool', 'concat'),
            out_dhw=gated_cfg.get('out_dhw', '160,192,192'),
            tab_epochs=gated_cfg.get('tab_epochs', 300),
            tab_batch=gated_cfg.get('tab_batch', 32),
            tab_lr=gated_cfg.get('tab_lr', 1e-3),
            roi_epochs=gated_cfg.get('roi_epochs', 150),
            roi_batch=gated_cfg.get('roi_batch', 2),
            roi_lr=gated_cfg.get('roi_lr', 1e-3),
            gate_use_ytab=gated_cfg.get('gate_use_ytab', False),
            lambda_gate_l1=gated_cfg.get('lambda_gate_l1', 1e-3),
            lambda_delta_l2=gated_cfg.get('lambda_delta_l2', 1e-4),
            val_size=common_cfg.get('val_size', 0.2),
            seed=common_cfg.get('seed', 42),
            start_fold=gated_cfg.get('start_fold', 1),
            num_workers=common_cfg.get('num_workers', 0),
            no_amp=common_cfg.get('no_amp', False),
        )
