# 追加で必要なコード

## 必須
- `src/chole_predict/training/train_residual.py` の本実装
- `src/chole_predict/training/train_gated.py` の本実装
- `src/chole_predict/training/roi_autogen_run.py` の本実装
- `scripts/prepare_analysis_table.py` あるいは `src/chole_predict/data/path_injection.py`
- `src/chole_predict/analysis/oof_merge.py`
- `scripts/make_main_figures.py`
- `scripts/make_supplementary_figures.py`

## 推奨
- `src/chole_predict/io/dicom_io.py` と ROI autogen 側 DICOM reader の統合
- `src/chole_predict/models/heatmap_utils.py`
- `src/chole_predict/models/roi_localizer.py`
- `src/chole_predict/qa/roi_autogen_qa.py`
- pytest による smoke test


## 2026-03-18 migration update
- `training/train_residual.py` now contains a direct migrated version of the original residual training script.
- `training/train_gated.py` now contains a direct migrated version of the original gated residual training script.
- `training/roi_autogen_run.py` now contains a direct migrated version of the original CT-only ROI auto-generation script.
- These three files are intentionally self-contained first-pass migrations. The next cleanup step is to replace duplicated helpers with imports from `data/`, `models/`, `roi/`, and `utils/`.


## roi_autogen_run.py split status
- Split into io/dicom_io.py, roi/{canonicalize,vw_json,geometry,crop,anchor}.py, models/roi_localizer.py, qa/roi_autogen_qa.py, data/{id_utils,case_schema,case_loader}.py, training/{roi_autogen_data,roi_autogen_train,roi_autogen_infer}.py, analysis/roi_eval.py.
- `training/roi_autogen_run.py` now keeps orchestration (`reqa_missing`, `run_cv`, `main`) and imports the lower-level modules.
- Next cleanup step would be moving `reqa_missing` and `run_cv` into separate modules and reducing repeated CLI argument definitions.


## 2026-03 migration update
- Added legacy-preserving wrappers for ROI autogen, sphere ROI generation, and full pipeline.
- Added YAML configs matching the current production commands.
- Added prepare_analysis_table.py for injecting roi_path_25/40/60 columns into the analysis CSV.
- The modular `roi_autogen_run.py` remains a scaffold; exact current reproducibility is preserved via the legacy backend.
