# cholesteatoma-hearing-prediction

Code for multimodal prediction of postoperative hearing outcomes after tympanoplasty for cholesteatoma using clinical variables and temporal bone CT-derived ROI inputs.

This repository is organized for reproducible training, out-of-fold evaluation, and manuscript figure generation.

## Overview

This project provides a modular workflow for:

1. ROI-center prediction from temporal bone CT
2. Spherical ROI generation from predicted centers
3. ROI path injection into the analysis table
4. Training of tabular-only, residual, and gated multimodal models
5. Aggregation of out-of-fold predictions for downstream analysis and figure generation

The codebase is structured to support manuscript-oriented reproduction runs while keeping public usage relatively simple.

## Repository layout

- `src/chole_predict/` — core library code
- `scripts/` — command-line entry points
- `configs/` — YAML configuration files
- `docs/` — reproducibility notes and supporting documentation
- `tests/` — lightweight tests and smoke checks

## Installation

We recommend using a clean Python environment.

```bash
pip install -r requirements.txt
pip install -e .
```
## Recommended usage

We recommend running the pipeline with version-controlled YAML configuration files so that all model settings are recorded in a single place.

### Full multimodal pipeline

```bash
python scripts/run_full_pipeline.py --config configs/full_pipeline_repro_oldrun.yaml
```

### Current production-equivalent workflow

```bash
python scripts/run_roi_autogen.py --config configs/roi_autogen_user.yaml
python scripts/make_sphere_roi.py --config configs/sphere_roi_user.yaml
python scripts/run_full_pipeline.py --config configs/full_pipeline_repro_oldrun.yaml
```

Using named configuration files helps distinguish manuscript-oriented reproduction runs from smoke tests and exploratory runs.

## Configuration files

Representative configuration files in configs/ include:

- configs/full_pipeline_repro_oldrun.yaml — manuscript-oriented reproduction settings
- configs/full_pipeline_user.yaml — user-edited local workflow settings
- configs/roi_autogen_user.yaml — ROI auto-generation settings
- configs/sphere_roi_user.yaml — spherical ROI generation settings

Configuration files should be adapted to local file locations and approved local datasets.

## Outputs

The full pipeline writes outputs under:

```text
experiments/<experiment_name>/
```

Typical outputs include out-of-fold prediction files for the tabular-only, residual, and gated models, along with aggregated analysis tables used for downstream plotting.

## Manuscript figures

The repository is organized so that each manuscript figure has a corresponding entry-point script under `scripts/` and an implementation module under `src/chole_predict/`.

### Main figures

- **Fig. 2** Primary endpoint: postoperative AC thresholds (OOF)  
  Script: `scripts/make_fig2_primary_ac.py`
- **Fig. 3** Key secondary endpoint: ABG ≤ 20 dB  
  Script: `scripts/make_fig3_abg_leq20.py`

### Supplementary figures

- **Fig. S1** ROI localization examples and ROI-center error summary  
  Script: `scripts/make_figS1_roi_localization_and_error.py`
- **Fig. S2** Observed vs predicted postoperative AC PTA mean (OOF yy-plot)  
  Script: `scripts/make_figS2_yyplot_primary_ac.py`
- **Fig. S3** CDF of absolute error for postoperative AC PTA mean  
  Script: `scripts/make_figS3_cdf_primary_ac.py`
- **Fig. S4** Representative RESID Grad-CAM overlays stratified by error change versus Tabular  
  Script: `scripts/make_figS4_gradcam_overlays.py`
- **Fig. S5** Quantitative spatial metrics of RESID Grad-CAM (where-ratio panel)  
  Script: `scripts/make_figS5_gradcam_where_ratio.py`
- **Fig. S6** Gate behavior analysis  
  Script: `scripts/make_figS6_gate_behavior.py`

### Figure-related source layout

- `src/chole_predict/plotting/` contains plotting modules for main and supplementary figures.
- `src/chole_predict/qa/` contains manuscript-oriented ROI localization QA utilities.
- `src/chole_predict/interpretability/` contains Grad-CAM case selection, generation, and quantitative analysis utilities.
- `configs/` contains example YAML files for figure generation.

## Data and path handling

This public repository does not bundle patient-level raw CT data, clinical source tables, or derived intermediate files generated during the original study workflow.

Figure scripts and analysis scripts may expect locally available files such as:

- `per_patient_results.csv`
- `pred_center_errors.csv`
- Grad-CAM intermediate outputs

YAML files under `configs/` are examples and should be adapted to local file locations and approved local datasets.

## Reproducibility notes

- Use the YAML configuration files in `configs/` to preserve run settings.
- Keep manuscript-oriented runs, smoke tests, and exploratory runs in separate configuration files.
- Version control both code and configuration files used for a reported experiment.
- For manuscript reproduction, prefer fixed seeds and explicit hyperparameter settings over implicit defaults.

## Scope of the public release

This repository is intended to provide the code required to reproduce the computational workflow on appropriately structured local data.

Items that may remain outside the public release include:

- patient-level raw imaging and clinical data
- protected health information
- institution-specific data extraction layers
- selected large intermediate outputs not suitable for repository distribution

## Notes for users

- Paths in YAML configuration files should be updated for local environments.
- Some scripts retained for exploratory analysis or reviewer-response material are not mapped to manuscript figure numbers unless explicitly listed above.
- When in doubt, start from a smoke-test configuration and then switch to the manuscript-oriented reproduction configuration.

## Citation and availability

If you use this repository in academic work, please cite the associated manuscript once available.

A manuscript-oriented code availability statement can point readers to this repository and clarify that raw patient-level data are not publicly released because of ethical, legal, and privacy restrictions.
