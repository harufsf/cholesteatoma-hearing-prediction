# FIGURE_AND_INTERPRETABILITY_LAYOUT.md

## Recommended placement

- `src/chole_predict/plotting/`
  - `figure2_primary_ac.py`
  - `figure3_abg_leq20.py`
  - `supp_gate_behavior.py`
  - `supp_error_distribution_primary_ac.py`
- `src/chole_predict/qa/`
  - `pred_center_manuscript_qa.py`
- `src/chole_predict/interpretability/`
  - `gradcam_case_selection.py`
  - `gradcam_generate.py`
  - `gradcam_where_ratio.py`
- `scripts/`
  - thin CLI wrappers only

## What still deserves later refactoring

1. `gradcam_generate.py` still imports legacy training scripts directly. This preserves exact architecture matching during migration, but should later be switched to `src/chole_predict/models/...` and `src/chole_predict/training/...` imports.
2. Shared plotting utilities like spine styling, grayscale conversion, frequency-column inference, and output naming can be factored into `src/chole_predict/plotting/common.py`.
3. `pred_center_manuscript_qa.py` and `make_sphere_roi.py` share DICOM/JSON resolution helpers and can later share a common `qa_center_io.py` or `pred_center_io.py` module.
4. Grad-CAM metrics and figure generation can later be split into `metrics.py` and `plots.py` under `interpretability/`.


## Additional organization updates

- Preferred Grad-CAM end-to-end runner: `src/chole_predict/interpretability/gradcam_run_and_panel.py` with wrapper `scripts/run_resid_gradcam_and_panel.py`.
  This runner shells out to the existing Grad-CAM generator and panel builder, forwards extra Grad-CAM arguments after `--`, and supports anonymized panel labels.
- Added supplementary plotting modules:
  - `src/chole_predict/plotting/supp_yyplot_primary_ac.py`
  - `src/chole_predict/plotting/supp_roi_center_error_hist_cdf.py`
- Added configs:
  - `configs/gradcam_run_and_panel.yaml`
  - `configs/supp_yyplot_primary_ac.yaml`
  - `configs/supp_roi_center_error_hist_cdf.yaml`


## plotting/common.py

Added shared helpers for figure scripts, including output-path resolution, PNG/PDF saving, primary AC column builders, CDF computation, common frame styling, and axis-limit utilities. This reduces repeated plotting boilerplate across Fig2, Fig3, yy-plot, ROI-center-error hist/CDF, and gate-behavior supplementary figures.


## Final manuscript figure mapping (current)
- Fig. 2: `figure2_primary_ac.py`
- Fig. 3: `figure3_abg_leq20.py`
- Fig. S1: `figS1_roi_localization_and_error.py` (assembled from three QA panels + ROI-center error CSV)
- Fig. S2: `supp_yyplot_primary_ac.py`
- Fig. S3: `figS3_cdf_primary_ac.py`
- Fig. S4: `run_resid_gradcam_and_panel.py` / `gradcam_run_and_panel.py`
- Fig. S5: `gradcam_where_ratio.py`
- Fig. S6: `supp_gate_behavior.py`

### Not in the current final figure order
- `supp_error_distribution_primary_ac.py`: kept as an optional exploratory / reviewer-response figure, not mapped to the current final numbering.
- `supp_roi_center_error_hist_cdf.py`: kept as a reusable component script; the current final manuscript integrates these panels into Fig. S1 rather than using it as a standalone figure.
