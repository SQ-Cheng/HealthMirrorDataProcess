# Frozen DINOv3 + Native ROI Color Regression

Ten independent regressors, inheriting the current Exp2 24h labels, seeds,
and per-analyte patient-disjoint splits. No rematching or split search.
Existing DINO-only and DINO+FaRL results are not overwritten. The launch
script also replaces Exp1's original-only ROI results with five-view results.

## Architecture

```text
Full-face frame/view -> frozen DINOv3 ViT-S/16 -> 384-d normalized CLS
Same source frame/view -> four native ROI masks -> 4 x 41 color descriptors
 -> training-only color StandardScaler
Concat(384 + 164 = 548)
 -> Linear(548,32) + LayerNorm(32) + SiLU + Dropout(0.25)
 -> Linear(32,1) -> robust-scaled raw laboratory value
```

The head has **17,665 trainable parameters**; the color-normalization buffers
have 328 scalar entries, not trainable parameters. The encoder is never updated.
All pixels, landmarks and DINO caches are reused from native224 processing.

## Data And Training

- Original 20 nonadjacent candidate frames per video; require at least 10
  valid for all four ROIs across all five views. Keep actual valid frames,
  with no replacement, padding, interpolation or missing-ROI imputation.
- Five training views: original, horizontal flip, center crop (0.9),
  brightness (x1.06), contrast (x1.08 around full-face per-channel means).
- ROI extraction is geometrically equivalent under each transform. Flip
  swaps image-left/right cheeks. Center color statistics select the same
  202x202 native center window that DINO resizes to 224; ROI statistics are
  not spatially resized and thus have no bicubic interpolation artifacts.
- Each `(source frame, view)` color vector is paired with that same DINO
  vector, not a video-averaged color vector copied across individual frames.
- Training batches contain up to 12 different clinical events, one
  video/view per event, all its accepted frames: up to 240 feature vectors.
  Every training video/view/frame occurs once per epoch; partial batches stay.
- Frame SmoothL1 beta=0.5, weighted by inverse accepted-frame count, so
  each video/view contributes equal total mass despite differing ROI validity.
- Fit color scaling on effective training frames/views only, with equal
  video mass; target median/IQR on effective training videos only.
- AdamW lr=2e-4, weight decay=1e-3; cosine minimum 1e-6; maximum 80 epochs;
  validation raw-video-MAE early-stopping patience 12; clip 1; AMP; no compile.
- Validation/test use original views only. Average accepted frame outputs
  per video and inverse-scale predictions; truth is the original raw value.
- Tasks: O2Hb fraction, lactate, urea, total bilirubin, platelets, Hb,
  A/a PO2 ratio, creatinine, direct HCT, direct eGFR.

## Comparison And Outputs

An existing frozen DINO-only head32 is re-evaluated on exactly the same
eligible original test frames/videos. This matches inference data, not
training cohorts: the old DINO model trained on full 20-frame videos, whereas
fusion training excludes ROI-invalid frames/videos. This distinction is
recorded in `outputs/experiment_manifest.json` and `paired_test_audit.csv`.
ROI-only uses video/view feature averaging, whereas fusion uses frame loss;
these are not strictly loss-controlled architecture ablations.

Automatic outputs: checkpoints with color and target scalers; frame-index
audit; continuous train/validation loss, MAE/r and learning-rate histories;
video predictions and raw metrics; PNG histories, scatter with identity
and fit lines, and MAE/RMSE/r/R2/explained-variance comparison bars.

```bash
bash study/exp2_face_dinov3_roi_color_regression/launch_screen.sh
screen -r exp1_dino_roi_views5
```

Preparation runs once, then a real-data one-epoch GPU smoke test. Exp1's 120
ROI models and the ten fusion heads run concurrently on four GPUs (small
cached-feature head jobs; no duplicated GPU encoders). Both generate figures
automatically. Logs are `study/exp1_roi_color_regression/logs/run.log` and
`study/exp2_face_dinov3_roi_color_regression/logs/run.log`; each experiment
writes its own `outputs/COMPLETE` only after successful result plots.
