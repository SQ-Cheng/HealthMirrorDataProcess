# Exp1: Local Face Color And Laboratory Regression

Status: **full-cohort native41 extraction and all 120 models completed**.
Actual progress is in `logs/run.log` and `outputs/tables/run_index.csv`;
`outputs/COMPLETE` marks successful training, bootstrap and figure completion.
This replaces the old ECG/SQA Exp1, not the current Exp2 or Exp6 experiments.
Irreplaceable human ECG annotations are preserved at
`study/common/annotations/ecg_sqa/`; obsolete code, models and results are removed.

## Goal And Reference Data

Predict individual raw laboratory values from local face color. Compare a
compact MLP and ridge regression on each local region and combined regions.
Every analyte/model/ROI configuration has an independent fitted model.

- Reference: `../exp2_face_pretrained_head32_regression/outputs/20frame_face224/`.
- Ten tasks: oxyhemoglobin fraction, lactate, urea, total bilirubin, platelets,
  hemoglobin, A/a PO2 ratio, creatinine, directly reported HCT and eGFR.
- Reuse exact current per-analyte labels and patient-disjoint train/val/test
  assignments; do not search another split or rematch laboratory events.
- Clinical matching remains the main same-admission nearest-assay within 24h,
  using validated session timestamps and the existing analyte/unit policies.
- Input: native224 RGB FFV1 crops, with the existing twenty selected
  nonadjacent frames per video and original source-frame/timestamp mappings.
- Before ROI quality filtering: 1,391 unique videos and 563 unique patients
  across the ten tasks. Analyte-specific counts differ. Final counts cannot
  be stated until ROI extraction is audited.

The primary statistical unit is **one video**, not twenty independent lab
samples. The twenty frames provide repeated color measurements for that video.
This has the same labels/test unit as Exp2 but a video-feature-level training
objective rather than the main Exp2 frame loss; it is not a loss-controlled
CNN architecture ablation.

## Landmark Detection And ROI Definitions

Use existing MediaPipe 0.10.32 and the official local `face_landmarker.task`
under `preprocess/face_crop_comparison/models/`. Use `FaceLandmarker.detect`
in IMAGE mode because the twenty selected frames are nonconsecutive and may
have gaps after crop filtering. Each frame is treated independently.

Parameters: `num_faces=1`, minimum face-detection and face-presence confidence
0.5, CPU inference, blendshapes and transformation matrices disabled. Existing
upstream crop acceptance (BlazeFace score >=0.75) remains unchanged. The
landmarker thresholds are distinct from that detector score; landmark output
does not provide a usable calibrated per-ROI confidence score.

| ROI | Native geometric mask |
|---|---|
| Forehead | Central forehead bounded laterally by inner temple positions and below by the upper eyebrow envelope, shifted upward by 3 native pixels |
| Image-left cheek | Landmark polygon between lower eyelid, outer face contour, nose side and mouth side |
| Image-right cheek | Symmetric image-right polygon |
| Lips | Outer lip polygon minus the inner mouth-opening polygon |

**Forehead upper edge is always y=0, the video top boundary.** Do not use the
mesh's upper face-oval point as a forehead/hairline cutoff. Use the central
70% of the temple span to reduce lateral hair/background, without moving the
upper edge down. Landmark index lists are fixed in `roi_schema.json` and were
approved after the 100-frame overlay review. Left/right names refer to image
coordinates, not assumed anatomy. The approved geometry has not changed.

Exclude eyelids, eyebrows, nostrils and non-cheek areas geometrically; cheek
masks are inset by 2 native pixels. Exclude teeth/tongue/oral cavity from the
lip mask. Do not threshold skin by RGB/HSV color: doing so could remove the
very color changes being studied. Hair, beard, occlusion and highlights will
be included in visual QA and retained-pixel diagnostics, not silently removed
by a phenotype-dependent skin-color rule.

No additional alignment or Kalman filter is introduced. The current224 crops
already use Kalman box smoothing and **no alignment**; size normalization is
not pose alignment or photometric calibration.

## Native Pixels And Quality

Features are computed **directly from original224 pixels selected by the
native ROI mask**. No crop resizing, interpolation, fixed-size canvas or
padding occurs in the feature path. Mouth holes and non-ROI pixels never
enter statistics. Preview panels may scale images for human inspection only;
saved preview pictures must not be used as feature inputs. The previously
approved 100 preview images are preserved, along with their original protocol
and geometry snapshots; their old display dimensions do not affect this
native-pixel baseline.

Reject a frame/ROI when landmarks are absent or nonfinite, its polygon is
invalid, more than 10% of the proposed polygon area lies outside the image,
or the original valid mask contains fewer than 128 pixels (skin regions)
or 64 pixels (lips). Record each cause explicitly. Never replace a failed
local ROI with the whole face or propagate a stale landmark across a gap.

Primary ROI comparisons use videos with at least ten of the twenty frames
valid for **all four** ROIs. All configurations use exactly these common
valid frames and the same videos/splits. Keep per-ROI validity and features
for other videos as an audited cache, rather than erase potentially useful
data; report those excluded counts. Do not run additional maximum-cohort
models in the initial study or conflate different cohorts in the main ranking.

## Color Features: 41 Per ROI

Compute from RGB float32 in [0,1], not ImageNet-standardized pixels. Convert
explicitly from RGB (not BGR) to float HSV and CIELAB with OpenCV. These are
camera RGB descriptors, not calibrated reflectance or hemoglobin spectra.

| Family | Definition | Dimensions |
|---|---|---:|
| RGB statistics | Mean, population SD, p10/p50/p90 for each R/G/B channel | 15 |
| Lab statistics | Mean, population SD, median for each L/a/b channel | 9 |
| HSV S/V statistics | Mean and population SD for S and V | 4 |
| Circular hue | Saturation-weighted mean sine, mean cosine and concentration; do not arithmetically average angles | 3 |
| Normalized chromaticity | Mean/SD of r=R/(R+G+B), g=G/(R+G+B); black pixels use zero | 4 |
| Log channel ratios | Mean/SD of log((R+eps)/(G+eps)), log((B+eps)/(G+eps)); eps=1/255 | 4 |
| Exposure fractions | All RGB <=5/255; any RGB >=250/255 | 2 |
| **Total** | No histograms or additional channel correlations | **41** |

The ordered names and formulas are saved in `feature_schema.json`. Floating
HSV hue is converted from degrees to radians. Saturation-weighted sine and
cosine are averaged; concentration is their vector magnitude. Achromatic
regions use zero for all three hue features when total saturation is negligible.
Exposure diagnostics remain features; dark or bright pixels are not discarded.

For each video/ROI, average the 41-dimensional descriptors over the common
valid frames with equal frame weights. This avoids treating duplicate labels
as independent measurements. Do
not add shape, pose, identity, timestamps or clinical history to the model.
No brightness/contrast augmentation, white balance correction, histogram
equalization, color matching or target-informed feature selection is used.

Validation reused the saved landmarks and original source frames from the
approved 100-frame preview, without another MediaPipe pass. All 369 accepted
ROI vectors are finite, all native pixel counts and geometry acceptance flags
match the reviewed output, and 90 frames remain valid for all four ROIs.
The detailed CSV/NPZ and report are under
`outputs/tables/native41_preview_validation/`. Invalid ROIs retain explicit
validity flags and NaN features; they are not imputed or silently dropped.

```bash
python -m unittest study.exp1_roi_color_regression.test_roi study.exp1_roi_color_regression.test_features
python -m study.exp1_roi_color_regression.validate_features
```

## Six ROI Configurations And Models

Configurations: forehead; image-left cheek; image-right cheek; lips; joint
left+right cheeks; joint forehead+left cheek+right cheek+lips. Joint features
are concatenated, preserving regional differences rather than pooling all
ROI pixels. Dimensions: 41 (single), 82 (cheeks), 164 (all).

For ten analytes x six configurations x two regressors: **120 models**.
MLP and ridge use identical common-cohort video features and labels.

### MLP

```text
Training-only standardized feature vector (D=41/82/164)
 -> Linear(D,64) -> SiLU -> Dropout(0.2)
 -> Linear(64,32) -> SiLU -> Dropout(0.2)
 -> Linear(32,1)
 -> train-only median/IQR-scaled raw laboratory value
```

Trainable parameters: 4,801 / 7,425 / 12,673 for the three input dimensions.
AdamW lr=5e-4, weight decay=1e-3, batch size=64 videos, max epochs=150,
validation raw-unit MAE early stopping patience=20, gradient clip=1.
SmoothL1 beta=0.5, unweighted video loss. ReduceLROnPlateau monitors val MAE,
factor=0.5, patience=5, minimum lr=1e-5. Use deterministic per-task seeds
derived from the existing reference, and keep them fixed across ROI modes.
No pretraining, second stage or neural-network hyperparameter search.

### Ridge

`StandardScaler -> Ridge(fit_intercept=True, solver="svd")`, with
alpha in `[1e-4,1e-3,1e-2,1e-1,1,10,100,1000,10000]`. Select alpha using
five-fold **patient-grouped CV inside the training split**, minimizing
mean raw-unit MAE. Fit feature/target transforms separately inside each
inner training fold. Refit on the full valid training set after selection.
Neither the outer test set nor val set chooses alpha. Outer val is reported.

Both regressors fit feature standardization and target median/IQR only from
their actual effective training cohort. Constant features have scale=1;
nonfinite features cause a recorded error, not silent sample deletion.

## Outputs And Execution Boundaries

Planned layout:

```text
study/exp1_roi_color_regression/
  README.md, protocol.json, roi_schema.json, feature_schema.json
  cache/                 # landmark/ROI audit and compact numeric features
  logs/                  # extraction/training logs
  outputs/tables/        # cohorts, failures, features schema, metrics, predictions
  outputs/models/        # MLP checkpoints and fitted ridge/transforms
  outputs/figures/       # ROI overlays, feature distributions and result PNG/PDF
```

Cache one extraction pass per source video; do not copy videos or save every
ROI image. About 18.3 MB stores all 27,820 x4 x41 float32 frame features
before exclusions, plus masks/landmark audit metadata. Save a small, explicitly
selected set of ROI previews for manual geometry review. Cache keys include
native video/sidecar fingerprints, frame index, MediaPipe weight hash and ROI/
color schema versions. Future extraction uses a bounded CPU worker pool;
MLPs can use GPU task scheduling, ridge and preprocessing stay on CPU.

Report video-level MAE/RMSE/Pearson r/Spearman rho/R2/explained variance,
per-split counts and original clinical-event counts. Compare against the
training-median constant predictor. Primary plots: ROI overlay QA; inclusion
counts by ROI/analyte/split; grouped-bar ROI x regressor performance;
held-out predicted-vs-measured scatter with identity and linear fit; MLP
training loss and separately scaled correlation curves. Use four columns
for the ten-analyte panels, hiding unused axes. Statistical intervals use
patient-cluster bootstrap (1,000 resamples) for held-out MAE/r/R2, without
tuning on test or treating videos as independent patients.

The pipeline uses a six-worker CPU extraction pool, four GPU MLP slots and
two CPU ridge slots. Completed contract-matching jobs are reused on relaunch.
Full extraction retained 1,343 of the 1,391 source videos (48 failed the common
native-ROI rule); per-analyte/split counts and excluded IDs are in `outputs/tables`.
No existing Exp2 source records or patient split files are modified.

```bash
bash study/exp1_roi_color_regression/launch_screen.sh
screen -r exp1_roi_color_regression
```

The immutable `protocol.json` records the approved design, including its
initial pre-training status; runtime/completion is represented by output
artifacts and logs, not by changing the extraction-cache configuration.

## Primary References

- https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python
- https://docs.opencv.org/4.13.0/de/d25/imgproc_color_conversions.html
- https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
- https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html
