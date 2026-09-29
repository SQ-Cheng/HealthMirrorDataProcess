# Exp8: RGB-Estimated Visible Spectra to Laboratory Values

This is an exploratory **two-step estimation**, not a calibrated hyperspectral
measurement. First, the frozen official MST++ checkpoint trained on NTIRE 2022
estimates a 31-band visible-spectrum cube (400-700 nm, 10-nm spacing) from each
RGB face frame. Second, an independent small regressor for each target predicts
the raw hemoglobin, total bilirubin, or lactate value from the estimated
spectrum. No local paired RGB/spectral ground truth is available, so spectral
fidelity on these patient videos cannot be measured. Camera spectral response,
white balance, JPEG compression, lighting, and the natural-scene-to-face domain
shift limit physical interpretation. The curves are **model-estimated normalized
responses**, not measured reflectance or clinical spectrometry.

## Source And Matching

The experiment reuses the corrected Exp2 face-regression task records and
compact MJPEG byte-offset index. These records already apply 24-hour,
same-hospitalization nearest-lab matching and a fixed patient-disjoint split.
Each task retains exactly its original video IDs, raw values and train-only
median/IQR scaler. There are 20 deterministic nonadjacent frames per video.
One video is one independent supervised example: its 20 spectral features are
averaged before the regression head. Targets and current record counts are:

| Target | Videos (train/val/test) | Patients |
|---|---:|---:|
| Hemoglobin, g/L | 1152 / 379 / 366 | 702 |
| Total bilirubin, umol/L | 441 / 148 / 144 | 327 |
| Lactate, mmol/L | 803 / 264 / 269 | 620 |

## Spectral Stage

The official MST++ inference and training code min-max normalizes each RGB
image independently; Exp8 uses the same preprocessing on the cached 128x128
full-color face crops. It does **not** apply ImageNet normalization, JPEG
recompression, brightness jitter or contrast jitter. The frozen 1,619,625-
parameter MST++ produces a 31x128x128 estimate. Values are clipped to [0,1]
as in the original prediction code. The central 96x96 pixels are pooled to a
31x4x4 spatial-spectral grid per frame. Only this approximately 38-MB float16
feature cache is persisted across all three tasks, not full spectral cubes.
Extraction is identical for train, val and test, and uses no lab labels.

The MST++ source is vendored verbatim under `vendor/` with its MIT license and
the pinned upstream commit in `weights_manifest.json`. The verified checkpoint
is located at `study/common/pretrained_weights/mst_plus_plus_ntire2022.pth`
(SHA-256 in the manifest). On a new machine, obtain it from the author's public
Google Drive model zoo with:

```bash
python -m pip install gdown
python -m gdown 18X6RkcQaIuiV5gRbswo7GLv7WJG9M_WM -O study/common/pretrained_weights/mst_plus_plus_ntire2022.pth
```

## Regression Stage

For each target, the 20 frame grids are averaged to one 31x4x4 video tensor.
The per-target head is `Flatten(496) -> LayerNorm -> Linear(496,64) -> SiLU
-> Dropout(0.2) -> Linear(64,32) -> SiLU -> Linear(32,1)`. MST++ stays frozen.
The head predicts the previously saved training-only robust-scaled raw value.
Training uses SmoothL1(beta=0.5), AdamW(lr=1e-3, weight_decay=1e-2), batch 32,
cosine decay to 1e-5, at most 160 epochs, and patience 20 on raw-unit
validation MAE. Validation/test labels never select weights or fit scaling.
The test set is evaluated once with the best validation checkpoint. Results
include per-video predictions, train/val history, MAE/RMSE/R2/Pearson, and
automatic figures comparing the saved RGB EfficientNet-B0 control.

The RGB control uses the same splits and labels, but a different training
unit and augmentation policy, so this is a method comparison rather than an
isolated causal ablation of the spectral transform.

## Start After Protocol Review

The NTIRE 2022 run is complete and retained in `outputs/`, `cache/`, and
`logs/`. To compare against an MST++ checkpoint genuinely retrained on the
Hyper-Skin **RGB-to-VIS** pair, place that checkpoint at
`study/common/pretrained_weights/mst_plus_plus_hyperskin_rgb_vis.pth` and run:

```bash
bash study/exp8_rgb_spectral_lab_regression/launch_screen.sh hyperskin
```

The skin-domain run uses `outputs_hyperskin/`, `cache_hyperskin/`, and
`logs_hyperskin/`; it does not overwrite the NTIRE run. The checkpoint is
rejected if it is byte-identical to the NTIRE model. After training, the
`spectral_source_comparison.png` figure compares both test results after
verifying patient IDs, video IDs, and raw labels match. Hyper-Skin's public
repository provides retraining code, but data access requires its EULA and
no skin-retrained checkpoint was found in its public model links. The
checkpoint must be obtained legitimately or trained on authorized data.

## Spatial Grid Ablation

To compare the original 31x4x4 feature grid with a 31x16x16 grid using the
same NTIRE checkpoint, task records, frames, split, optimizer, and training
schedule:

```bash
bash study/exp8_rgb_spectral_lab_regression/launch_screen.sh ntire2022 16
```

Each 16x16 cell averages a 6x6 patch of the 96x96 central spectral image.
The video feature is 7,936 values; the head's first linear layer therefore
grows to match, while its hidden widths remain 64 and 32. The new cache,
log, and model results are kept separately in `cache_grid16/`,
`logs_grid16/`, and `outputs_grid16/`. A paired test-label check precedes
`outputs_grid16/figures/feature_grid_comparison.png` and the machine-readable
`outputs_grid16/reference_comparison.csv`.
