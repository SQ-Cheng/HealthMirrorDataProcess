# Exp2 Raw-Video Head32 Raw-Value Regression

Canonical IDs:

- `exp2_raw_video_20frame_head32_regression_balanced_split`
- `exp2_raw_video_allframes_head32_regression_balanced_split`

Both variants live in this directory. Versioned artifacts are stored under
`outputs/20frame` and `outputs/allframes`; logs use the same names under `logs`.
The source builder enumerates every raw `video.avi`, maps its hospital ID, and reads
the capture interval from the corresponding frame-level `video.avi.ts`. It does not
use cleaned ECG/rPPG session CSV files.

## Data Policy

- Maximum video-lab interval distance: 24 hours.
- One label per video and target: choose the nearest valid measurement by interval
  distance, then video-midpoint distance, then report timestamp.
- The retained 20-frame variant predicts `hemoglobin_low`, `po2_low`, and
  `oxyhemoglobin_fraction` from 20 deterministic non-adjacent frames sampled
  from 5% through 95% of each video.
- The all-frame variant predicts `hemoglobin_low`, `po2_low`, and `lactate_high`
  from every decodable frame in each video.
- Training views: original, horizontal flip, 90% center crop, brightness +6%, and
  contrast +8%.
- Validation and test use every frame selected by the variant with only the
  original view.
- PO2 uses only the exact source item `氧分压`; temperature-corrected PO2 is
  excluded before video-lab matching.
- Oxyhemoglobin fraction uses only exact `氧合血红蛋白分数` values reported in
  percent, excludes explicitly venous specimens, and enforces the physical
  range 0-100%.

Each variant contains its own source audit and compact JPEG byte-offset index.
Decoded images are never persisted.

## Model And Optimization

Each architecture/target pair has an independent model:

- MobileNetV3-Small and EfficientNet-B0 with local ImageNet pretrained weights.
- Classification head replaced by
  `Linear -> LayerNorm -> SiLU -> Dropout -> Linear(1)`, hidden width 32.
- Stage 1 freezes the backbone and trains the head at `2e-4`.
- Stage 2 unfreezes the full model and fine-tunes at `1e-5`.
- Both stages minimize unweighted SmoothL1 loss (`beta=0.5`) on robust-scaled raw
  values with early stopping on inverse-transformed validation video-level MAE.

Each target scaler is fitted only on training videos:
`scaled = (raw value - training median) / training IQR`. Predictions and all
reported MAE/RMSE values are inverse-transformed to g/L, mmHg, mmol/L, or %. Clinical
thresholds remain only for secondary AUC/bACC reporting.

For oxyhemoglobin fraction, `<94%` is an operational split-stratification and
secondary-metric threshold. It does not alter, clip, or weight the continuous
raw-value regression target.

Splits are patient-disjoint 60/20/20. The 20-frame retraining reuses and validates
the exact samples, raw values, and split assignments from the corresponding
face-plus-history experiment. Both architectures receive identical records and
splits.

The all-frame run has eight independent architecture/target jobs dispatched
dynamically across four GPUs. Every successful experiment automatically generates
its validated result figures after training.

## Start

```bash
bash study/exp2_face_pretrained_head32_regression/launch_screen.sh allframes
```

Attach with:

```bash
screen -r exp2_face_pretrained_head32_regression_allframes
```

The retained 20-frame entry is:

```bash
bash study/exp2_face_pretrained_head32_regression/launch_screen.sh 20frame
```

Newly configured targets can be appended with `--add-targets`; completed
checkpoints and metrics are retained, while the compact byte-offset index is
rebuilt only when the added target introduces previously unseen videos.

## Longitudinal Test Analysis

The trained 20-frame models can be evaluated for within-patient temporal tracking
on held-out test patients with at least two independent lab/video time points:

```bash
python -m study.exp2_face_pretrained_head32_regression.analyze_longitudinal_test
```

The analysis reuses the saved 20-frame video predictions, removes repeated video
assignments to the same lab event, and compares adjacent true and predicted changes.
Patient-cluster bootstrap confidence intervals and patient-level permutation tests
account for repeated transitions within a patient. Human-readable figures and the
report are separated from machine-readable CSV tables under
`outputs/20frame/longitudinal_test`.
