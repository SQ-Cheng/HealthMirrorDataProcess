# Exp2 Raw-Video Head32 Abnormal-Score Regression

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
- The retained 20-frame variant predicts `hemoglobin_low` and `po2_low` from 20
  deterministic non-adjacent frames sampled from 5% through 95% of each video.
- The all-frame variant predicts `hemoglobin_low`, `po2_low`, and `lactate_high`
  from every decodable frame in each video.
- Training views: original, horizontal flip, 90% center crop, brightness +6%, and
  contrast +8%.
- Validation and test use every frame selected by the variant with only the
  original view.

Each variant contains its own source audit and compact JPEG byte-offset index.
Decoded images are never persisted.

## Model And Optimization

Each architecture/target pair has an independent model:

- MobileNetV3-Small and EfficientNet-B0 with local ImageNet pretrained weights.
- Classification head replaced by
  `Linear -> LayerNorm -> SiLU -> Dropout -> Linear(1)`, hidden width 32.
- Stage 1 freezes the backbone and trains the head at `2e-4`.
- Stage 2 unfreezes the full model and fine-tunes at `1e-5`.
- Both stages minimize unweighted SmoothL1 abnormal-score loss (`beta=0.5`) with
  early stopping on validation video-level MAE.

Hemoglobin uses 130 g/L for male patients and 120 g/L otherwise; PO2 uses 80 mmHg.
Lactate uses an upper threshold of 2 mmol/L. Scores are transformed with `asinh`;
positive scores always indicate the abnormal side.

Splits are patient-disjoint 60/20/20. The existing 512-candidate balanced split
search is retained and audits raw-value and abnormal-score distributions. Both
architectures receive identical records and splits.

The all-frame run has six independent architecture/target jobs dispatched
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
