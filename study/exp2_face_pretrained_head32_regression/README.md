# Exp2 Raw-Video 20-Frame Abnormal-Score Regression

Canonical ID: `exp2_raw_video_20frame_head32_regression_balanced_split`.

This experiment predicts continuous abnormal scores from raw face videos. It does
not use cleaned ECG/rPPG session CSV files to select videos or estimate capture
time. The source builder enumerates every raw `video.avi`, maps its hospital ID,
and reads the capture interval from the corresponding frame-level `video.avi.ts`.

## Data Policy

- Maximum video-lab interval distance: 24 hours.
- One label per video and target: choose the nearest valid measurement by interval
  distance, then video-midpoint distance, then report timestamp.
- Targets: `hemoglobin_low` and `po2_low`.
- Frames: 20 deterministic non-adjacent frames per video, sampled from 5% through
  95% of the MJPEG source frame sequence.
- Training views: original, horizontal flip, 90% center crop, brightness +6%, and
  contrast +8%.
- Validation and test use the same 20 source frames with only the original view.

The source audit is regenerated under `outputs/source_data/`. It records every raw
video and the reason it was retained or rejected. JPEG byte offsets are cached in
`outputs/frame_index/`; decoded images are never persisted.

## Model And Optimization

Each architecture/target pair has an independent model:

- MobileNetV3-Small and EfficientNet-B0 with local ImageNet pretrained weights.
- Classification head replaced by
  `Linear -> LayerNorm -> SiLU -> Dropout -> Linear(1)`, hidden width 32.
- Stage 1 freezes the backbone and trains the head at `2e-4`.
- Stage 2 unfreezes the full model and fine-tunes at `1e-5`.
- Both stages minimize unweighted SmoothL1 abnormal-score loss (`beta=0.5`) with
  early stopping on validation video-level MAE.

Hemoglobin uses 130 g/L for male patients and 120 g/L otherwise. PO2 uses 80 mmHg.
The target is `asinh((threshold - value) / scale)`, with scales 10 g/L and
10 mmHg respectively. Positive scores are on the abnormal-low side.

Splits are patient-disjoint 60/20/20. The existing 512-candidate balanced split
search is retained and audits raw-value and abnormal-score distributions. Both
architectures receive identical records and splits.

Four independent architecture/target jobs are dispatched dynamically across the
four visible GPUs. Each training run saves full CSV/PNG history, checkpoints,
frame predictions, video predictions, and train/validation/test metrics.

## Start

```bash
bash study/exp2_face_pretrained_head32_regression/launch_screen.sh --overwrite
```

Attach with:

```bash
screen -r exp2_face_pretrained_head32_regression
```

Regenerate result figures after completion with:

```bash
python study/exp2_face_pretrained_head32_regression/plot_results.py
```
