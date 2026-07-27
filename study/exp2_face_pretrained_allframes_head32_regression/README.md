# Exp2 Single-Task All-Frame Abnormal-Score Regression

Canonical ID: `exp2_regression_allframes_head32_single_task`.

This experiment keeps the data, patient split, all-frame streaming, five training
views, pretrained backbones, 32-dimensional head, and two-stage optimization used by
`exp2_face_pretrained_allframes_head32`. The prediction target is a continuous
abnormal score instead of a binary label.

## Tasks and Models

Each architecture/target pair has an independent model:

- Architectures: MobileNetV3-Small and EfficientNet-B0
- Targets: `hemoglobin_low`, `pco2_low`, `po2_low`,
  `high_blood_pressure`, and `lactate_high`
- Head: Linear -> LayerNorm -> SiLU -> Dropout -> Linear(1), hidden width 32

The five deterministic training views are original, horizontal flip, 90% center
crop, brightness +6%, and contrast +8%. Validation and test use only the original
frame.

## Abnormal Score

Directional standardized distance is positive on the task's abnormal side, negative
on its normal side, and zero at the clinical boundary. The regression target is the
inverse-hyperbolic-sine transform of that distance:

```text
low task:  asinh((lower_threshold - value) / scale)
high task: asinh((value - upper_threshold) / scale)
blood pressure:
  asinh(max((systolic - 140) / 20, (diastolic - 90) / 10))
```

Hemoglobin uses 130 g/L for male patients and 120 g/L otherwise. PCO2, PO2, and
lactate use thresholds/scales of 34/5 mmHg, 80/10 mmHg, and 2/1 mmol/L. Exact
machine-readable definitions are written to `outputs/score_definition.json`.

Videos containing both normal and abnormal events for a target remain excluded.
For a clean video with multiple events, the closest event by absolute lab-video time
difference is retained. The corrected Asia/Shanghai timestamps and 24-hour matching
limit are validated before records are built. Splits remain patient-disjoint.
Preprocessing searches 512 class-stratified 60/20/20 assignments and selects one
using video-level raw-value and abnormal-score distributions. It rejects a split
when its KS or IQR-normalized Wasserstein limits are exceeded. Both architectures
receive identical records and splits.

## Training and Outputs

Training minimizes unweighted SmoothL1 loss (`beta=0.5`) over all five-view frame
inputs. Normal, boundary, and abnormal targets use identical loss weights. Stage 1
freezes the encoder and trains the head at `2e-4`; stage 2 unfreezes all parameters
and fine-tunes at `1e-5`. Early stopping and final stage selection minimize
validation video-level MAE.

Each run saves:

- `history.csv` and `history.png`
- video-level MAE, RMSE, median AE, R2, Pearson, Spearman, and sign metrics
- `video_predictions.csv` and compressed `frame_predictions.npz`
- head-stage, fine-tune-stage, and selected model checkpoints

Decoded frames are not persisted. The experiment reuses the compact JPEG byte-offset
index from `exp2_face_pretrained_allframes_head32`. Four persistent GPU slots consume
the ten jobs from a dynamic queue; each slot immediately starts the next pending job.

## Manual Start

```bash
bash study/exp2_face_pretrained_allframes_head32_regression/launch_screen.sh --overwrite
```

## Result Figures

Regenerate the video-level result figures with:

```bash
python study/exp2_face_pretrained_allframes_head32_regression/plot_results.py
```

The script writes training curves, test regression/sign metrics, split
generalization plots, and test predicted-versus-true scatter plots to
`outputs/figures/`.
