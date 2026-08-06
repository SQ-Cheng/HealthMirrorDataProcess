# Exp2 Face + Prior Lab History Head32 Raw-Value Regression

This controlled experiment uses the retained 20-frame Head32 regression setup
with prior measurements of the same analyte. It trains independent
MobileNetV3-Small and EfficientNet-B0 models for Hemoglobin, PO2, and
oxyhemoglobin fraction.

## Controlled Data

- The video records, labels, patient-disjoint train/validation/test split, selected
  20 frames, five training views, and random seed must match
  `exp2_face_pretrained_head32_regression/outputs/20frame` exactly.
- For each video label, history contains every same-analyte result that is in the
  same admission-discharge episode and strictly earlier than the current label.
- Other admissions, the current result, and future results are excluded.
- Each retained history row records its raw value and
  `history_time - current_label_time` in hours in a machine-readable CSV.
- Each target uses train-only robust scaling:
  `(raw_value - train_median) / train_IQR`. Validation and test values never fit
  the scaler. Predictions are inverse-transformed before raw-unit metrics.
- The compact NPZ passed to the model contains the historical raw value transformed
  by the same target scaler and `-log1p(age_hours / 24)`. Sequences are not truncated.

## Train-range weighted variant

`launch_range_weighted.sh` trains Hemoglobin and oxyhemoglobin fraction into
`outputs/20frame_range_weighted` without changing the unweighted outputs. Five
fixed-width raw-value ranges are fitted between the training-set p01 and p99;
each range receives a square-root inverse-frequency weight normalized to mean
one. Validation/test losses and all reported metrics remain unweighted. After
training, the launcher creates both normal result figures and matched-test-set
comparisons against `outputs/20frame`.

## Model

Each historical measurement is encoded by:

```text
(value feature, time feature)
  -> Linear(2,16) -> SiLU -> Linear(16,16) -> LayerNorm(16) -> SiLU
  -> masked mean over all prior measurements
```

The 16-dimensional history representation is concatenated with the image backbone
feature before the original regression head. The head hidden width remains 32:

```text
concat(image feature, history feature)
  -> Linear(*,32) -> LayerNorm(32) -> SiLU -> Dropout -> Linear(32,1)
```

The history encoder has 352 trainable parameters. Stage 1 freezes the image
backbone and trains the history encoder plus regression head at `2e-4`. Stage 2
fine-tunes the whole model at `1e-5`. The loss is unweighted SmoothL1 in robust-
scaled target space; MAE/RMSE and prediction files use `g/L` or `mmHg`.

## Start

```bash
bash study/exp2_face_history_head32_regression/launch_screen.sh
```

Four dynamic workers are assigned to GPUs 0-3. Result figures are generated
automatically after all four architecture/target jobs finish.
