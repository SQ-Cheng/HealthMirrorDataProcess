# Exp2 Raw-Video Head32 Raw-Value Regression

The retained 20-frame control uses the exact source builder from the corresponding
face-plus-history experiment. That builder enumerates every raw `video.avi`, maps
its hospital ID, and reads
the capture interval from the corresponding frame-level `video.avi.ts`. It does not
use cleaned ECG/rPPG session CSV files.

## Data Policy

- Maximum video-lab interval distance: 24 hours.
- One label per video and target: choose the nearest valid measurement by interval
  distance, then video-midpoint distance, then report timestamp.
- The experiment predicts oxyhemoglobin fraction, lactate, urea, total bilirubin,
  platelet count, hemoglobin, A/a PO2 ratio, and creatinine from 20 deterministic
  non-adjacent frames sampled from 5% through 95% of each video.
- Training views: original, horizontal flip, 90% center crop, brightness +6%, and
  contrast +8%.
- Validation and test use every frame selected by the variant with only the
  original view.
- Oxyhemoglobin fraction accepts `氧合血红蛋白分数` and `氧合血红蛋白` values
  reported in percent, excludes explicitly venous specimens, and enforces the
  physical range 0-100%.

Each variant contains its own source audit and compact JPEG byte-offset index.
Decoded images are never persisted.

## Model And Optimization

Each architecture/target pair has an independent model:

- EfficientNet-B0 with local ImageNet pretrained weights.
- Classification head replaced by
  `Linear -> LayerNorm -> SiLU -> Dropout -> Linear(1)`, hidden width 32.
- Stage 1 freezes the backbone and trains the head at `2e-4`.
- Stage 2 unfreezes the full model and fine-tunes at `1e-5`.
- Both stages minimize unweighted SmoothL1 loss (`beta=0.5`) on robust-scaled raw
  values with early stopping on inverse-transformed validation video-level MAE.

Each target scaler is fitted only on training videos:
`scaled = (raw value - training median) / training IQR`. Predictions and all
reported MAE/RMSE values are inverse-transformed to canonical lab units. Clinical
thresholds remain only for secondary AUC/bACC reporting.

For oxyhemoglobin fraction, `<94%` is an operational split-stratification and
secondary-metric threshold. It does not alter, clip, or weight the continuous
raw-value regression target.

Splits are patient-disjoint 60/20/20. The 20-frame retraining reuses the history
experiment's compact frame-offset index and validates the exact samples, raw
values, and split assignments from that experiment. Both architectures receive
identical records and splits. The intended comparison changes only the presence
of the prior-lab history encoder and its input.

The eight independent target jobs are dispatched dynamically across four GPUs.
Every successful experiment automatically generates validated result figures.

## Start

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

For the current eight-target raw-value regressors, run
`python -m study.exp2_face_history_head32_regression.analyze_bidirectional_change`.
This separately measures rise/fall recall on adjacent test lab events without
retraining and writes figures and tables to
`outputs/20frame/bidirectional_change_analysis/`.

## Training/backbone ablations

`launch_ablations_screen.sh` runs three new eight-target experiments sequentially
on four GPUs without altering the completed EfficientNet-B0 baseline:

1. Direct joint head+backbone training from ImageNet initialization at `2e-5`,
   at most 100 epochs, patience 12.
2. The baseline two-stage schedule, but fine-tune only EfficientNet
   `features[7:9]` (about 28% of backbone parameters) and the head. Earlier
   blocks and their BatchNorm running statistics stay frozen.
3. The baseline two-stage schedule with an ImageNet-pretrained ShuffleNetV2
   x1.0 backbone and the same 32-dimensional head. Source batch sizes and
   bicubic preprocessing are held equal to EfficientNet for this comparison.

All three use the saved baseline task records, train-only scalers, patient
split, 20-frame byte-offset index, five training views and per-task random
seeds. Each target has its own model. An unsuccessful job stops the sequence;
successful completion creates per-variant figures and
`outputs/ablations/comparison/figures/` with test-set comparisons. The baseline
is read only. Weight files are in `study/common/pretrained_weights/`; obtain
them with `python -m study.common.download_weights` on a new machine.

```bash
bash study/exp2_face_pretrained_head32_regression/launch_ablations_screen.sh
```

## Patient-diverse batch ablation

This ablation changes only the training sampler. Each 48-source-frame batch
draws four frames per video block and tries to use 12 distinct patients;
when too few distinct patients remain, it fills the batch from the remaining
blocks. Every selected frame appears exactly once per epoch and still expands
to all five views. Evaluation order, saved records, split, scaler, seeds,
model, optimizer, schedule and early stopping remain unchanged. Training
outputs are isolated under `outputs/ablations/patient_diverse_batches/`.
Completion automatically generates baseline comparison tables and figures
there, in addition to the normal per-task result figures.

```bash
bash study/exp2_face_pretrained_head32_regression/launch_patient_diverse_screen.sh
```

The schedule-only follow-up keeps the completed patient-diverse data and
sampler unchanged. It uses head LR `1e-4`, at most 30 epochs and patience 8;
fine-tune LR `3e-6` decaying toward `1e-7`, at most 40 epochs and patience 8.
The head-stage LR floor remains `1e-6`. Its independent output
is `outputs/ablations/patient_diverse_schedule_30_40/`, including automatic
paired test and validation-history figures against the original patient-diverse
schedule.

```bash
bash study/exp2_face_pretrained_head32_regression/launch_patient_diverse_schedule_screen.sh
```

Two further independent regularization ablations use that completed 30/40
run as their control. One changes only AdamW weight decay from `1e-4` to
`1e-2` in both stages. The other keeps `1e-4` and fixes EfficientNet-B0
BatchNorm running statistics in both stages; its BatchNorm affine parameters
remain trainable during fine-tuning. They run sequentially, with four dynamic
GPU workers per variant. Each variant generates its normal figures and a
paired control comparison; after both finish, three-way figures and test
metrics appear under `outputs/ablations/patient_diverse_regularization_comparison/`.

```bash
bash study/exp2_face_pretrained_head32_regression/launch_patient_diverse_regularization_screen.sh
```

The interleaved-view batch ablation uses the same 30/40 schedule, seed, split,
model, optimizer and five views. It changes only batch construction: each
240-image batch contains at most one view of each source frame, favoring
different patients before using additional frames from the same patient.
Each frame still contributes all five views once per epoch, so the training
sample inventory is unchanged. Five per-view passes can add up to two partial
optimizer steps per epoch compared with the grouped-view control. Repeated
JPEG decoding can lower throughput, but no persistent image cache is created.
It automatically saves the usual figures plus paired test and validation
comparisons against `patient_diverse_schedule_30_40`.

The following starts a detached monitor. After both regularization ablations
finish, it automatically runs the interleaved-view experiment on four GPUs:

```bash
bash study/exp2_face_pretrained_head32_regression/launch_interleaved_after_regularization_screen.sh
```
