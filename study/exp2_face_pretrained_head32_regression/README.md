# Exp2 Raw-Video Head32 Raw-Value Regression

## Native 224 Inputs

The default input now selects validated `face224.mkv` crops from the original
raw videos when the production crop protocol is present. This avoids legacy
128-to-224 upsampling. Use `HEALTHMIRROR_FACE_SOURCE=legacy128` to reproduce
previous runs. The automatic reruns preserve existing results under separate
`*_face224` outputs. See [the shared protocol](../common/FACE224_PROTOCOL.md)
for source validation, cache reuse, saved splits and comparison figures.

The additional native-224 patient-diverse 30/40-epoch ablation is queued after
the existing native-224 experiment queue, without altering or interrupting it:

```bash
bash study/exp2_face_pretrained_head32_regression/launch_patient_diverse_224_screen.sh
screen -r face224_patient_diverse_30_40
```

It reuses `outputs/20frame_face224` records/scalers and the shared FFV1 index,
and reproduces the saved legacy ablation's task seeds and patient-diverse
schedule (head 1e-4/30 epochs, fine-tune 3e-6/40 epochs, patience 8/8,
minimum rates 1e-6/1e-7, no warmup). Results remain separate in
`outputs/ablations/patient_diverse_schedule_30_40_face224`; logs use the matching
`logs/ablations` path. Normal plots, legacy-128/native-224 comparisons, and a
same-cohort comparison against the native-224 main run are generated automatically.
The monitor requires both the preceding queue's released lock and completion
marker. A stopped, incomplete queue is reported as an error rather than launching
into failed prerequisites.

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

To rerun the controlled all-frame variants, use
`bash study/exp2_face_pretrained_head32_regression/launch_allframes_aligned_screen.sh`.
The detached screen runs 24 h, 12 h, then 6 h sequentially on four GPUs. Each
variant reuses its completed 20-frame counterpart's matched source data and
exact target-specific videos, labels, and patient split; only the frame policy
changes. The 24 h result replaces `outputs/allframes/`, while the shorter
windows go to `outputs/ablations/lab_match_{12,6}h_allframes/`. All three share
the compact all-frame offset index in `cache/allframes_index/`, not decoded
frame files. Logs are under `logs/allframes_aligned/`, and each completed
variant writes standalone figures and a paired 20-frame test comparison.

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

### 12-hour lab/video matching window

The saved 24-, 12-, and 6-hour video/laboratory timing distributions can be
audited without training via
`python -m study.exp2_face_pretrained_head32_regression.analyze_match_window_timing`.
The results under `outputs/match_window_timing/` include per-target and
per-split quantiles, time-bin counts, direction relative to the video session,
and 4-by-2 ECDF and stacked-bin figures. A video repeated across analytes is
counted once per analyte; the summary also reports distinct physical videos.

`launch_match_12h_screen.sh` repeats the baseline eight-target EfficientNet-B0
regression experiment with only the maximum lab-to-video-interval distance
reduced from 24 to 12 hours. It rebuilds the raw-video nearest-lab source,
reruns the 512-candidate patient-disjoint distribution search for each target,
refits train-only robust scalers, and reuses the validated 20-frame byte-offset
index. Architecture, five views, optimizer, two training stages, and early
stopping remain unchanged. Outputs and the continuous log are isolated under
`outputs/ablations/lab_match_12h/` and `logs/ablations/lab_match_12h/`.
On successful completion, it generates its own result figures plus
`match_window_comparison.csv`, `shared_test_comparison.csv`, and matching
comparison figures against `outputs/20frame/`. The two independently searched
splits have different full test cohorts; their full-cohort comparison is
descriptive. A separate common-video analysis evaluates both models on the
same patients' videos held out from both training sets.

```bash
bash study/exp2_face_pretrained_head32_regression/launch_match_12h_screen.sh
```

The matching-window workflow also supports a 6-hour limit, with independent
source matching, split search, training, and results under
`outputs/ablations/lab_match_6h/` and `logs/ablations/lab_match_6h/`:

```bash
bash study/exp2_face_pretrained_head32_regression/launch_match_6h_screen.sh
```

To apply the existing patient-diverse 30/40 two-stage schedule to each
matching window, run `launch_window_schedule_monitor_screen.sh`. Its detached
screen waits for the 6-hour baseline screen to finish successfully, then runs
the 12-hour and 6-hour variants sequentially on four GPUs. Each variant uses
its window's saved 20-frame records, patient split, train-only scalers and
five views. Results and checkpoints are kept in
`outputs/ablations/lab_match_{12,6}h_patient_diverse_schedule_30_40/`, with
per-window test comparisons to the matching-window baseline in each variant's
`figures/schedule_comparison.png` and `schedule_comparison.csv`. The monitor
log is `logs/ablations/match_window_schedule_30_40/run.log`.

```bash
bash study/exp2_face_pretrained_head32_regression/launch_window_schedule_monitor_screen.sh
```

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

The warmup follow-up changes only the learning-rate timing relative to the
completed patient-diverse 30/40 run. Each stage adds two linear warmup epochs:
50% then 100% of its original initial LR. The unchanged cosine schedule then
runs for the original 30 head or 40 fine-tune epochs, with the original
`T_max`, LR floor, and patience. Total stage limits are 32 and 42 epochs.
Its independent output is
`outputs/ablations/patient_diverse_schedule_30_40_warmup2/`; normal result
figures and paired comparisons against 30/40 are generated automatically.

The following script starts a detached screen monitor that waits for the
face-only binary patient-diverse 30/40 experiment to complete before running
this four-GPU warmup ablation:

```bash
bash study/exp2_face_pretrained_head32_regression/launch_warmup_monitor_screen.sh
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

## Label-density and contrastive-initialization ablations

These are two independent comparisons against the completed patient-diverse
30/40 control. They reuse its eight tasks, patient split, 20 frames/video,
five training views, train-only robust scalers, batch sampler, job seeds,
EfficientNet-B0 architecture and downstream two-stage schedule. The first
variant changes only the training loss: 12 equal-width bins fitted between
the training labels' 1st and 99th percentiles receive inverse-square-root
frequency weights. The tails are clipped into the edge bins, weights are
capped at a 4:1 ratio and normalized to mean one. Validation and test metrics
remain unweighted.

The second variant changes only backbone initialization. Before the usual
head and full-backbone stages, it trains the ImageNet-initialized backbone for
eight epochs with a temporary 128-dimensional contrastive projection head,
which is then discarded. Only training labels construct pairs: images from
the same video or with values within 0.25 training IQR are positives,
including close-valued different lab events from the same patient; values
at least 0.75 IQR apart are negatives. Intermediate-distance pairs are
excluded. The NT-Xent temperature is 0.20. Contrastive batches contain
12 anchor videos, 12 close-valued partners, two source frames/video and two
different randomly sampled views/frame (96 images). AdamW uses backbone
LR `1e-4`, projection-head LR `3e-4`, weight decay `1e-4`, and cosine decay
to `1e-6`. This is label-aware contrastive pretraining, not unsupervised
SimCLR; no validation or test labels are used in pretraining.

The variants run sequentially with dynamic four-GPU task scheduling. Each
produces normal per-target figures and paired comparisons with the control;
the final three-way figures and metrics are in
`outputs/ablations/patient_diverse_density_simclr_comparison/`. Detailed
train-bin weights, contrastive pairing audits, pretraining histories and
encoder checkpoints remain in the individual variant directories.

```bash
bash study/exp2_face_pretrained_head32_regression/launch_density_simclr_screen.sh
```
