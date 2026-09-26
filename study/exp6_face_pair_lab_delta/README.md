# Exp6: paired-face laboratory delta regression

This experiment predicts the change between two laboratory measurements from
two chronologically corresponding face videos.

## Model

- One independent model per laboratory target.
- Shared ImageNet-pretrained EfficientNet-B0 encoder for both faces.
- The 1,280-dimensional feature difference (`second - first`) is passed to a
  32-dimensional `Linear + LayerNorm + SiLU + Dropout + Linear` head.
- Stage 1 freezes the encoder and trains the head at `2e-4`.
- Stage 2 unfreezes the full encoder and fine-tunes all parameters at `1e-5`.

## Data

- Targets: O2Hb fraction, lactate, urea, troponin, platelet count,
  hemoglobin, A/a PO2 ratio, creatinine, and total bilirubin.
- Each raw video is assigned its nearest laboratory event within 24 hours. If
  multiple videos map to one event, only the closest video is retained.
- Consecutive same-patient events form an ordered pair when both lab and video
  timestamps increase and the videos differ.
- Label: canonical second value minus canonical first value.
- Twenty deterministic nonadjacent frames are streamed from each video.
- Training expands every frame pair into original, horizontal flip, center
  crop, brightness, and contrast views. The same view is applied to both sides.
- Split is patient-disjoint. Among 512 candidates, the closest train/validation/
  test delta distributions are selected. Delta robust scaling uses train only.

## Run

```bash
screen -dmS exp6_pair_delta bash -lc \
  'cd /root/autodl-tmp/HealthMirrorDataProcess && \
   bash study/exp6_face_pair_lab_delta/launch_screen.sh \
   2>&1 | tee study/exp6_face_pair_lab_delta/logs/run.log'
```

The launcher dynamically schedules one task on each of four GPUs and assigns a
new task whenever a GPU finishes. Figures are generated automatically after all
tasks complete.

## Patient-diverse 30/40 ablation

`shared_patient_diverse_30_40` keeps the same nine pair records, patient split,
target scalers, frame index, shared pretrained backbone, 20 frame pairs, five
synchronized views, patient-weighted loss, and validation/test protocol as the
main experiment. Relative to the main run, each 24-source-pair training batch
mixes up to 12 patients (two frame pairs per patient); every source frame pair
is still used once per epoch. The head runs for at most 30 epochs at `1e-4`
with patience 8 and a `1e-6` cosine floor. Full-backbone fine-tuning runs for
at most 40 epochs at `3e-6` with patience 8 and a `1e-7` floor. This matches
the controlled changes in Exp2 `patient_diverse_schedule_30_40`.

Launch with `bash study/exp6_face_pair_lab_delta/launch_patient_diverse_30_40_screen.sh`.
The script starts a detached four-GPU screen session. All outputs go to
`outputs/shared_patient_diverse_30_40/`; at completion it writes the standard
figures, per-task baseline comparisons, and `baseline_comparison.csv` without
changing the main experiment outputs.

## Three-view shared-backbone variant

`shared_views3` reuses the existing prepared pairs, patient splits, train-only
scalers, frame index, shared EfficientNet-B0 model, and two-stage training
schedule. It applies only original, horizontal-flip, and center-crop views to
both faces in each training pair. Validation and test still use original frames.
Results, checkpoints, histories, predictions, and the three standard figures
are written under `outputs/shared_views3/`; the existing outputs are untouched.

Launch it in screen with:

```bash
screen -L \
  -Logfile /root/autodl-tmp/HealthMirrorDataProcess/study/exp6_face_pair_lab_delta/logs/shared_views3.log \
  -dmS exp6_shared_views3 bash -lc \
  'cd /root/autodl-tmp/HealthMirrorDataProcess && \
   CUDA_VISIBLE_DEVICES=0,1,2,3 MKL_THREADING_LAYER=GNU PYTHONUNBUFFERED=1 \
   exec bash study/exp6_face_pair_lab_delta/launch_views3_screen.sh'
```

## Independent-backbone variant

The controlled variant keeps the same prepared pairs, patient splits, target
scalers, frames, synchronized views, feature-difference fusion, and 32-unit
head. It replaces the shared encoder with two independently parameterized
EfficientNet-B0 encoders. Both start from the same ImageNet checkpoint; the
early-face and late-face parameters can diverge during full fine-tuning.

Results are stored separately under `outputs/independent_backbones/` and do not
overwrite the shared-backbone results. Launch it with:

```bash
screen -L \
  -Logfile /root/autodl-tmp/HealthMirrorDataProcess/study/exp6_face_pair_lab_delta/logs/independent_backbones.log \
  -dmS exp6_independent_backbones bash -lc \
  'cd /root/autodl-tmp/HealthMirrorDataProcess && \
   CUDA_VISIBLE_DEVICES=0,1,2,3 MKL_THREADING_LAYER=GNU PYTHONUNBUFFERED=1 \
   exec bash study/exp6_face_pair_lab_delta/launch_independent_backbones_screen.sh'
```
