# Native-224 face-only raw laboratory regression

Eight independent EfficientNet-B0 models, each with a 32-dimensional hidden head,
predict oxyhemoglobin fraction, lactate, urea, total bilirubin, platelet count,
hemoglobin, A/a PO2 ratio and creatinine.

## Data and Training

- Input: twenty nonadjacent accepted 224x224 RGB frames per original video.
- Training views: original, horizontal flip, center crop, brightness and contrast.
- Evaluation: original view only; average the twenty frame predictions per video.
- Labels: nearest assay within 24h of the original capture interval, with canonical
  session timestamps, Asia/Shanghai laboratory timestamps and hospitalization checks.
- Patient-disjoint saved splits are retained after native-crop exclusions.
- Train-only median/IQR scaling, SmoothL1 loss (beta=0.5), AdamW, weight decay 1e-4.
- Head: lr=2e-4, at most 40 epochs, patience 10.
- Full-backbone fine-tuning: lr=1e-5, at most 60 epochs, patience 12.
- Cosine floor: 1e-6. Torch compile: reduce-overhead. Four-GPU task scheduling.

This experiment rejects 128/MJPEG inputs. The shared legacy decoder is retained
for other studies whose main experiment still exists only at 128 resolution.

## Current Results

- Main: `outputs/20frame_face224/`.
- Matching-window ablations: `outputs/ablations/lab_match_{6,12}h_face224/`.
- Patient-diverse 30/40: `outputs/ablations/patient_diverse_schedule_30_40_face224/`.
- 12h patient-diverse five-fold:
  `outputs/ablations/lab_match_12h_patient_diverse_schedule_30_40_face224/5fold/`.

Runs contain checkpoints, frame/video predictions, metrics and training histories;
figures are generated automatically. Corresponding logs use the same native suffix.
The patient-diverse schedule uses head lr=1e-4 / 30 epochs / patience 8 and
fine-tuning lr=3e-6 / 40 epochs / patience 8, with cosine floors 1e-6 / 1e-7.

## Entry Points

Continue the prepared native protocols without repeating completed jobs:

```bash
bash study/common/launch_face224_reruns_screen.sh
bash study/exp2_face_pretrained_head32_regression/launch_patient_diverse_224_screen.sh
bash study/common/launch_face224_12h_5fold_screen.sh
```

The latter two screens wait for their predecessors. Checkpoint/label/scaler
validation precedes reuse of an individually completed job. Interrupted unfinished
jobs restart; completed tasks and patient assignments are preserved.

Fresh clinical preparation/training is available through `run_all.py`, whose
default output and input protocol are native 224. It refuses an existing result
unless an explicit overwrite/add-targets option is supplied.

CPU-only longitudinal change tracking is in
[`../exp2_face224_lab_change_tracking/`](../exp2_face224_lab_change_tracking/README.md).

## Protected Clinical References

`outputs/20frame/` and `outputs/5fold/splits/` contain only immutable clinical
labels, split assignments and scalers needed by the 128-only Exp3 experiment.
They are not 128 face-model results or pixel caches. Old face-only models,
predictions, ablations, launchers and image indexes have been removed.
