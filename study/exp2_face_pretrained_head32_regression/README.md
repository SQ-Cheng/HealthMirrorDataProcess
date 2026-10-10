# Native-224 face-only raw laboratory regression

Eight independent EfficientNet-B0 models, each with a 32-dimensional hidden head,
predict oxyhemoglobin fraction, lactate, urea, total bilirubin, platelet count,
hemoglobin, A/a PO2 ratio and creatinine. The direct-field extension adds
hematocrit and creatinine-based reported eGFR, making ten independent regression tasks.

## Direct-Field Extension and Preoperative Ablation

`run_preoperative_ablation.py` supplements the 24h main result with
`hematocrit_low` and `egfr_low`. Only directly reported `*红细胞压积` /
`红细胞压积` (%) and `eGFR(CKD-EPI 肌酐)` are used.
No Hb/MCHC reconstruction, formula-derived eGFR, or cystatin-C eGFR is pooled.
The original eight checkpoints, task records and matching source files are preserved.
The supplementary targets use the same 512-candidate patient-disjoint balanced
split search and train-only median/IQR scaling.

The ten-task ablation lives in
`outputs/ablations/preoperative_nearest_unlimited_face224`; its log is in
`logs/ablations/preoperative_nearest_unlimited_face224/run.log`.
Videos entirely before a unique valid CABG in the same admission use the nearest
report strictly before CABG start, with no distance limit. The assay may precede
or follow the video. No preoperative label comes from during/after surgery or
another admission. Preoperative videos without a preoperative assay are excluded
for that target. Other videos keep their exact main 24h rule. Missing or ambiguous
CABG metadata never authorizes unrestricted matching.

Existing patients retain their per-target main splits. Only new patients receive
fixed-seed 60/20/20 assignments. Distribution audits do not silently move patients.
Twenty nonadjacent native224 frames, five views, independent EfficientNet-B0/head32
models, frame-level SmoothL1, 12 distinct assay events per 240-frame batch, two-stage
40/60 epochs, lr 2e-4/1e-5, patience 10/12, and other main settings are unchanged.
HCT cutoffs (male 37%, other 34%) and eGFR 60 are auxiliary split/metric definitions,
not regression targets or diagnostic claims.
Sources: [HCT](https://www.medlineplus.gov/ency/article/003646.htm),
[eGFR](https://www.kidney.org/what-criteria-ckd).

Prepare once and launch the resumable four-GPU queue:

```bash
HEALTHMIRROR_FACE_SOURCE=face224 /root/miniconda3/envs/healthmirrorenv/bin/python -u \
  -m study.exp2_face_pretrained_head32_regression.run_preoperative_ablation --prepare-only
bash study/exp2_face_pretrained_head32_regression/launch_preoperative_ablation_screen.sh
```

Matching completed jobs are skipped on relaunch; different saved run contracts are
rejected rather than overwritten. Main figures expand to ten tasks once both new
main runs finish. The ablation automatically produces normal regression figures,
full-cohort comparisons, and a separate comparison on shared test videos with
unchanged target values. Full cohorts are not treated as an identical-test comparison.

## Data and Training

The shared-backbone ablation is a separate experiment:
[exp2_face_shared_backbone_head32_regression](../exp2_face_shared_backbone_head32_regression/README.md).
Its code, outputs, checkpoints and logs are not stored in this main experiment.

- Input: twenty nonadjacent accepted 224x224 RGB frames per original video.
- Training views: original, horizontal flip, center crop, brightness and contrast.
- Evaluation: original view only; average the twenty frame predictions per video.
- Labels: nearest assay within 24h of the original capture interval, with canonical
  session timestamps, Asia/Shanghai laboratory timestamps and hospitalization checks.
- A fresh 512-candidate patient-disjoint distribution search is used after native-crop exclusions;
  the classification main experiment reuses the exact resulting split for each target.
- Default loss: per-frame SmoothL1. Each full batch contains twelve distinct matched
  lab events, one video per event, twenty frames per video and one view: 240 frames.
  All five views occur across different batches each epoch; no video/frame/view is dropped.
  Different events from the same patient can occur together. No twenty-frame loss pooling.
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

Overwrite and rerun the two matched 24h frame-loss main experiments on four GPUs:

```bash
bash study/common/launch_face_main_24h_screen.sh
screen -r exp2_face_main_24h_frame_loss
```

Main log: `study/common/logs/face_main_24h_frame_loss/run.log`.
Only `outputs/20frame_face224/` and the classification `outputs/face224/` are
overwritten; existing ablations are untouched. Standalone regression uses the
same defaults through `python -m study.exp2_face_pretrained_head32_regression.run_all --overwrite`.

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
