# Native-224 Paired-Face Laboratory Change Regression

Eleven independent models predict the difference between two distinct matched
laboratory measurements of the same patient: oxyhemoglobin fraction, lactate,
urea, troponin, platelet count, hemoglobin, A/a PO2 ratio, creatinine, total bilirubin,
hematocrit and creatinine-based reported eGFR. HCT/eGFR use direct report fields only,
not reconstructed HCT or equation-derived/cystatin-C eGFR.

The two faces share an ImageNet-pretrained EfficientNet-B0 encoder. The head
combines feature differences and has 32 hidden features. Twenty selected frames
per video form paired inputs, expanded to the same five synchronized views:
original, flip, center crop, brightness and contrast. Evaluation averages the
twenty original-view pair predictions.

The current 24h main rebuilds clinical pairing from the complete current laboratory
table and searches 512 patient-disjoint split candidates to balance delta distributions.
Each video matches its nearest same-admission report within 24h of the original
capture interval using validated Session Timestamp timing. Native224 frame eligibility
is checked before choosing the nearest video for each unique report, so an unusable
crop cannot displace a valid alternative. Consecutive unique patient measurements
with strictly increasing laboratory and video times form the pairs.
Target differences are robust-scaled
using training rows only. Training uses SmoothL1(beta=0.5), AdamW (wd=1e-4),
head lr=2e-4 / 40 epochs / patience 10, then full-backbone fine-tuning
lr=1e-5 / 60 epochs / patience 12, cosine floor=1e-6.

Every logical training batch contains 12 distinct observed laboratory-delta pairs,
20 corresponding frame pairs per observation and one synchronized view per pair:
240 frame pairs / 480 face images. All five views and all selected frames are
included once per epoch. To fit 16GB GPUs, the logical batch is processed in two
120-frame-pair microbatches, with weighted gradient accumulation, one gradient clip,
and one optimizer update. Existing inverse-patient-pair-count loss weights are retained.
This is a frame-pair loss, not a loss on averaged video predictions.
Validation/test use all twenty original-view frame pairs and average their outputs.
Measured raw deltas, rather than float32 inverse-scaled targets, are the authoritative
evaluation truth.

## Launch

```bash
# CPU-only preparation preview; existing results are untouched:
HEALTHMIRROR_FACE_SOURCE=face224 /root/miniconda3/envs/healthmirrorenv/bin/python -u \
  -m study.exp6_face_pair_lab_delta.run_full_data --prepare-only
# Wait for the active Exp2 HCT/eGFR/preoperative queue, then overwrite and train:
bash study/exp6_face_pair_lab_delta/launch_after_current_screen.sh
```

The monitor runs in detached screen `exp6_full24h_after_exp2` without allocating GPUs.
It requires successful predecessor completion and an unchanged predecessor contract;
an interrupted/failed predecessor never silently launches the next experiment.
At launch the latest lab table is reloaded, only `outputs/face224` is replaced,
and its training log is overwritten. A single real twelve-pair, two-stage smoke
check runs before the eleven-model dynamically scheduled four-GPU queue.
Matching-window ablation outputs are not deleted or retrained.
Monitor: `logs/face224/autostart.log`; training: `logs/face224/run.log`.

## Current Outputs

- Main 24h: `outputs/face224/`.
- Windows: `outputs/ablations/lab_match_6h_face224/` and
  `outputs/ablations/lab_match_12h_face224/`.
- Matching-window comparisons use the native-224 main, not deleted 128 results.

Each run saves model.pt, histories, metrics and pair_predictions.csv.
Figures are produced automatically.

```bash
bash study/common/launch_face224_reruns_screen.sh
# A single prepared protocol:
bash study/exp6_face_pair_lab_delta/launch_screen.sh --hours 24
```

The queue validates saved input hashes, skips complete protocols, and validates
individually completed tasks before reusing them. Native regression rejects 128
inputs. Previous 128 regression models, ablations and their launchers are removed.

## Protected 128-Only Classification Inputs

Root `outputs/task_records/`, scalers, run_index, source_data and the nine
root `runs/*/pair_predictions.csv` files are immutable clinical reference inputs
for the separate, protected 128-only direction-classification experiment.
Root `cache/frames20/` is its required shared index. No old regression model
weights or rendered frame caches remain in these reference locations.
