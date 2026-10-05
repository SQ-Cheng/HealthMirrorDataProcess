# Native 224 Face Inputs And Automatic Reruns

## Input Resolution And Compatibility

Exp2 image/history loaders, their binary classifiers, Exp6 paired image loaders,
and Exp8 now support the FFV1 crops written beside each original video:
`/root/shared/HealthMirrorRawData/mirror*_data/patient_*/face224.mkv`.
These are confidence >=0.75, top expansion 20%, Kalman bbox smoothing, **no
alignment**, 224 x 224 full-color crops. Expanded boxes with >10% outside-source
area are rejected. Invalid/held frames are removed, not replaced with black or
duplicated frames. FFV1 uses GOP 1 and preserves the cropped pixels exactly.

`HEALTHMIRROR_FACE_SOURCE` controls the shared reader:

- `auto` (default): use native 224 when the raw root contains the production
  protocol; otherwise retain the legacy 128 MJPEG workflow.
- `face224`: require validated native crops; never fall back to 128 pixels.
- `legacy128`: explicitly reproduce the previous data path and old indexes.

`HEALTHMIRROR_RAW_ROOT` optionally changes the raw-data root. Existing 128
NPZ indexes still load; they are not interpreted as FFV1 packets. Native and
legacy index schemas/source policies are checked before reuse. The default
native 20-frame index is `study/common/cache/face224_20frame/`, separate from
old experiment caches. Payload byte offsets and FFV1 codec headers are saved,
not another image/tensor cache. Workers seek and read one independent encoded
packet, then decode it through bounded per-worker codec/pixel LRUs. RGB channel
order and random-order decoding are tested pixel-exact against full decoding.
Twenty-frame index creation seeks directly to each selected PTS using Matroska
cues, rather than reading every large encoded payload. Full-frame indexes still
scan every retained packet. Selected PTS and time bases are checked against the
verified production sidecar in both cases.
Original 224 images bypass identity resize in the GPU augmentation path; the
center-crop view still resizes its deliberately smaller ROI, as before.

For each clinical video, the raw directory's local/session ID and hospital ID
are checked against its source record. Temporal matching still uses the saved
**original session** interval and canonical Session Timestamp, never the
duration or compressed index of a filtered crop. `face224_frames.csv` preserves
the source index and exact times after frames are removed. Twenty frames are
selected deterministically from retained frames, with the original minimum
source-index separation of two frames. The 5-view train policy and original-only
validation/test policy remain unchanged for Exp2/Exp6.

## Unusable Sources

Missing originals, zero accepted frames and insufficient nonadjacent accepted
frames are excluded with an audit, not replaced by legacy inputs. Explicit
original-data failures (JPEG/timestamp count mismatch, invalid/nonmonotonic
timestamp indices, too few frames or unexpected timestamp schema) are also
excluded: their temporal correspondence cannot be inferred safely. Other
processing failures stop the automatic queue instead of being silently accepted.
Old studies reference mirror5, but the current raw root has mirrors 1, 2, 4, 6,
and 7 only. Missing mirror5 originals therefore reduce the native cohort.

The queue keeps each reference experiment's saved labels, matching window and
patient-disjoint split assignments. It does not search a different split to
compensate for the reduced cohort. Regression scalers are refitted on retained
training records only; binary positive weights and Exp6 patient weights follow
their unchanged native trainers, using the retained training cohort. No patient
is reassigned across train/validation/test. Every run exports `cohort_counts.csv`
and `excluded_records.csv`. Original-data failures are also listed globally in
`study/common/outputs/face224_reruns/preprocessing_exclusions.csv`.

## Automatic Queue

```bash
# Validate references and local weights without starting any training.
/root/miniconda3/envs/healthmirrorenv/bin/python -m \
  study.common.rerun_face224 --check-only

# Wait in screen, then train sequential experiment groups on four GPUs.
bash study/common/launch_face224_reruns_screen.sh
screen -r face224_experiments
```

The monitor checks the preprocessing lock and final full-session inventory, not
just disappearance of a screen process. It refuses unfinished processing,
unknown failures, unexpected crop protocols or changes to frozen reference
labels/predictions. It then builds one shared compact index and runs:

| Order | Experiment | New output relative to experiment directory |
| --- | --- | --- |
| 1 | Exp2 face-only regression, 24 h | `outputs/20frame_face224/` |
| 2 | Exp2 face-only binary classification, 24 h | `outputs/face224/` |
| 3 | Exp2 regression, 6 h | `outputs/ablations/lab_match_6h_face224/` |
| 4 | Exp2 regression, 12 h | `outputs/ablations/lab_match_12h_face224/` |
| 5 | Exp6 delta regression, 24 h | `outputs/face224/` |
| 6 | Exp6 delta regression, 6 h | `outputs/ablations/lab_match_6h_face224/` |
| 7 | Exp6 delta regression, 12 h | `outputs/ablations/lab_match_12h_face224/` |
| 8 | Exp8 MST++ NTIRE, 4 x 4 feature grid | `outputs_face224/` |
| 9 | Exp8 MST++ NTIRE, 16 x 16 feature grid | `outputs_grid16_face224/` |

Exp2 keeps the eight current analytes; Exp6 keeps its nine current delta tasks;
Exp8 keeps hemoglobin, total bilirubin and lactate. Exp2 and Exp6 use the same
native models, head32, seeded per-task training, AdamW, loss/weighting, compile
policy and two-stage defaults as their retained baselines: head lr 2e-4, up to
40 epochs/patience 10; fine-tune lr 1e-5, up to 60 epochs/patience 12, unchanged
cosine schedules and augmentation. Four persistent GPU workers dynamically
consume target jobs within each group. The monitor and all groups run inside
one detached screen. Failures are logged and stop subsequent groups.

Exp8 uses the existing verified **NTIRE** checkpoint; no Hyper-Skin checkpoint
is present. MST++ accepts native 224 pixels without resizing back to 128. The
central region stays 75% of each axis (128: [16:112], 224: [28:196]), followed by
the original 31 x grid x grid pooling. Native and legacy spectral caches/results
are separate. The existing Exp8 feature extraction and head training workflow
is preserved; its small three-head experiment uses one GPU, not four duplicate
training jobs. Both existing spatial-grid variants are retained and rerun.

## Results And Logs

Each new run generates its normal result/history figures plus
`figures/face224_vs_legacy_full_test.{png,pdf}` and
`figures/face224_vs_legacy_common_test.{png,pdf}`. The common-test comparison
joins exact patient/video identities (and both videos for Exp6 pairs), validates
raw labels, and recomputes metrics on the same held-out examples. Full-cohort
comparisons are descriptive because cohort sizes can differ. Training cohorts
and fitted scalers can also differ, so performance differences are not solely
attributable to resolution. Native 12/6-hour runs additionally compare against
their native 24-hour counterparts; Exp8 grid16 compares against native grid4.

Global machine-readable comparison: `study/common/outputs/face224_reruns/comparison_all.csv`.
Human-readable result index: `study/common/outputs/face224_reruns/REPORT.md`.
Continuous screen log: `study/common/logs/face224_reruns/run.log`.
Per-experiment logs use the output's relative layout under `logs/` rather than
`outputs/`. No original 128 results are overwritten. All generated caches,
results, logs and large weights stay out of Git.

## Additional Patient-Diverse 30/40 Ablation

`bash study/exp2_face_pretrained_head32_regression/launch_patient_diverse_224_screen.sh`
starts a separate detached monitor named `face224_patient_diverse_30_40`.
It waits for the entire queue above to complete and release its lock, then
uses the native 24-hour main run's records, scalers and shared index. The
original eight-target patient-diverse 30/40 schedule, task seeds, 20 frames
and five views are retained. Existing queue work is not interrupted or reordered.
Outputs and logs are in the experiment's `outputs/ablations/` and
`logs/ablations/`, under `patient_diverse_schedule_30_40_face224`.
Normal result plots and comparisons with both the corresponding legacy
ablation and native-224 main run are generated automatically.

Three further 12h five-fold protocols (regression patient-diverse 30/40,
classification patient-diverse 30/40, and original classification) wait behind
both preceding jobs. See [FACE224_12H_5FOLD.md](FACE224_12H_5FOLD.md) for shared
patient folds, exact stage settings, outputs, and the detached launch command.
