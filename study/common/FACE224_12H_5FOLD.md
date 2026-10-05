# Native 224, 12-Hour Five-Fold Experiments

These are three additional experiments on the eight current Exp2 analytes,
using the completed native-224 12-hour regression cohort. No prior results
or patient splits are overwritten.

| Protocol | Batch policy | Head lr / maximum epochs / patience | Fine-tune lr / maximum epochs / patience |
| --- | --- | --- | --- |
| Regression | Patient-diverse | 1e-4 / 30 / 8 | 3e-6 / 40 / 8 |
| True binary classification | Patient-diverse | 1e-4 / 30 / 8 | 3e-6 / 40 / 8 |
| True binary classification | Original chunked | 2e-4 / 40 / 10 | 1e-5 / 60 / 12 |

Patient-diverse rates have minimums 1e-6 (head) and 1e-7 (fine-tuning).
The original schedule has a minimum of 1e-6 in both stages. All runs retain
EfficientNet-B0 ImageNet weights, a 32-dimensional head, 20 nonadjacent frames,
five training views, original-only validation/test, AdamW with weight decay
1e-4, the original cosine schedules, no warmup, and the existing compile policy.
The 12-hour limit is nearest-lab distance from the original video capture
interval; it is not recomputed from the duration of filtered face crops.

## Shared Folds

`study/common/outputs/face224_12h_5fold/splits/` contains the single shared
fold assignment and 40 task/fold records, reused by all three protocols.
The existing 256-candidate patient-stratified search is reused without changing
its objective: continuous raw-value and abnormal-score distribution distances,
binary positive-rate differences, and video/patient size balance. Selection
does not use model performance. Each task is independently patient-grouped.

In fold k, test is group k, validation is group (k+1) modulo 5, and training
uses the other three groups. Each video/patient is tested exactly once per
task, and no patient occurs in different splits within a model. Each regression
scaler is fitted only on that fold's training records. Classification uses
the unchanged train-only negative/positive weighting policy. Targets and their
raw values, source samples, and clinical binary labels are validated against
the saved source; all three protocols use identical target/fold job seeds.

This is a new five-fold assignment, not reuse of the original single
60/20/20 split. Comparisons between the two binary protocols are paired on
identical OOF identities. Their classification metrics are not directly
compared with regression r/MAE as if they were the same quantity.

## Launch And Outputs

```bash
# Source checks or CPU-only fold preparation, without training.
python -m study.common.run_face224_12h_5fold --check-only
python -m study.common.run_face224_12h_5fold --prepare-only

# Detached monitor: wait for both prior native-224 jobs, then train.
bash study/common/launch_face224_12h_5fold_screen.sh
screen -r face224_12h_5fold
```

The monitor waits for both the original native-224 queue and the appended
patient-diverse 24h ablation to finish and release their locks. An incomplete
stopped predecessor is reported as an error. It does not consume GPU resources
while waiting. The three protocols run sequentially in the table's order;
four persistent GPU workers dynamically take each protocol's 40 jobs.

Relative to the corresponding experiment directory, new results are:

- Regression: `outputs/ablations/lab_match_12h_patient_diverse_schedule_30_40_face224/5fold/`.
- Patient-diverse classification: `outputs/ablations/lab_match_12h_patient_diverse_schedule_30_40_face224/5fold/`.
- Original classification: `outputs/ablations/lab_match_12h_face224/5fold/`.

Each contains `fold_0` through `fold_4`, train/validation metric histories,
checkpoints, per-video predictions, normal fold figures, `oof_predictions.csv`,
`cv_test_metrics.csv`, and `cv_summary.csv` (fold mean/SD and pooled OOF metrics).
Regression includes pooled predicted-versus-measured scatter plots with fitted
lines. Classification includes OOF ROC curves and fold metric/history figures.
Classifier comparisons use grouped bars with fold-SD error bars and pooled OOF
bars, in `study/common/outputs/face224_12h_5fold/figures/`.

All three families share the existing compact FFV1 index; no extra video or
image cache is created. Successfully completed task/fold jobs can be reused
only under the same saved experiment contract; failed or partial jobs are not
marked complete. Global log: `study/common/logs/face224_12h_5fold/run.log`.
Per-protocol logs mirror the result layout under `logs/ablations/`.
