# Sequential History Baseline + Face Residual Regression

Ten independent task pipelines, using the latest complete merged lab table:
oxyhemoglobin fraction, lactate, urea, total bilirubin, platelets, hemoglobin,
A/a PO2 ratio, creatinine, hematocrit and directly reported creatinine eGFR.
Existing experiments are never overwritten.

## Data

- Reuse the current ten-task face-only main experiment's exact patient-disjoint
  splits, labels, 24h matching and native224 FFV1 20-frame index. Verify the lab
  fingerprint and every raw label before training; stale references fail closed.
- Rebuild all ten same-analyte histories from the latest canonical lab sources.
  Only measurements strictly before the current lab time and in the same unique
  hospitalization are used. No sequence truncation; empty histories are retained.
- Features are train-only robust-scaled values and
  `-log1p((current_lab_time-history_time)/24 hours)`.
- The target-time history rule follows the original experiment. Some history
  results can be later than video acquisition; this is not a prospective forecast.
- Source whitelists and quality rules remain aligned with the audited primary
  experiment; the new experiment does not silently change field definitions.

## Training

1. Train a full history-only regressor for every target. Encoder:
   `2 -> 16 -> 16 + LayerNorm + masked mean` (352 parameters); head:
   `16 -> 32 + LayerNorm + SiLU + Dropout(0.25) -> 1` (641 parameters).
   Both LR stages train all 993 parameters; there is no pretrained history encoder.
2. Within each outer training split, run three patient-grouped cross-fit folds.
   Each fold fits its scaler on its inner training patients and uses a disjoint
   inner validation set for stopping. Outer val/test patients never enter these
   folds. Held-out fold predictions define every training residual exactly once.
3. Freeze the history baseline. Train an ImageNet EfficientNet-B0 with a width-32
   scalar head to predict `(raw_lab - history_prediction) / original_train_IQR`.
   Training uses OOF history predictions; outer val/test use the full history model.
   Reported prediction is `history_prediction + mean(face_residual_predictions)`.

All full/OOF history jobs finish before any face training starts. Four GPU workers
dynamically take independent targets; one training process per GPU. Both history
and face use AdamW, weight decay `1e-4`, SmoothL1 beta `0.5`, cosine schedules:
`2e-4`, max 40 epochs, patience 10; then `1e-5`, max 60, patience 12; floor `1e-6`.
For face training the first stage freezes the backbone, the second unfreezes all
parameters. Compile is retained for face training, not the tiny history MLPs.
Face compilation uses default-mode Inductor, dynamic batch sizes and no CUDA
Graphs; evaluation uses the original eager model. Compilation caches are cleared
at both ends of each independent face job to prevent cross-task retention.
Each full face batch contains 12 distinct clinical events, 20 frames and one view
per selected video: 240 frame-level losses. All five views are covered per epoch:
original, horizontal flip, center crop, brightness and contrast.

## Run

```bash
bash study/exp2_face_history_residual_head32_regression/launch_screen.sh
```

Preparation alone: `python -m study.exp2_face_history_residual_head32_regression.run --prepare-only`.
Launch already prepared data with `launch_screen.sh --train-prepared`; the
source table and reference split hashes must be unchanged and training must
not already have started.
To overwrite and restart every history/OOF/face model while keeping the exact
prepared inputs, use `launch_screen.sh --train-prepared --overwrite`. Stop the
previous screen and its training processes first. Only generated training
results are removed; labels, splits, scalers, histories and shared videos/indexes
are retained.
End-to-end smoke test: add `--smoke-test --targets hemoglobin_low --devices 0`.
The smoke run uses `outputs_smoke/`, 1+1 epochs, no compile and one face batch.
Unit tests: `python -m unittest study.exp2_face_history_residual_head32_regression.test_experiment`.

## Outputs

- `logs/run.log`: continuous stage/epoch/progress log.
- `outputs/task_records/`, `history_records/`, `residual_records/`: data and leakage audit.
- `outputs/runs/<target>/history_full/`, `history_oof/fold1..3/`, `face_residual/`:
  all checkpoints, histories and predictions. A combined deployment checkpoint
  is saved at `outputs/runs/<target>/model.pt`.
- `face_residual/residual_*predictions.*` contain residual-only outputs;
  `face_residual/video_predictions.csv` contains composed raw-unit predictions.
  Face training history metrics describe the composed prediction; losses are
  residual losses. Train history baselines are OOF diagnostics, not full-fit values.
- `outputs/figures/`: automatic PNG-only metric bar charts, predicted/true and
  residual scatter plots with fitted lines, and training histories with a separate
  correlation scale in the same panel. Ten-target figures use 5 columns x 2 rows.
- `outputs/tables/`: matched test comparison with the newly trained history
  baseline and current face-only model; `outputs/REPORT.md` documents results.
  Old eight-task face+history results are not a matched control: their lab source
  and cohort differ. No figures compare unmatched aggregate metrics.

Existing cached videos are streamed; no duplicated image/video cache is created.

`models.HistoryFaceResidualRegressor.load(path)` loads the combined checkpoint
for raw-unit inference. It expects ImageNet-normalized RGB frames and history
features built using the saved full-fit lab scaler; aggregate its raw outputs
across the twenty original frames. No history retraining occurs during face fitting.
