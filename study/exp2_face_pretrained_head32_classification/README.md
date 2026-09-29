# Exp2 face-only Head32 classification

True binary counterpart of `exp2_face_pretrained_head32_regression`. Data,
patient splits, 20-frame policy, five views, EfficientNet-B0 Head32 model, and
two-stage optimization match the regression experiment. The only learning-task
change is one-logit binary classification with weighted BCE.

Troponin I is replaced by total bilirubin >21 umol/L.

## Patient-diverse 30/40 ablation

`outputs/ablations/patient_diverse_schedule_30_40/` retains the main binary
experiment's eight labels, exact patient split, 20-frame cache, five views,
EfficientNet-B0 Head32 model, seed, train-only positive-class weighting,
two-stage full-backbone protocol, validation bACC selection, and test protocol.
The train sampler groups four source frames per video and mixes up to 12
patients in each 48-source-frame batch, without dropping frames.

The head stage uses `lr=1e-4`, cosine floor `1e-6`, at most 30 epochs, and
patience 8. The full-backbone stage uses `lr=3e-6`, cosine floor `1e-7`, at most
40 epochs, and patience 8. AdamW weight decay and all other settings match
the binary baseline. These match the resolved hyperparameters of the
regression `patient_diverse_schedule_30_40` ablation.

Launch in a detached four-GPU screen with:

```bash
bash study/exp2_face_pretrained_head32_classification/launch_patient_diverse_schedule_screen.sh
```

After all eight jobs, the runner automatically writes a baseline comparison
CSV and 4-by-2 test-metric and validation-history figures in the variant's
`figures/` directory. Existing binary results are not overwritten.

## Lab/video matching-window ablations

The 12-hour and 6-hour binary variants use the completed regression
matching-window cohorts and their independently searched, patient-disjoint
splits. Their videos and labels are exact time-filtered subsets of the 24-hour
binary source. All eight targets use the unchanged EfficientNet-B0 Head32
classifier, 20 indexed frames, five training views, two-stage learning rates,
batch policy, early stopping and per-task seed. Each variant recalculates
`pos_weight` from its own training videos. Results stay in
`outputs/ablations/lab_match_12h/` and `lab_match_6h/`.

Run once to attach a detached screen. It waits until both 12-hour and 6-hour
regression patient-diverse 30/40 tasks and their comparison figures are
complete, then trains the two classification variants sequentially on four
GPUs. The log is `logs/ablations/match_windows/run.log`.

```bash
bash study/exp2_face_pretrained_head32_classification/launch_match_window_monitor_screen.sh
```

Each completed variant automatically writes `full_test_comparison.csv` and
`shared_test_comparison.csv` plus bACC/AUROC bar charts under `figures/`.
Full-cohort comparisons have different test sets and are descriptive; the
shared-test analysis compares predictions for the exact same held-out videos.
