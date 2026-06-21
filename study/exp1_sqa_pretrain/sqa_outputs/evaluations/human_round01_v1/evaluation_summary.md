# SQA human round 01 evaluation

## Scope

This evaluation version is isolated from the original weak-model results. It evaluates
the TCN and ResNet `round01_human_v1` best checkpoints on:

- the patient-disjoint manually annotated splits produced during fine-tuning;
- 7,587 unlabeled windows from 2,592 uncleaned raw patient files;
- controlled Gaussian, baseline, impulse, clipping, and dropout corruptions;
- the same raw windows used by the original weak-model evaluation.

Raw ECG has no human quality labels. Raw-data results describe score behavior,
model agreement, artifact association, and corruption sensitivity—not accuracy.

## Manually labeled held-out windows

| Model | Split | Task | Before AUROC | After AUROC | Before Brier | After Brier | After balanced accuracy |
|---|---|---|---:|---:|---:|---:|---:|
| TCN | random | QRS | 0.250 | 1.000 | 0.522 | 0.025 | 1.000 |
| TCN | random | Morph | 0.250 | 1.000 | 0.561 | 0.025 | 0.958 |
| TCN | challenge | QRS | 0.333 | 1.000 | 0.516 | 0.024 | 0.944 |
| TCN | challenge | Morph | 0.333 | 1.000 | 0.560 | 0.027 | 0.944 |
| ResNet | random | QRS | 1.000 | 1.000 | 0.442 | 0.042 | 0.500 |
| ResNet | random | Morph | 0.833 | 1.000 | 0.497 | 0.042 | 0.500 |
| ResNet | challenge | QRS | 0.778 | 1.000 | 0.371 | 0.025 | 0.500 |
| ResNet | challenge | Morph | 0.778 | 1.000 | 0.408 | 0.024 | 0.500 |

The held-out sets are very small (13 random and 10 challenge usable unique windows,
with very few positives), so AUROC/AUPRC estimates are unstable. ResNet ranks the
held-out samples correctly and has much lower Brier error, but the validation-selected
threshold misses the rare positive cases, producing balanced accuracy 0.5. Threshold
calibration requires more labeled positives.

## Behavior on uncleaned raw ECG

| Model | Task | Weak mean | Human mean | Human accept ≥0.5 | Human accept ≥0.8 |
|---|---|---:|---:|---:|---:|
| TCN | QRS | 0.745 | 0.425 | 0.427 | 0.160 |
| TCN | Morph | 0.781 | 0.426 | 0.430 | 0.148 |
| ResNet | QRS | 0.802 | 0.293 | 0.270 | 0.106 |
| ResNet | Morph | 0.817 | 0.318 | 0.298 | 0.122 |

Human fine-tuning makes both models substantially more conservative. This is consistent
with the annotation batch containing many targeted bad samples, but these acceptance
rates are not population accuracy estimates.

Cross-encoder Pearson agreement improves from 0.202 to 0.876 for QRS and from 0.157
to 0.882 for morphology. The mean absolute probability gap nevertheless increases,
showing that ranking agreement improved more than calibration agreement.

Artifact-burden association becomes more appropriately negative: approximately
-0.45 for TCN and -0.43 for ResNet after fine-tuning, compared with -0.06 to -0.34
before fine-tuning.

## Controlled corruptions

Results are mixed:

- TCN now decreases strongly for baseline drift and impulse artifacts.
- ResNet and TCN respond more appropriately to some dropout/impulse cases.
- Severe Gaussian noise still raises both models' scores.
- Severe clipping raises both fine-tuned models' scores, whereas the weak ResNet had
  decreased appropriately.
- ResNet impulse response remains inappropriate, and ResNet morphology still rises
  under severe baseline corruption.

The first human round therefore improves real-sample discrimination and cross-encoder
ranking consistency, but does not solve out-of-distribution corruption behavior.
Future annotation should add explicit Gaussian/high-frequency noise and clipping
examples, plus more positive held-out samples for threshold calibration.

## Directory layout

```text
human_round01_v1/
├── manifest.json
├── evaluation_summary.md
├── labeled/
│   ├── before_after_metrics.csv
│   ├── category_changes.csv
│   └── heldout_before_after.png
├── raw_unlabeled/
│   ├── evaluation_report.md
│   ├── window_predictions.csv
│   ├── perturbation_summary.csv
│   └── *.png
└── comparison_to_weak/
    ├── score_changes.csv
    ├── cross_encoder_agreement.csv
    ├── perturbation_comparison.csv
    └── *.png
```

The original weak evaluation remains unchanged at `sqa_outputs/unlabeled_evaluation/`.
