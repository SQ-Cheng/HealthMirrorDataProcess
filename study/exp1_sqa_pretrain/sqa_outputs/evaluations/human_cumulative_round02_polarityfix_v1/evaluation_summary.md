# SQA cumulative human round 02 evaluation

## Scope

The TCN and ResNet Window heads were retrained from the original weakly supervised checkpoints using the cumulative round01+round02 annotations. Raw ECG is polarity-flipped (`-1`); cleaned anchors retain their stored polarity (`+1`). Both encoders were frozen and verified byte-for-byte unchanged.

After cross-round duplicate consolidation, 336 definite samples were usable: 210 train, 55 validation, 42 random test, and 29 challenge test. Two consistently labeled cross-round duplicates were merged. Uncertain labels were excluded from the supervised loss.

## Patient-disjoint manually labeled evaluation

| Model | Split | Task | N | Before AUROC | After AUROC | Before Brier | After Brier | After balanced accuracy |
|---|---|---|---:|---:|---:|---:|---:|---:|
| TCN | random | QRS | 42 | 0.975 | 0.991 | 0.306 | 0.031 | 0.957 |
| TCN | random | MORPH | 42 | 0.984 | 0.991 | 0.329 | 0.032 | 0.957 |
| TCN | challenge | QRS | 29 | 0.978 | 1.000 | 0.372 | 0.015 | 1.000 |
| TCN | challenge | MORPH | 29 | 1.000 | 1.000 | 0.403 | 0.014 | 1.000 |
| RESNET | random | QRS | 42 | 0.934 | 0.998 | 0.257 | 0.030 | 0.957 |
| RESNET | random | MORPH | 42 | 0.963 | 1.000 | 0.270 | 0.028 | 0.957 |
| RESNET | challenge | QRS | 29 | 0.917 | 0.978 | 0.356 | 0.054 | 0.919 |
| RESNET | challenge | MORPH | 29 | 0.950 | 0.978 | 0.385 | 0.056 | 0.919 |

Both models improve ranking and calibration on both held-out subsets. These test samples were actively selected rather than population-random, so they measure performance on informative/challenging cases, not deployment prevalence.

## Unlabeled raw ECG behavior

Evaluated 7,587 polarity-corrected raw windows from 2,592 patient files. Since these windows have no human labels, the following values are score behavior rather than accuracy.

| Model | Task | Weak mean | Human mean | Weak accept ≥0.8 | Human accept ≥0.8 |
|---|---|---:|---:|---:|---:|
| TCN | QRS | 0.865 | 0.496 | 0.800 | 0.371 |
| TCN | MORPH | 0.886 | 0.496 | 0.855 | 0.370 |
| RESNET | QRS | 0.850 | 0.422 | 0.815 | 0.288 |
| RESNET | MORPH | 0.870 | 0.430 | 0.861 | 0.294 |

TCN-vs-ResNet Pearson agreement rises from 0.546 to 0.905 for QRS and from 0.515 to 0.915 for morphology. The fine-tuned models are substantially less saturated at high scores and retain a broad score distribution.

## Controlled severe corruptions

Values are the mean probability change after adding corruption; negative is the desired direction for severe corruption.

| Model | Task | Corruption | Weak Δp | Human-r02 Δp |
|---|---|---|---:|---:|
| TCN | QRS | gaussian | -0.003 | -0.293 |
| TCN | QRS | baseline | +0.007 | -0.165 |
| TCN | QRS | impulse | -0.228 | -0.469 |
| TCN | QRS | clipping | -0.007 | -0.107 |
| TCN | QRS | dropout | -0.004 | +0.130 |
| TCN | MORPH | gaussian | +0.002 | -0.277 |
| TCN | MORPH | baseline | -0.004 | -0.168 |
| TCN | MORPH | impulse | -0.218 | -0.470 |
| TCN | MORPH | clipping | -0.003 | -0.110 |
| TCN | MORPH | dropout | -0.003 | +0.135 |
| RESNET | QRS | gaussian | +0.015 | -0.139 |
| RESNET | QRS | baseline | +0.065 | -0.079 |
| RESNET | QRS | impulse | +0.015 | -0.009 |
| RESNET | QRS | clipping | -0.103 | -0.117 |
| RESNET | QRS | dropout | -0.078 | -0.032 |
| RESNET | MORPH | gaussian | +0.004 | -0.136 |
| RESNET | MORPH | baseline | +0.059 | -0.038 |
| RESNET | MORPH | impulse | +0.010 | -0.010 |
| RESNET | MORPH | clipping | -0.098 | -0.115 |
| RESNET | MORPH | dropout | -0.083 | -0.063 |

Round02 fixes the main Gaussian/high-frequency and clipping failure modes: both models now lower scores under severe Gaussian noise and clipping. Baseline response also becomes negative. TCN strongly rejects impulses; ResNet's impulse response remains weak. Severe dropout remains a known TCN failure because its score increases by about 0.13, while ResNet decreases appropriately.

The natural-data artifact-burden correlations remain negative (approximately -0.45 for both models). Controlled perturbations are synthetic sensitivity tests and must not be interpreted as clinical accuracy.

## Outputs

- `labeled/`: patient-disjoint before/after metrics, per-category behavior, candidate predictions, and held-out visualization.
- `raw_unlabeled/`: raw-window predictions, natural distributions, agreement, artifact associations, controlled perturbations, report, and plots.
- `comparison_to_weak/`: paired weak-vs-human natural score, agreement, and perturbation tables.
- `manifest.json`: checkpoint hashes, annotation counts, polarity, and evaluation protocol.
