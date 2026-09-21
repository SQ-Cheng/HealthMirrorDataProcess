# Exp2 face-only Head32 regression with SimCLR initialization

Controlled ablation of `study/exp2_face_pretrained_head32_regression` (20-frame
variant). The records, nearest-lab labels, patient-disjoint split, robust
scalers, frame-offset index, EfficientNet-B0 model, Head32 regression head,
five supervised-training views, optimization, and evaluation are reused
without rebuilding data.

## Three training stages

1. **SimCLR encoder adaptation**: start from local ImageNet EfficientNet-B0
   weights. Each pair uses two distinct selected frames and two distinct views
   from the same patient video. Other videos from the same patient are excluded
   from the NT-Xent denominator, so they are not false negatives. Only training
   split videos are visible to this stage. The temporary 256-to-128 projection
   MLP is discarded afterward.
2. **Head training**: freeze the SimCLR-initialized encoder and train the same
   32-dimensional scalar regression head as the baseline.
3. **Full fine-tuning**: unfreeze the complete encoder and head and select the
   best validation-MAE checkpoint using the baseline trainer.

Learning rates are `1e-5` for the SimCLR backbone, `2e-4` for its temporary
projector, `2e-4` for frozen-backbone head training, and `1e-5` for full
fine-tuning. SimCLR uses a 20-epoch cosine schedule to 10% of each parameter
group's initial learning rate and temperature 0.10.

This strict ablation keeps the baseline regression targets, including
Troponin I. The separate true-binary experiments replace Troponin I with total
bilirubin, but that target change is not applied here because it would confound
the SimCLR comparison.

## Entrypoints

Direct detached four-GPU run:

```bash
bash study/exp2_face_pretrained_head32_regression_simclr/launch_screen.sh
```

Wait for the current `exp2_three_binary` screen, run a one-target bounded smoke
test, discard only the smoke outputs, then launch the formal four-GPU run:

```bash
bash study/exp2_face_pretrained_head32_regression_simclr/monitor_then_launch.sh
```

Attach to the automated chain with `screen -r exp2_simclr_monitor`. Formal
training is logged to `logs/run.log`. On success, `outputs/figures` contains
the standard regression plots, SimCLR histories, and
`simclr_vs_original_regression.png`; the machine-readable comparison is
`outputs/simclr_vs_baseline.csv`.
