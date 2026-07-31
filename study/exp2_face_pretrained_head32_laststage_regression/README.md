# Exp2 20-Frame Head32 Last-Stage Abnormal-Score Regression

Canonical ID: `exp2_raw_video_20frame_head32_laststage_regression_balanced_split`.

This is the controlled partial-finetuning counterpart of
`exp2_face_pretrained_head64_regression`. It uses the same abnormal-score regression
targets, 24-hour nearest-lab matching, 20 non-adjacent frames, five training views,
patient-disjoint balanced split, learning rates, epoch limits, early stopping,
batching, and four-GPU dynamic scheduler for:

- `hemoglobin_low`
- `po2_low`
- `lactate_high`

The head is `Linear -> LayerNorm -> SiLU -> Dropout -> Linear(1)` with hidden width
32. Stage 1 trains only the head at `2e-4`. Stage 2 trains the head plus only the final
semantic backbone stage at `1e-5`:

- MobileNetV3-Small: `features[9:13]`
- EfficientNet-B0: `features[7:9]`

Earlier backbone stages remain frozen and in eval mode, including their BatchNorm
running statistics. The experiment reuses the completed Head64 source-data and frame
index caches, regenerates task splits deterministically with the same seed, and aborts
unless every task record and split exactly matches the Head64 reference.

Both controlled experiments use `torch.compile(mode="reduce-overhead")`. Successful
completion automatically generates the same regression figures and machine-readable
outputs.

## Start

Normally `monitor_then_launch.sh` starts this after Head64 finishes. Manual entry:

```bash
bash study/exp2_face_pretrained_head32_laststage_regression/launch_screen.sh \
  20frame --overwrite
```

```bash
screen -r exp2_face_pretrained_head32_laststage_regression_20frame
```
