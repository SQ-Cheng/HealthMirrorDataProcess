# Exp2 Pretrained Face Backbones: Head-32

This is an isolated follow-up to `../exp2_face_pretrained`. The data, targets,
pretrained encoders, augmentation, patient splits, optimization, early stopping, and
evaluation are unchanged. The only model change is reducing the replacement head's
hidden width from 256 to 32:

```text
Linear(backbone_features -> 32) -> LayerNorm -> SiLU -> Dropout(0.25) -> Linear(32 -> 1)
```

This experiment trains one independent binary classifier for every usable target and
backbone combination:

- ResNet18 (ImageNet-1K V1)
- MobileNetV3-Small (ImageNet-1K V1)
- EfficientNet-B0 (ImageNet-1K V1)

It reuses the corrected native RGB cache in `../exp2_face_only/outputs_aug20_24h`.
Frames are resized from `128x128` to `224x224` at load time and normalized with the
ImageNet mean and standard deviation.

For each target, conflicting videos are removed for that target and repeated copies of
the same video/frame are deduplicated. Splits are patient-level 60/20/20 and shared by
all three architectures. Metrics and model selection are video-level after averaging
the 20 frame probabilities.

Training has two stages:

1. Freeze the encoder and train a replacement single-task head at `1e-3`.
2. Unfreeze all parameters and fine-tune at `1e-4`.

Both stages use early stopping and write train/validation loss, balanced accuracy, and
ROC-AUC after every epoch.

## Start in screen

```bash
bash study/exp2_face_pretrained_head32/launch_screen.sh
```

Attach with:

```bash
screen -r exp2_face_pretrained_head32
```

The experiment reuses the verified local checkpoints and weight manifest from
`../exp2_face_pretrained/pretrained_weights`. Results and logs are written only under
this directory, so the parent experiment is never overwritten.

## Automatic handoff

`monitor_then_launch.sh` waits for the current `exp2_face_pretrained` screen to exit,
checks that every architecture/target job in its manifest completed successfully and
has all required artifacts, and then starts this head-32 experiment. It refuses to
launch after an incomplete or failed parent run.

```bash
screen -dmS exp2_head32_handoff_monitor \
  bash study/exp2_face_pretrained_head32/monitor_then_launch.sh
```

Monitor progress in `logs/handoff_monitor.log`.

## Useful test entry

```bash
python -m study.exp2_face_pretrained_head32.run_all \
  --smoke-test --output-dir /tmp/exp2_face_pretrained_head32_smoke --overwrite
```
